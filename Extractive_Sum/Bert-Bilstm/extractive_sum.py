from rouge import Rouge
import torch
import torch.nn as nn
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from textrank4zh import TextRank4Sentence, util
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
import json
import copy
from model_processors import Summarizer


import sys
sys.setrecursionlimit(1000000) #修改递归最大深度
class myDataset(Dataset.Dataset):
    def __init__(self, path):
        super(myDataset, self).__init__()
        self.path = path
        with open(self.path, 'r', encoding='utf-8') as file:
            self.json_dict = json.load(file)

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, index):
        content = self.json_dict[index]['content']
        summary = self.json_dict[index]['summary']
        return content, summary


class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)
        # self.hidden_size = hidden_size

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros((input.size(0), self.hidden_size), requires_grad=False)
            cx = input.new_zeros((input.size(0), self.hidden_size), requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LayerNormLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.linear = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, bias=bias)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ])

    def forward(self, input : torch.Tensor, hidden=None):
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros((self.num_layers * num_directions, batch_size, self.hidden_size), requires_grad=False)
            cx = input.new_zeros((self.num_layers * num_directions, batch_size, self.hidden_size), requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None for i in range((self.num_layers * num_directions))] for _ in range(seq_len)]
        ct = [[None for i in range((self.num_layers * num_directions))] for _ in range(seq_len)]

        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])


        y = y.transpose(0,1).contiguous()
        y = self.sigmoid(self.linear(y))
        y = y.squeeze(-1)

        return y, (hy, cy)


class LinearFowardPrediction(nn.Module):
    def __init__(self):
        super(LinearFowardPrediction, self).__init__()
        self.linear = nn.Linear(2048, 1)
        # self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        # x = self.dropout(x) + x
        # x = self.linear(x)
        sent_scores = self.sigmoid(self.linear(x))

        return sent_scores


def sent_segmentation(text):
    delimiters = util.sentence_delimiters
    delimiters = set([util.as_text(item) for item in delimiters])
    res = [util.as_text(text)]
    util.debug(text)
    util.debug(delimiters)
    for sep in delimiters:
        text, res = res, []
        for seq in text:
            res += seq.split(sep)
    res = [s.strip() for s in res if len(s.strip()) > 0]
    return res


def sent_sortion(source_list, target_list):
    res = copy.deepcopy(target_list)
    for idx in range(batch_size):
        id_list = [0 for i in range(len(target_list[idx]))]
        for index, sentence in enumerate(source_list[idx]):
            for j in range(len(target_list[idx])):
                if sentence.encode('utf-8').decode('utf-8') == target_list[idx][j].encode('utf-8').decode('utf-8'):
                    id_list[j] = index
        for i, id in enumerate(id_list):
            j = i + 1
            while j < len(id_list):
                if id > id_list[j]:
                    temp = id_list[j]
                    id_list[j] = id
                    id = temp
                    temp = res[idx][j]
                    res[idx][j] = res[idx][i]
                    res[idx][i] = temp
                j += 1
    return res


def length_normalization(target, aux, position_list, norm_len):
    res = target.copy()
    new_position = position_list.copy()
    for index in range(batch_size):
        length = len(tokenizer.tokenize("".join(target[index])))
        # print("primary length: ",length)
        if length >= norm_len + 50:
            while True:
                if len(tokenizer.tokenize("".join(res[index]))) >= norm_len + 50:
                    res[index].pop()
                    for pos in range(len(position_list[index]) - 1, -1, -1):
                        if position_list[index][pos] == 1:
                            new_position[index][pos] = 0
                            break
                else:
                    break
        elif length < norm_len - 50:
            for sent_id, sent in enumerate(aux[index]):
                flag = True
                if len(tokenizer.tokenize("".join(res[index]))) >= norm_len - 50:
                    break
                for collected_sent in res[index]:
                    if collected_sent.encode('utf-8').decode('utf-8') == sent.encode('utf-8').decode('utf-8'):
                        flag = False
                if flag:
                    res[index].append(sent)
                    new_position[index][sent_id] = 1
        else:
            pass
    return res, new_position


def etractive_bert_data_process(ctnt_sent_processed):
    sent_combine_list = [[] for i in range(batch_size)]
    sent_segment_id_list = [[] for i in range(batch_size)]
    cls_position_list = [[] for i in range(batch_size)]
    temp = ""
    ids = []
    cls_pos = []
    for ii in range(batch_size):
        index = 0
        while index < len(ctnt_sent_processed[ii]):
            ll = len(tokenizer.tokenize(ctnt_sent_processed[ii][index]))
            if ll > 512 :
                temp_var = tokenizer.tokenize(ctnt_sent_processed[ii][index])[0:500]
                temp_var.insert(-1,'[SEP]')
                # print(temp_var)
                ctnt_sent_processed[ii][index] = "".join(temp_var)
            ll = len(tokenizer.tokenize(ctnt_sent_processed[ii][index]))
            # print("ll",ll)
            if len(tokenizer.tokenize(temp)) + ll > 512:
                sent_combine_list[ii].append(temp)
                sent_segment_id_list[ii].append(ids)
                cls_position_list[ii].append(cls_pos)
                cls_pos = []
                ids = []
                temp = ""
                index -= 1
            else:
                if len(ids) == 0:
                    ids.extend([0 for i in range(len(tokenizer.tokenize(ctnt_sent_processed[ii][index])))])
                else:
                    if ids[-1] == 1:
                        ids.extend([0 for i in range(len(tokenizer.tokenize(ctnt_sent_processed[ii][index])))])
                    else:
                        ids.extend([1 for i in range(len(tokenizer.tokenize(ctnt_sent_processed[ii][index])))])
                cls_pos.append(len(tokenizer.tokenize(temp)))
                temp += ctnt_sent_processed[ii][index]
            index += 1
            # len_of_id = len(ids)
        if len(temp.strip()) != 0 and ids is not None:
            sent_combine_list[ii].append(temp)
            # len_of_id_1 = len(ids)
            # print("leng of id ",len_of_id_1)
            sent_segment_id_list[ii].append(ids)
            cls_position_list[ii].append(cls_pos)
            cls_pos = []
            ids = []
            temp = ""
    return sent_combine_list, sent_segment_id_list, cls_position_list


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model_pretrained = None
if bert_model_pretrained != None:
    model = BertModel.from_pretrained(bert_model_pretrained)
else:
    model = BertModel.from_pretrained('bert-base-chinese')
dataset = myDataset("./train_news.json")
testdataset = myDataset("./eval_news.json")
batch_size = 1
dataloader = Dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
testdataloader = Dataloader.DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=0)
rouge = Rouge()
# lstmModel = nn.LSTM(input_size=768, hidden_size=1024, batch_first=False, bidirectional=True)
lstmModel = LayerNormLSTM(input_size=768, hidden_size=1024, num_layers=2, bidirectional=True)
# LFPModel = LinearFowardPrediction()
loss_func = nn.BCELoss()
model = model.to('cuda:0')
lstmModel = lstmModel.to('cuda:0')
# LFPModel = LFPModel.to('cuda:0')
loss_func = loss_func.to('cuda:0')
optimizer = torch.optim.Adam(lstmModel.parameters(), lr=2e-4)
# bert_optim = torch.optim.Adam(model.parameters(),lr=2e-6)
epochs = 1
total_step = 355
scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps=0.1*total_step,t_total=total_step)
cluster_extract_model = Summarizer()

def get_extractive_sum(dataloader, epochs, train: bool, pretrained_model=None, batch_size=4):
    # if not train:
    #     judge_file = open('./judge.json', 'w', encoding='utf-8')
    #     print('[', file=judge_file)
    #     judge_file.close()
    tr4s = TextRank4Sentence()
    for epoch in range(epochs):

        for batch_id, item in enumerate(dataloader):

            content, summary = item
            extractive_sent_list = [[] for i in range(batch_size)]
            content_sent_list = []
            summary_sent_list = []
            for bsz_id in range(batch_size):
                # textrank_sent_list = []
                # if len(sent_segmentation(content[bsz_id])) > 10:
                #     tr4s.analyze(text=content[bsz_id],lower=True,source='all-filters')
                #     for item in tr4s.get_key_sentences(num=10,sentence_min_len=1):
                #         # print(item)
                #         textrank_sent_list.append(item['sentence'])
                #     content_sent_list.append(textrank_sent_list)
                # else:
                content_sent_list.append(sent_segmentation(content[bsz_id]))
                summary_sent_list.append(sent_segmentation(summary[bsz_id]))


            # print("content_sent_list",content_sent_list)
            # # print(summary_sent_list)
            # print("len of content sent list",len(content_sent_list[0]))

            max_score_sent = ""
            extractive_pos_label = []
            for i in range(batch_size):
                extractive_pos_label.append([0 for xx in range(len(content_sent_list[i]))])
            # print("num of sentences : {} and {}".format(len(content_sent_list[0]),len(content_sent_list[1])) )
            for b_id, sum_sent_batch in enumerate(summary_sent_list):
                # if b_id == 2:
                #     print("sum_sent_batch",sum_sent_batch)
                #     print("content_sent:",content_sent_list[b_id])
                #     raise ("asd")
                for sum_sent in sum_sent_batch:
                    max_score = 0
                    sum_token = " ".join(tokenizer.tokenize(sum_sent))
                    if len(sum_token) == 0:
                        continue
                    sent_position = 0
                    flag = True
                    count = 0
                    aux_dict = {}
                    for sent_id, ctnt_sent in enumerate(content_sent_list[b_id]):
                        ctnt_token = " ".join(tokenizer.tokenize(ctnt_sent))
                        aux_para_list = []
                        # print("ctnt_token:",ctnt_token)
                        # print("b_id",b_id)
                        # print("sum_token:",sum_token)
                        if len(ctnt_token) == 0:
                            continue
                        else:
                            rouge_score = rouge.get_scores(ctnt_token, sum_token)

                        if rouge_score[0]["rouge-l"]['f'] > max_score:
                            max_score = rouge_score[0]["rouge-l"]['f']
                            max_score_sent = ctnt_sent
                            sent_position = sent_id
                            aux_para_list.append(max_score_sent)
                            aux_para_list.append(sent_position)
                            aux_dict[max_score] = aux_para_list

                    for time in range(len(aux_dict)):
                        if count >= 3:
                            break
                        cur_max_list = aux_dict.popitem()[1]
                        cur_max_score_sent = cur_max_list[0]
                        cur_sent_position = cur_max_list[1]
                        for collected_sent in extractive_sent_list[b_id]:
                            if collected_sent.encode('utf-8').decode('utf-8') == cur_max_score_sent.encode('utf-8').decode(
                                    'utf-8'):
                                flag = False
                        if flag:
                            extractive_sent_list[b_id].append(cur_max_score_sent)
                            # print("sent position",sent_position)
                            # print("len of extra label",len(extractive_pos_label[b_id]))
                            extractive_pos_label[b_id][cur_sent_position] = 1
                            count += 1

            extractive_sent_list = sent_sortion(content_sent_list, extractive_sent_list)

            ctnt_sent_processed = [[] for i in range(batch_size)]
            for i in range(batch_size):

                for sent in content_sent_list[i]:
                    length = len(tokenizer.tokenize(sent))
                    if length < 510:
                        temp_sent = '[CLS]' + sent + '[SEP]'
                        ctnt_sent_processed[i].append(temp_sent)
                    else:
                        temp_sent = tokenizer.tokenize(sent)
                        temp_sent = temp_sent[:509]
                        temp_sent = "".join(temp_sent)
                        temp_sent = '[CLS]' + temp_sent + '[SEP]'
                        ctnt_sent_processed[i].append(temp_sent)

            cmb_res, seg_id_res, cls_pos_res = etractive_bert_data_process(ctnt_sent_processed)
            # print("len of cmb_res:",len(cmb_res))

            combine_feature = []
            max_sent_num = 0
            # print(seg_id_res)
            for seq_batch, id_batch, cls_pos_batch in zip(cmb_res, seg_id_res, cls_pos_res):
                # print(len(id_batch[1]))
                features = []

                for seq, id, cls_pos in zip(seq_batch, id_batch, cls_pos_batch):

                    # print(len(id))
                    single_feature = torch.Tensor()
                    seq_token = tokenizer.tokenize(seq)

                    seq_input_ids = tokenizer.convert_tokens_to_ids(seq_token)
                    seq_input_ids_tensor = torch.tensor(seq_input_ids).long().unsqueeze(0)
                    segment_id = id
                    # print(segment_id)
                    segment_id_tensor = torch.tensor(segment_id).long()
                    seq_input_ids_tensor = seq_input_ids_tensor.to('cuda:0')
                    segment_id_tensor = segment_id_tensor.to('cuda:0')
                    # print(seq_input_ids_tensor.size(),segment_id_tensor.size())
                    # print("seq_inputs_ids_tensor",seq_input_ids_tensor)
                    # print("segment_ids",segment_id_tensor)

                    output = model(seq_input_ids_tensor, token_type_ids=segment_id_tensor)
                    output_clone = output[0].clone().detach().requires_grad_(True)

                    for position in cls_pos:

                        if len(single_feature) == 0:

                            single_feature = output_clone[:, position, :]

                        else:

                            single_feature = torch.cat((single_feature, output_clone[:, position, :]), dim=1)

                    single_feature = single_feature.view(1, -1, 768)

                    # print(single_feature.size())
                    features.append(single_feature)

                concat_feature = torch.Tensor()

                for feature in features:

                    if len(concat_feature) == 0:

                        concat_feature = feature
                    else:

                        concat_feature = torch.cat((concat_feature, feature), dim=1)
                # print("sent_num = ", concat_feature.size(1))
                if concat_feature.size(1) > max_sent_num:
                    max_sent_num = concat_feature.size(1)
                    # print("max_sent_num: ",max_sent_num)

                combine_feature.append(concat_feature)

            final_feature = torch.Tensor()
            # print("len of combine", len(combine_feature))
            for feature in combine_feature:

                if max_sent_num != feature.size(1):

                    exp_dim = max_sent_num - feature.size(1)
                    exp_tensor = torch.zeros([1, exp_dim, 768])
                    exp_tensor = exp_tensor.to('cuda:0')
                    temp = feature.clone().detach()
                    temp = torch.cat((temp, exp_tensor), dim=1)

                else:

                    temp = feature.clone().detach()

                if len(final_feature) == 0:

                    final_feature = temp
                else:

                    final_feature = torch.cat((final_feature, temp), dim=0)

            # print("final:", final_feature.size())
            # features = features.view(batch_size, -1, 768)
            label = []
            for single_label in extractive_pos_label:
                if len(single_label) == max_sent_num:
                    label.append(single_label)

                else:
                    single_label.extend([0 for i in range(max_sent_num - len(single_label))])
                    label.append(single_label)
            label = np.array(label)
            # print(label)
            # label = np.array(label)
            label = torch.from_numpy(label).long()
            # print("label:", label.size())
            if pretrained_model is not None:
                lstmModel.load_state_dict(torch.load(pretrained_model))
            final_feature = final_feature.to('cuda:0')
            label = label.to('cuda:0')

            final_feature = final_feature.transpose(0, 1).contiguous()

            #======================================TRAIN=================================================
            if train:
                output, _ = lstmModel(final_feature) # output.size = (batch_size,seq_len)
                label = label.float()
                extractive_position = output.clone().detach()
                # print("output.size:",output.size())
                for batch in range(batch_size):
                    for i in range(max_sent_num):
                        if output[batch][i] >= 0.5:
                            extractive_position[batch][i] = 1
                        else:
                            extractive_position[batch][i] = 0
                extractive_position = extractive_position.cpu()
                extractive_position = extractive_position.numpy()
                extractive_position = extractive_position.tolist()
                # print("extractive_position,",extractive_position)

                for index in range(batch_size):
                    extractive_result = []
                    label_result = []
                    # print("len of content sent list,",len(content_sent_list[index]))
                    extractive_position[index] = extractive_position[index][:len(content_sent_list[index])]
                    label_position = label[index][:len(content_sent_list[index])]
                    for sent, pos_id, label_id in zip(content_sent_list[index], extractive_position[index],
                                                      label_position):

                        if pos_id == 1:
                            extractive_result.append(sent)
                        if label_id == 1:
                            label_result.append(sent)

                    extractive_result = '。'.join(extractive_result)
                    label_result = '。'.join(label_result)
                    extractive_result_token = ' '.join(tokenizer.tokenize(extractive_result))
                    label_result_token = ' '.join(tokenizer.tokenize(label_result))
                    print("========================================================================================")
                    print("extractive position: ", extractive_position[index])
                    print("label:", label[index])
                    # print("break point1")
                    if len(extractive_result.strip()) != 0 and len(extractive_result)-len(label_result) < 200 and len(label_result.strip()) != 0:
                        # print("break point2")
                        result_score = rouge.get_scores(extractive_result_token, label_result_token)
                        print("rouge score -- f:{} p:{} r:{}".format(result_score[0]["rouge-l"]["f"],
                                                                     result_score[0]["rouge-l"]["p"],
                                                                     result_score[0]["rouge-l"]["r"]))
                    else:
                        # print("break point3")
                        print("rouge score -- f:{} p:{} r:{}".format(0, 0, 0))

                # print(output.size())  # output's size : (batch_size,max_sent_num,2)
                optimizer.zero_grad()
                # print(output.size(),label.size())
                loss = loss_func(output, label)  # label's size : (batch_size,max_sent_num)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # bert_optim.step()
                print("batch_id : {} loss : {}".format(batch_id, loss))

                if (epoch + 1) % 1 == 0 and (batch_id + 1) % len(dataloader) == 0:
                    print("BiLSTM Model Saved")
                    torch.save(lstmModel.state_dict(), "./bert-bilstm-extrat-model-final.pth")
                    model.save_pretrained("./bert-extractive-chinese")

            #=======================================TEST=================================================
            else:
                with torch.no_grad():
                    # lstmModel.eval()
                    judge_file = open('./judge.json', 'a', encoding='utf-8')
                    output, _ = lstmModel(final_feature)
                    # output = LFPModel(output)
                    # output = F.softmax(output, dim=-1)
                    # print("output size:",output.size())
                    extractive_position = output.clone().detach()

                    # output = output.transpose(0,1).contiguous()
                    # extractive_position = extractive_position.transpose(0,1).contiguous()
                    # print(extractive_position.size())
                    # print("max_sent_num:",max_sent_num)
                    for batch in range(batch_size):
                        for i in range(max_sent_num):
                            if output[batch][i] >= 0.5:
                                extractive_position[batch][i] = 1
                            else:
                                extractive_position[batch][i] = 0
                    # print("extractive_position",extractive_position)
                    extractive_position = extractive_position.cpu()
                    extractive_position = extractive_position.numpy()
                    extractive_position = extractive_position.tolist()
                    # print("extractive_position,",extractive_position)

                    for index in range(batch_size):
                        extractive_result = []
                        label_result = []
                        cluster_res = cluster_extract_model(content[index])
                        # print("len of content sent list,",len(content_sent_list[index]))
                        extractive_position[index] = extractive_position[index][:len(content_sent_list[index])]
                        label_position = label[index][:len(content_sent_list[index])]
                        for sent, pos_id, label_id in zip(content_sent_list[index], extractive_position[index],label_position):

                            if pos_id == 1:
                                extractive_result.append(sent)
                            if label_id == 1:
                                label_result.append(sent)

                        extractive_result = '。'.join(extractive_result)
                        label_result = '。'.join(label_result)
                        extractive_result_token = ' '.join(tokenizer.tokenize(extractive_result))
                        label_result_token = ' '.join(tokenizer.tokenize(label_result))
                        cluster_result_token = ' '.join(tokenizer.tokenize(cluster_res))
                        print("========================================================================================")
                        print("训练式抽取结果：", extractive_result)
                        print(" ")
                        print("聚类式抽取结果：",cluster_res)
                        print(" ")
                        print("人工摘要：",summary[index])
                        print(" ")
                        # print("extractive position: ", extractive_position[index])
                        # print("label:", label[index])
                        if len(extractive_result.strip()) != 0 and len(label_result.strip()) != 0:
                            result_score = rouge.get_scores(extractive_result_token,label_result_token)
                            result_score_clus = rouge.get_scores(cluster_result_token,label_result_token)
                            print("训练式摘要的rouge值 -- f:{} p:{} r:{}".format(result_score[0]["rouge-l"]["f"],result_score[0]["rouge-l"]["p"],result_score[0]["rouge-l"]["r"]))
                            print(" ")
                            print("聚类式摘要的rouge值 -- f:{} p:{} r:{}".format(result_score_clus[0]["rouge-l"]["f"],result_score_clus[0]["rouge-l"]["p"],result_score_clus[0]["rouge-l"]["r"]))

                        else:
                            print("rouge值异常 -- f:{} p:{} r:{}".format(0,0,0))

                        # print("========================================================================================")
                        # df = {"content": extractive_result, "summary": summary[index]}
                        # encode_json = json.dumps(df,ensure_ascii=False)
                        # print(encode_json, file=judge_file)
                        #
                        # if batch_id + 1 != len(dataloader) or index + 1 != batch_size:
                        #     print(',', file=judge_file)

    # if not train:
    #     judge_file = open('./judge.json', 'a', encoding='utf-8')
    #     print(']', file=judge_file)
    #     judge_file.close()


get_extractive_sum(testdataloader, epochs, False, "./bert-bilstm-extrat-model-final.pth", batch_size=batch_size)
