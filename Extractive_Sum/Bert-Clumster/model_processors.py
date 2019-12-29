from BertParent import BertParent
from typing import List
from ClusterFeatures import ClusterFeatures
from abc import abstractmethod
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from textrank4zh import util

class ModelProcessor(object):

    def __init__(
        self,
        model: str = 'bert-base-chinese',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        random_state: int = 12345
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param greedyness: associated with the neuralcoref library. Determines how greedy coref should be.
        :param language: Which language to use for training.
        :param random_state: The random state to reproduce summarizations.
        """

        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option

        self.random_state = random_state


    def process_content_sentences(self, body: str, min_length:int = 10, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences with neural coreference.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences with coreference applied.
        """
        delimiters = util.sentence_delimiters
        delimiters = set([util.as_text(item) for item in delimiters])
        res = [util.as_text(body)]
        util.debug(body)
        util.debug(delimiters)
        for sep in delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if (min_length < len(s.strip()) < max_length)]
        return res

    @abstractmethod
    def run_clusters(
        self,
        content: List[str],
        ratio:float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True
    ) -> List[str]:
        """
        Classes must implement this to run the clusters.
        """
        raise NotImplementedError("Must Implement run_clusters")

    def run(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str ='kmeans'
    ) -> str:
        sentences = self.process_content_sentences(body, min_length, max_length)

        if sentences:
            sentences = self.run_clusters(sentences, ratio, algorithm, use_first)

        return ' '.join(sentences)

    def __call__(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans'
    ) -> str:
        return self.run(body, ratio, min_length, max_length, algorithm=algorithm, use_first=use_first)


class SingleModel(ModelProcessor):
    """
    Deprecated for naming sake.
    """

    def __init__(
        self,
        model='bert-base-chinese',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int=-2,
        reduce_option: str = 'mean',
        random_state: int=12345
    ):
        super(SingleModel, self).__init__(model, custom_model, custom_tokenizer, hidden, reduce_option,
                                          random_state=random_state)

    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool= True) -> List[str]:
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio)

        if use_first:
            if hidden_args[0] != 0:
                hidden_args.insert(0,0)

        return [content[j] for j in hidden_args]


class Summarizer(SingleModel):

    def __init__(
        self,
        model: str = 'bert-base-chinese',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        random_state: int = 12345
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param random_state: The random state to reproduce summarizations.
        """
        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, reduce_option, random_state)

