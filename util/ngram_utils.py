
"""utils for ngram for ZEN model."""

import os
import logging

NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)

class ZenNgramDict(object):
    """
    Dict class to store the ngram
    """
    def __init__(self, ngram_freq_path, tokenizer=None, max_ngram_in_seq=128):
        """Constructs ZenNgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        if os.path.isdir(ngram_freq_path):
            ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.max_ngram_len = 8
        self.id_to_ngram_list = ["[pad]"]
        self.ngram_to_id_dict = {"[pad]": 0}
        self.ngram_to_freq_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                items = line.strip().split(",")
                if len(items) != 2:
                    continue
                ngram,freq = items
                if tokenizer:
                    tokens = tuple(tokenizer.tokenize(ngram))
                    if len(tokens) > self.max_ngram_len:
                        self.max_ngram_len = len(tokens)
                    self.id_to_ngram_list.append(tokens)
                    self.ngram_to_id_dict[tokens] = i + 1
                    self.ngram_to_freq_dict[tokens] = int(freq)
                else:
                    self.ngram_to_freq_dict[ngram] = int(freq)

    def save(self, ngram_freq_path):
        ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        with open(ngram_freq_path, "w+", encoding="utf-8") as fout:
            for ngram,freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))