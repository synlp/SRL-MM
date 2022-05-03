import argparse
import sys

from tqdm import tqdm
import os
from os import path
from .corenlp import StanfordCoreNLP
from nltk.tree import Tree
import json
from random import randint


FULL_MODEL = './stanford-corenlp-full-2018-10-05'
if not os.path.exists(FULL_MODEL):
    print('Stanford CoreNLP does not exist under %s' % FULL_MODEL)
    exit(1)


def read_txt(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = line.split()
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

    return sentence_list, label_list


def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char


def extract_syn(parse_str, verb_index, sent_length):
    c_parse = Tree.fromstring(parse_str)
    current_index = 0
    syn_label = ['' for _ in range(sent_length)]
    words = c_parse.leaves()
    for s in c_parse.subtrees(lambda t: t.height() > 2):
        leaves = s.leaves()

        if len(leaves) == 0:
            continue

        start_index = words[current_index:].index(leaves[0]) + current_index
        end_index = start_index + len(leaves)
        current_index = start_index
        if verb_index[-1] < start_index or end_index <= verb_index[0]:
            continue

        current_sub_index = start_index
        for i, t in enumerate(s):
            sub_start_index = current_sub_index
            sub_end_index = sub_start_index + len(t.leaves())
            current_sub_index = sub_end_index
            node = t.label()
            for index in range(sub_start_index, sub_end_index):
                if index in verb_index:
                    syn_label[index] = 'V'
                elif index < verb_index[0]:
                    syn_label[index] = node + '_L'
                else:
                    syn_label[index] = node + '_R'

    return syn_label


def find_verb_index(labels):
    index_list = []
    for i, l in enumerate(labels):
        if l == 'V':
            index_list.append(i)
    assert len(index_list) > 0
    return index_list


def request_features_from_stanford(data_dir, flag):
    all_sentences, all_labels = read_txt(path.join(data_dir, flag + '.tsv'))
    sentences_str = []
    for sentence in all_sentences:
        sentence = [change(i) for i in sentence]
        # if sentence[-1] == '·':
        #     sentence[-1] = '.'
        sentences_str.append(' '.join(sentence))

    all_data = []
    syn_label_set = set()
    tmp_sent = ''
    with StanfordCoreNLP(FULL_MODEL, lang='en', port=randint(38400, 38596)) as nlp:
        for i in tqdm(range(len(sentences_str))):
            sentence = sentences_str[i]
            labels = all_labels[i]
            ori_sentence = all_sentences[i]
            props = {
                'timeout': '500000',
                'annotators': 'pos, parse, depparse',
                'tokenize.whitespace': 'true',
                'ssplit.eolonly': 'true',
                'pipelineLanguage': 'en',
                'outputFormat': 'json'}
            if not sentence == tmp_sent:
                results = nlp.annotate(sentence, properties=props)
                tmp_sent = sentence
                result = results['sentences'][0]
                results['word'] = [t['word'] for t in result['tokens']]
                results['pos_label'] = [t['pos'] for t in result['tokens']]
                parse_str = ' '.join(result['parse'].split())

            sent_length = len(labels)

            syn_label = extract_syn(parse_str, find_verb_index(labels), sent_length)
            results['syn_label'] = syn_label
            syn_label_set.update(syn_label)
            results['ori_syn_label'] = parse_str
            # results = nlp.request(annotators='deparser', data=sentence)
            # results = nlp.request(annotators='pos', data=sentence)
            results['sequence_label'] = labels
            results['ori_sentence'] = ori_sentence

            all_data.append(results)
    # assert len(all_data) == len(sentences_str)
    with open(path.join(data_dir, flag + '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    # syn_label_set = list(syn_label_set)
    # syn_label_set.sort()
    # print(syn_label_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()

    data_dir = args.dataset

    print(data_dir)

    for flag in ['train', 'dev', 'test', 'brown']:
        if os.path.exists(path.join(data_dir, flag + '.tsv')):
            request_features_from_stanford(data_dir, flag)

