import json
from os import path
import os
import sys

HOME_DIR = sys.argv[1]

INPUT_DIR = path.join(HOME_DIR, 'json')
OUTPUT_DIR = path.join(HOME_DIR, 'tsv')

if not path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def group_predicates(srl_list, start_index):
    predicate_index = -1
    all_predicates = []
    predicate_srl = []
    for i, srl_node in enumerate(srl_list):
        verb_index = srl_node[0]

        if i == 0:
            predicate_index = verb_index

        # All verbs in CoNLL 2012 (Ontonotes 5) are single words; CoNNL 2005 contains verbs with multiple word
        # if srl_node[3] == 'V':
        #     assert srl_node[1] == srl_node[2]

        if not verb_index == predicate_index and len(predicate_srl) > 0:
            predicate_index = verb_index
            all_predicates.append(predicate_srl)
            predicate_srl = []

        srl_node[0] -= start_index
        srl_node[1] -= start_index
        srl_node[2] -= (start_index - 1)
        predicate_srl.append(srl_node)

    if len(predicate_srl) > 0:
        all_predicates.append(predicate_srl)

    num_p = sum([len(p) for p in all_predicates])

    assert num_p == len(srl_list)

    return all_predicates


for flag in ['train', 'dev', 'test', 'brown']:

    input_data = []
    data_path = path.join(INPUT_DIR, flag + '.json')
    if not os.path.exists(data_path):
        continue
    print('process %s' % flag)

    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line == '':
                input_data.append(json.loads(line))

    output_data = []
    ignored_sentence_num = 0
    sentence_num = 0
    instance_num = 0
    token_num = 0
    max_sent_length = -1

    for data in input_data:
        sentences = data['sentences']
        srls = data['srl']

        start_index = 0
        for sent, srl in zip(sentences, srls):
            # the sentence does not have any predicates

            token_num += len(sent)

            if len(srl) == 0:
                ignored_sentence_num += 1
                start_index += len(sent)
                continue

            grouped_predicates = group_predicates(srl, start_index)

            start_index += len(sent)
            sentence_num += 1
            instance_num += len(grouped_predicates)

            if len(sent) > max_sent_length:
                max_sent_length = len(sent)

            for predicate_srl in grouped_predicates:
                labels = ['O' for _ in range(len(sent))]

                for srl_node in predicate_srl:
                    b_index = srl_node[1]
                    e_index = srl_node[2]
                    span_len = e_index - b_index

                    assert span_len > 0

                    if srl_node[3] == 'V':
                        for j in range(b_index, e_index):
                            labels[j] = srl_node[3]
                    else:
                        labels[b_index] = 'B-' + srl_node[3]

                        if span_len > 1:
                            for j in range(b_index+1, e_index):
                                labels[j] = 'I-' + srl_node[3]

                for w, l in zip(sent, labels):
                    output_data.append('%s\t%s' % (w, l))

                output_data.append('')

    with open(path.join(OUTPUT_DIR, flag + '.tsv'), 'w', encoding='utf8') as f:
        for line in output_data:
            f.write(line)
            f.write('\n')

    print('token number in %s: %d' % (flag, token_num))
    print('ignored sentence number in %s: %d' % (flag, ignored_sentence_num))
    print('sentence number in %s: %d' % (flag, sentence_num))
    print('instance number in %s: %d' % (flag, instance_num))
    print('max sentence length in %s: %d' % (flag, max_sent_length))
