from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, ZenModel, BertTokenizer, Biaffine, MLP, CRF
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import ZenNgramDict
from srl_helper import save_json, load_json, get_pos_label_list, get_syn_label_list
import subprocess
import os
import json

from dep_parser import DepInstanceParser
from collections import defaultdict


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'mlp_dropout': 0.33,
    'n_mlp': 400,
    'use_crf': False,
}

class KVMN(nn.Module):
    def __init__(self, hidden_size, key_size, val_size, b_add=True, b_both_kv=False, b_self_prob=False, type_="dep"):
        super(KVMN, self).__init__()
        self.temper = hidden_size ** 0.5
        self.key_embedding = nn.Embedding(key_size, hidden_size)
        self.val_embedding = nn.Embedding(val_size, hidden_size)
        self.hidden_state_compress = nn.Linear(hidden_size * 2, hidden_size)
        self.b_add = b_add
        self.b_both_kv = b_both_kv
        self.b_self_prob = b_self_prob
        self.type = type_

    def forward(self, hidden_state, key_seq, value_matrix, key_mask_matrix, output_kvmn_weight=False):
        embedding_key = self.key_embedding(key_seq)
        embedding_val = self.val_embedding(value_matrix)
        hidden_state = self.hidden_state_compress(hidden_state)

        if self.b_self_prob is True:
            key_seq_h = hidden_state.permute(0, 2, 1)[:,:,:key_seq.shape[-1]]
            if key_seq_h.shape[-1] < key_seq.shape[-1]:
                key_seq_h = torch.cat([key_seq_h, torch.zeros([key_seq_h.shape[0], key_seq_h.shape[1], (key_seq.shape[-1] - key_seq_h.shape[2])])], dim=-1)
            u = torch.matmul(hidden_state.float(), key_seq_h.float()) / self.temper
        else:
            key_seq_h = embedding_key.permute(0, 2, 1)
            u = torch.matmul(hidden_state.float(), key_seq_h.float()) / self.temper

        tmp_key_mask_matrix = torch.clamp(key_mask_matrix, 0, 1)
        exp_u = torch.exp(u)
        
        delta_exp_u = torch.mul(exp_u, tmp_key_mask_matrix.float())

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        #batch_size, max_seq_len, max_key_len
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        if self.type == "pos" or self.type == "syn":
            embedding_val = torch.stack([embedding_val] * p.shape[-1], 1)
        
        embedding_val = embedding_val.permute(3, 0, 1, 2)
        
        o = torch.mul(p, embedding_val.float()).type_as(hidden_state)

        o = o.permute(1, 2, 3, 0) #batch_size, max_seq_len, max_key_len, hidden_size
        o = torch.sum(o, 2)

        if self.b_both_kv is True:
            # batch_size, max_key_len, hidden_size
            embedding_key_matrix = torch.stack([embedding_key] * p.shape[1], 1).permute(3, 0, 1, 2)
            ko = torch.mul(p, embedding_key_matrix.float()).type_as(hidden_state).permute(1, 2, 3, 0)
            ko = torch.sum(ko, 2)
            o = torch.add(o, ko)

        if self.b_add is True:
            o = torch.add(o, hidden_state)

        if output_kvmn_weight is True:
            return o, p
        return o


class SRTagger(nn.Module):

    def __init__(self, labelmap, hpara, model_path, key_size=0, val_size=0, from_pretrained=True, b_both_kv=False,
                 b_self_prob=False,
                 dep_order='first_order', direct=True, freq_limit=5, knowledge='dep',
                 pos_labelmap=None, syn_labelmap=None,
                 keys_dict=None, keys_freq_dict=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_crf = self.hpara['use_crf']
        self.dep_order = dep_order
        self.direct = direct
        self.freq_limit = freq_limit
        self.knowledge=knowledge

        self.prepare_vals_dict()
        self.prepare_keys_dict(keys_dict, keys_freq_dict)

        self.other_para = {
            'key_size': key_size,
            'val_size': val_size,
            'b_both_kv': b_both_kv,
            'b_self_prob': b_self_prob,
            'dep_order': dep_order,
            'direct': direct,
            'freq_limit': freq_limit,
            'knowledge': knowledge,
            'pos_labelmap': pos_labelmap,
            'syn_labelmap': syn_labelmap,
            'keys_dict': keys_dict,
            'keys_freq_dict': keys_freq_dict
        }
        
        if hpara['use_zen']:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.xlnet = None
        self.zen = None
        self.zen_ngram_dict = None
        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                print(model_path)
                self.xlnet = XLNetModel.from_pretrained(model_path)
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                key_list = list(state_dict.keys())
                reload = False
                for key in key_list:
                    if key.find('xlnet.') > -1:
                        reload = True
                        state_dict[key[key.find('xlnet.') + len('xlnet.'):]] = state_dict[key]
                    state_dict.pop(key)
                if reload:
                    self.xlnet.load_state_dict(state_dict)
            else:
                config, model_kwargs = XLNetModel.config_class.from_pretrained(model_path, return_unused_kwargs=True)
                self.xlnet = XLNetModel(config)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        # self.tokenizer.add_never_split_tokens(["[V]", "[/V]"])
        if "dep" in self.knowledge:
            self.kvmn_dep = KVMN(hidden_size, key_size, val_size, b_both_kv=b_both_kv, b_self_prob=b_self_prob, type_="dep")

        pos_label_size = len(pos_labelmap)
        self.pos_labelmap = pos_labelmap
        if "pos" in self.knowledge:
            self.kvmn_pos = KVMN(hidden_size, key_size, pos_label_size, b_both_kv=b_both_kv, b_self_prob=b_self_prob, type_="pos")

        syn_label_size = len(syn_labelmap)
        self.syn_labelmap = syn_labelmap
        if "syn" in self.knowledge:
            self.kvmn_syn = KVMN(hidden_size, key_size, syn_label_size, b_both_kv=b_both_kv, b_self_prob=b_self_prob, type_="syn")

        plus = self.knowledge.count('+') + 1

        self.mlp_pre_h = MLP(n_in=hidden_size*plus,
                             n_hidden=self.hpara['n_mlp'],
                             dropout=self.hpara['mlp_dropout'])
        self.mlp_arg_h = MLP(n_in=hidden_size*plus,
                             n_hidden=self.hpara['n_mlp'],
                             dropout=self.hpara['mlp_dropout'])

        self.srl_attn = Biaffine(n_in=self.hpara['n_mlp'],
                                 n_out=self.num_labels,
                                 bias_x=True,
                                 bias_y=True)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    def get_enhanced_output(self, valid_output, predicates_):
        batch_size, max_len, hidden_size = valid_output.shape
        v_o = torch.zeros((batch_size, max_len, hidden_size * 2))
        for i_ in range(batch_size):
            for j_ in range(max_len):
                v_o[i_][j_] = torch.cat([valid_output[i_][j_], predicates_[i_]], dim=-1)
        return v_o

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                verb_index=None, labels=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                dep_key_list=None,
                dep_adj_matrix=None,
                dep_type_matrix=None,
                output_kvmn_weight=False,
                pos_matrix=None,
                syn_matrix=None,
                pos_mask_matrix=None,
                syn_mask_matrix=None,
                kvmn_position=0
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape
        
        max_len = attention_mask_label.shape[1]
        

        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            valid_output[i][:sent_len] = temp[:sent_len]

        valid_output = self.dropout(valid_output)
        
        predicates_ = torch.zeros(batch_size, feat_dim, dtype=valid_output.dtype, device=valid_output.device)
        for i in range(batch_size):
            predicates_[i] = valid_output[i][verb_index[i][0]]
        
        enhanced_valid_output = self.get_enhanced_output(valid_output, predicates_)
        enhanced_valid_output = enhanced_valid_output.to(valid_output.device)
        
        if kvmn_position == 0:
            valid_output_concat = []
            if "dep" in self.knowledge:
                valid_output_dep = self.kvmn_dep(enhanced_valid_output, dep_key_list, dep_adj_matrix, dep_type_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_dep)
            if "pos" in self.knowledge:
                valid_output_pos = self.kvmn_pos(enhanced_valid_output, dep_key_list, pos_matrix, pos_mask_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_pos)
            if "syn" in self.knowledge:
                valid_output_syn = self.kvmn_syn(enhanced_valid_output, dep_key_list, syn_matrix, syn_mask_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_syn)

            valid_output = torch.cat(valid_output_concat, dim=-1)

        batch_size, _, hidden_dim = valid_output.shape
        predicates = torch.zeros(batch_size, hidden_dim, dtype=valid_output.dtype, device=valid_output.device)
        for i in range(batch_size):
            predicates[i] = valid_output[i][verb_index[i][0]]

        if kvmn_position == 1:
            valid_output_concat = []
            if "dep" in self.knowledge:
                valid_output_dep = self.kvmn_dep(enhanced_valid_output, dep_key_list, dep_adj_matrix, dep_type_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_dep)
            if "pos" in self.knowledge:
                valid_output_pos = self.kvmn_pos(enhanced_valid_output, dep_key_list, pos_matrix, pos_mask_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_pos)
            if "syn" in self.knowledge:
                valid_output_syn = self.kvmn_syn(enhanced_valid_output, dep_key_list, syn_matrix, syn_mask_matrix, output_kvmn_weight)
                valid_output_concat.append(valid_output_syn)

            valid_output = torch.cat(valid_output_concat, dim=-1)

        pre_h = self.mlp_pre_h(predicates)
        arg_h = self.mlp_arg_h(valid_output)

        # [batch_size, seq_len, n_labels]
        s_labels = self.srl_attn(arg_h, pre_h).permute(0, 2, 1)

        if labels is not None:
            if self.crf is not None:
                return -1 * self.crf(emissions=s_labels, tags=labels, mask=attention_mask_label)
            else:
                s_labels = s_labels[attention_mask_label]
                labels = labels[attention_mask_label]
                return self.loss_function(s_labels, labels)
        else:
            if self.crf is not None:
                return self.crf.decode(s_labels, attention_mask_label)[0]
            else:
                pre_labels = torch.argmax(s_labels, dim=2)
                return pre_labels


    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp'] = args.n_mlp

        hyper_parameters['use_crf'] = args.use_crf

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_model_dir, vocab_dir):
        best_eval_model_dir = os.path.join(output_model_dir, 'model')
        if not os.path.exists(best_eval_model_dir):
            os.makedirs(best_eval_model_dir)

        output_model_path = os.path.join(best_eval_model_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(best_eval_model_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(best_eval_model_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        output_other_arg_file = os.path.join(best_eval_model_dir, 'others.json')
        save_json(output_other_arg_file, self.other_para)

        output_config_file = os.path.join(best_eval_model_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.xlnet:
                writer.write(self.xlnet.config.to_json_string())
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(best_eval_model_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.xlnet:
            vocab_name = 'spiece.model'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(best_eval_model_dir, vocab_name))
        subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        other_arg_file = os.path.join(model_path, 'others.json')
        others = load_json(other_arg_file)

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, **others)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res

    def read_features(self, data_path):
        tool = "stanford"
        all_data = read_json(data_path)
        all_feature_data = []
        for data in all_data:
            tokens = []
            basicDependencies = []
            sentences = data['sentences']
            for sentence in sentences:
                tokens.extend(sentence['tokens'])
                basicDependencies.extend(sentence['basicDependencies'])
            ori_sentence = data["ori_sentence"]
            label = data["sequence_label"]
            pos_label = data["pos_label"]
            syn_label = data["syn_label"]

            verb_index = []
            label_len = len(label)
            for j in range(label_len):
                if label[j] == 'V':
                    verb_index.append(j)
            dep_instance_parser = DepInstanceParser(basicDependencies=basicDependencies, tokens=tokens)
            if self.dep_order == "first_order":
                first_dep_adj_matrix, first_dep_type_matrix = dep_instance_parser.get_first_order(direct=self.direct)
            elif self.dep_order == "second_order":
                first_dep_adj_matrix, first_dep_type_matrix = dep_instance_parser.get_second_order(direct=self.direct)
            elif self.dep_order == "third_order":
                first_dep_adj_matrix, first_dep_type_matrix = dep_instance_parser.get_third_order(direct=self.direct)
            #dep_path_dep_adj_matrix, dep_path_dep_type_matrix = dep_instance_parser.get_dep_path(start_range, end_range, direct=self.direct)

            # dep_key_list = [self.keys_dict[key] for key in dep_instance_parser.words]

            all_feature_data.append({
                "words": dep_instance_parser.words,
                "ori_sentence": ori_sentence,
                # "dep_key_list":dep_key_list,
                "first_order_dep_adj_matrix": first_dep_adj_matrix,
                "first_order_dep_type_matrix": first_dep_type_matrix,
                "pos_label": pos_label,
                "syn_label": syn_label,
                "label": label,
                "verb_index": verb_index
            })
        return all_feature_data

    def load_data(self, data_path, do_predict=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        #lines = readfile(data_path, flag)
        examples = self.read_features(data_path)

        #examples = self.process_data(lines, flag)

        return examples

    @staticmethod
    def process_data(lines, flag):

        examples = []
        for i, (sentence, label, verb_index) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None,
                                         label=label, verb_index=verb_index))
        return examples

    def get_dep_labels(self):
        # if self.tool == "spacy":
        #     labels = ["nsubj","prep","det","amod","pobj","advmod","ROOT","attr","","appos","punct","npadvmod","nmod",
        #          "compound","aux","cc","conj","acomp","nsubjpass","auxpass","poss","dobj","preconj","neg","relcl",
        #          "ccomp","xcomp","pcomp","intj","acl","dep","prt","nummod","quantmod","mark","advcl","dative","agent",
        #          "case","oprd","parataxis","csubj","expl","predet","meta","csubjpass"]
        # else:
        labels = ["ROOT","det","nsubj","mark","acl","advmod","nmod:poss","amod","dobj","case","nmod","compound","punct",
                "nsubjpass","auxpass","cc","conj","advcl","cop","acl:relcl","ccomp","aux","csubjpass","nummod","dep",
                "xcomp","appos","nmod:npmod","compound:prt","root","nmod:tmod","neg","mwe","parataxis","det:predet",
                "expl","iobj","cc:preconj","csubj","discourse"]
        final_labels = ["self_loop"]
        for label in labels:
            if self.direct:
                final_labels.append("{}_in".format(label))
                final_labels.append("{}_out".format(label))
            else:
                final_labels.append(label)
        return final_labels

    def prepare_vals_dict(self):
        dep_type_list = self.get_dep_labels()
        vals_dict = {"none": 0}
        for dep_type in dep_type_list:
            vals_dict[dep_type] = len(vals_dict)
        self.vals_dict = vals_dict

    def prepare_keys_dict(self, keys_dict, keys_frequency_dict):
        self.keys_dict = keys_dict
        self.keys_frequency_dict = keys_frequency_dict

    def convert_examples_to_features(self, examples):

        dep_label_map = self.vals_dict
        features = []

        length_list = []
        tokens_list = []
        labels_list = []
        poses_list = []
        syns_list = []
        valid_list = []
        label_mask_list = []
        eval_mask_list = []
        dep_adj_matrix_list = []
        dep_type_matrix_list = []
        pos_matrix_list = []
        syn_matrix_list = []
        pos_mask_list = []
        syn_mask_list = []
        
        pos_labelmap = self.pos_labelmap
        syn_labelmap = self.syn_labelmap
        
        for (ex_index, example) in enumerate(examples):
            # text_list = example.text_a
            # label_list = example.label
            # verb_index = example.verb_index
            text_list = example['ori_sentence']
            label_list = example['label']
            verb_index = example['verb_index']
            pos_label_list = example["pos_label"]
            syn_label_list = example["syn_label"]
            # tt = example[""]
            dep_key_list = []
            tokens = []
            labels = []
            poses = []
            syns = []
            valid = []
            label_mask = []
            eval_mask = []

            if len(text_list) > self.max_seq_length - 2:
                continue

            assert verb_index[-1] - verb_index[0] == len(verb_index) - 1

            # add [V] and [\V] to the beginning and ending of the predicate
            new_textlist = [w for w in text_list[:verb_index[0]]]
            new_textlist.append('[V]')
            new_textlist.extend([w for w in text_list[verb_index[0]: verb_index[-1] + 1]])
            new_textlist.append('[/V]')
            new_textlist.extend([w for w in text_list[verb_index[-1] + 1:]])
            assert len(new_textlist) == len(label_list) + 2
            text_list = new_textlist

            tmp = 0
            for i, word in enumerate(text_list):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                if word == '[V]' or word == '[/V]':
                    for _ in range(len(token)):
                        valid.append(0)
                    tmp += 1
                    continue
                label_1 = label_list[i - tmp]
                pos_l = pos_label_list[i - tmp]
                syn_l = syn_label_list[i - tmp]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        if pos_l not in self.pos_labelmap:
                            pos_l = 'O'
                        if syn_l not in self.syn_labelmap:
                            syn_l = 'OTHER'
                        poses.append(pos_labelmap[pos_l])
                        syns.append(syn_labelmap[syn_l])
                        if label_1 == 'V':
                            eval_mask.append(0)
                        else:
                            eval_mask.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
                dep_key_list.append(word)
            assert tmp == 2
            assert len(tokens) == len(valid)
            assert len(eval_mask) == len(label_mask)

            length_list.append(len(tokens))
            tokens_list.append(tokens)
            labels_list.append(labels)
            poses_list.append(poses)
            syns_list.append(syns)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            eval_mask_list.append(eval_mask)

            max_words_num = sum(valid)
            def get_adj_with_value_matrix(dep_adj_matrix, dep_type_matrix):
                final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                for pi in range(max_words_num):
                    for pj in range(max_words_num):
                        if dep_adj_matrix[pi][pj] == 0:
                            continue
                        if pi >= self.max_seq_length or pj >= self.max_seq_length:
                            continue
                        if self.keys_frequency_dict[dep_key_list[pj]] < self.freq_limit:
                            continue
                        final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                        final_dep_type_matrix[pi][pj] = dep_label_map[dep_type_matrix[pi][pj]]
                return final_dep_adj_matrix, final_dep_type_matrix

            first_order_dep_adj_matrix, first_order_dep_type_matrix = get_adj_with_value_matrix(example["first_order_dep_adj_matrix"], example["first_order_dep_type_matrix"])
            
            def get_pos_mask_matrix():
                pos_mask_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                for p_i in range(max_words_num):
                    for p_j in range(max(p_i - 2, 0), min(max_words_num, p_i + 3)):
                        pos_mask_matrix[p_i][p_j] = 1
                return pos_mask_matrix
                        
            pos_mask_matrix = get_pos_mask_matrix()
            #syn_mask_matrix = get_syn_mask_matrix(tt)
            syn_mask_matrix = pos_mask_matrix
            pos_matrix_list.append(poses)
            syn_matrix_list.append(syns)
            pos_mask_list.append(pos_mask_matrix)
            syn_mask_list.append(syn_mask_matrix)
            
            dep_adj_matrix_list.append(first_order_dep_adj_matrix)
            dep_type_matrix_list.append(first_order_dep_type_matrix)

            dep_key_list = [self.keys_dict[change_word(key)] for key in dep_key_list]

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        self.seq_pad_length = seq_pad_length
        #seq_pad_length = self.max_seq_length + 2
        label_pad_length = max(label_len_list)
        #label_pad_length = self.max_seq_length

        for indx, (example, tokens, labels, valid, label_mask, eval_mask, first_order_dep_adj_matrix, first_order_dep_type_matrix, p_m, s_m, pos_mask_matrix, syn_mask_matrix) in \
                enumerate(zip(examples, tokens_list, labels_list, valid_list, label_mask_list, eval_mask_list, dep_adj_matrix_list, 
                dep_type_matrix_list, pos_matrix_list, syn_matrix_list, pos_mask_list, syn_mask_list)):

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 0)

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])

            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(0)

            assert sum(valid) == len(label_ids)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                label_ids.append(0)
                label_mask.append(0)
                eval_mask.append(0)
                p_m.append(0)
                s_m.append(0)

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length
            assert len(p_m) == label_pad_length
            assert len(s_m) == label_pad_length

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              verb_index=verb_index,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              eval_mask=eval_mask,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              dep_adj_matrix=first_order_dep_adj_matrix,
                              dep_type_matrix=first_order_dep_type_matrix,
                              dep_key_list=dep_key_list,
                              pos_matrix=p_m,
                              syn_matrix=s_m,
                              pos_mask_matrix=pos_mask_matrix,
                              syn_mask_matrix=syn_mask_matrix
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_verb_idx = torch.tensor([[f.verb_index[0]] for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        all_pos_matrix = torch.tensor([f.pos_matrix for f in feature], dtype=torch.long)
        all_syn_matrix = torch.tensor([f.syn_matrix for f in feature], dtype=torch.long)

        max_len = all_lmask_ids.shape[1]
        def get_dep_matrix(ori_dep_type_matrix):
            dep_type_matrix = np.zeros((max_len, max_len), dtype=np.int)
            max_words_num = len(ori_dep_type_matrix)
            for i in range(max_words_num):
                dep_type_matrix[i][:max_words_num] = ori_dep_type_matrix[i]
            return dep_type_matrix

        def get_dep_key_list(dep_key_list):
            t_dep_key_list = np.zeros((max_len), dtype=np.int)
            for i in range(len(dep_key_list)):
                t_dep_key_list[i] = dep_key_list[i]
            return t_dep_key_list
        
        def get_pos_syn_mask_matrix(ori_matrix):
            new_matrix = np.zeros((max_len, max_len), dtype=np.int)
            max_words_num = len(ori_matrix)
            for i in range(max_words_num):
                new_matrix[i][:max_words_num] = ori_matrix[i]
            return new_matrix

        all_dep_adj_matrix = []
        all_dep_type_matrix = []
        all_dep_key_list = []
        all_pos_mask_matrix = []
        all_syn_mask_matrix = []

        for f in feature:
            dep_adj_matrix_ = get_dep_matrix(f.dep_adj_matrix)
            dep_type_matrix_ = get_dep_matrix(f.dep_type_matrix)
            dep_key_list_ = get_dep_key_list(f.dep_key_list)
            pos_mask_matrix = get_pos_syn_mask_matrix(f.pos_mask_matrix)
            syn_mask_matrix = get_pos_syn_mask_matrix(f.syn_mask_matrix)
            p_m_ = get_dep_key_list(f.pos_matrix)
            s_m_ = get_dep_key_list(f.syn_matrix)
            all_dep_adj_matrix.append(dep_adj_matrix_)
            all_dep_type_matrix.append(dep_type_matrix_)
            all_dep_key_list.append(dep_key_list_)
            all_pos_mask_matrix.append(pos_mask_matrix)
            all_syn_mask_matrix.append(syn_mask_matrix)
            

        all_dep_adj_matrix = torch.tensor(all_dep_adj_matrix, dtype=torch.long)
        all_dep_type_matrix = torch.tensor(all_dep_type_matrix, dtype=torch.long)
        all_dep_key_list = torch.tensor(all_dep_key_list, dtype=torch.long)
        all_pos_mask_matrix = torch.tensor(all_pos_mask_matrix, dtype=torch.long)
        all_syn_mask_matrix = torch.tensor(all_syn_mask_matrix, dtype=torch.long)
        

        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)
        dep_key_list = all_dep_key_list.to(device)
        dep_adj_matrix = all_dep_adj_matrix.to(device)
        dep_type_matrix = all_dep_type_matrix.to(device)
        pos_matrix = all_pos_matrix.to(device)
        syn_matrix = all_syn_matrix.to(device)
        pos_mask_matrix = all_pos_mask_matrix.to(device)
        syn_mask_matrix = all_syn_mask_matrix.to(device)

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None

        return input_ids, input_mask, l_mask, eval_mask, all_verb_idx, label_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids, dep_key_list, dep_adj_matrix, dep_type_matrix, \
               pos_matrix, syn_matrix, pos_mask_matrix, syn_mask_matrix


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, verb_index=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.verb_index = verb_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, verb_index, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 dep_adj_matrix=None, dep_type_matrix=None, dep_key_list=None,
                 pos_matrix=None, syn_matrix=None, pos_mask_matrix=None, syn_mask_matrix=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.verb_index = verb_index
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

        self.dep_adj_matrix = dep_adj_matrix
        self.dep_type_matrix = dep_type_matrix
        self.dep_key_list = dep_key_list
        
        self.pos_matrix = pos_matrix
        self.syn_matrix = syn_matrix
        
        self.pos_mask_matrix = pos_mask_matrix
        self.syn_mask_matrix = syn_mask_matrix


def readfile(filename, flag):
    data = []
    sentence = []
    label = []
    verb_index = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '':
                    if len(sentence) > 0:
                        data.append((sentence, label, verb_index))
                        sentence = []
                        label = []
                        verb_index = []
                    continue
                splits = line.split()
                sentence.append(splits[0])
                sr = splits[1]
                if sr == 'V':
                    verb_index.append(len(label))
                label.append(sr)
            if len(sentence) > 0:
                data.append((sentence, label, verb_index))
        else:
            raise ValueError()
            # for line in lines:
            #     line = line.strip()
            #     if line == '':
            #         continue
            #     label_list = ['NN' for _ in range(len(line))]
            #     data.append((line, label_list))
    return data

def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

def get_vals_size():
    labels = ["nsubj","prep","det","amod","pobj","advmod","ROOT","attr","","appos","punct","npadvmod","nmod",
             "compound","aux","cc","conj","acomp","nsubjpass","auxpass","poss","dobj","preconj","neg","relcl",
             "ccomp","xcomp","pcomp","intj","acl","dep","prt","nummod","quantmod","mark","advcl","dative","agent",
             "case","oprd","parataxis","csubj","expl","predet","meta","csubjpass"]

    return len(labels)

def prepare_key_dict(train_data_path, dev_data_path, test_data_path, brown_data_path=None):
    keys_frequency_dict = defaultdict(int)
    if brown_data_path is None:
        files = [train_data_path, dev_data_path, test_data_path]
    else:
        files = [train_data_path, dev_data_path, test_data_path, brown_data_path]
    for datafile in files:        
        all_data = read_json(datafile)
        for data in all_data:
            for word in data['ori_sentence']:
                keys_frequency_dict[change_word(word)] += 1
    keys_dict = {"[UNK]":0}
    for key, freq in sorted(keys_frequency_dict.items(), key=lambda x: x[1], reverse=True):
        keys_dict[key] = len(keys_dict)
    
    return keys_dict, keys_frequency_dict, len(keys_dict.keys())

def read_json(data_path):
    data = []
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            jsondata = json.loads(line)
            jsondata["ori_word"] = jsondata["word"]
            basicDependencies = []
            for sentence in jsondata["sentences"]:
                basicDependencies.extend(sentence["basicDependencies"])
            jsondata["word"] = ["" for _ in range(len(basicDependencies))]
            for dep in basicDependencies:
                jsondata["word"][dep["dependent"]-1] = dep["dependentGloss"]
            data.append(jsondata)
    return data
