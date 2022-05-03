import json

# tsv
def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            srl_label = splits[1]
            if srl_label not in label_list:
                label_list.append(srl_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list

# json
def get_label_list_json(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            jsondata = json.loads(line)
            label = jsondata["sequence_label"]
            for srl_label in label:
                if srl_label not in label_list:
                    label_list.append(srl_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def get_pos_list_json(train_data_path):
    label_list = []

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            jsondata = json.loads(line)
            label = jsondata["pos_label"]
            for pos_label in label:
                if pos_label not in label_list:
                    label_list.append(pos_label)

    return label_list


def get_syn_list_json(train_data_path):
    label_list = []

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            jsondata = json.loads(line)
            label = jsondata["syn_label"]
            for syn_label in label:
                if syn_label not in label_list:
                    label_list.append(syn_label)

    return label_list


# pos
def get_pos_label_list(train_data_path):
    # label_list = ['ADJ','ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    label_list = get_pos_list_json(train_data_path)

    label_map = {v: k+1 for k, v in enumerate(label_list)}
    label_map["O"] = 0
    return label_map
    
# syn
def get_syn_label_list(train_data_path):
    # label_list = ['ADJP', 'ADVP', 'CLP', 'DNP', 'DP', 'DVP', 'LCP', 'LST', 'NP', 'PP', 'QP', 'VP', 'punctuation']
    # label_list = ['$_L', '$_R', "''_L", "''_R", ',_L', ',_R', '-LRB-_L', '-LRB-_R', '-RRB-_L', '-RRB-_R', '._L', '._R', ':_L', ':_R', 'ADJP_L', 'ADJP_R', 'ADVP_L', 'ADVP_R', 'CC_L', 'CC_R', 'CD_L', 'CD_R', 'CONJP_L', 'CONJP_R', 'DT_L', 'DT_R', 'FRAG_L', 'FRAG_R', 'FW_L', 'FW_R', 'INTJ_L', 'INTJ_R', 'IN_L', 'IN_R', 'JJR_L', 'JJR_R', 'JJS_L', 'JJ_L', 'JJ_R', 'LST_L', 'LST_R', 'MD_L', 'MD_R', 'NAC_L', 'NNPS_L', 'NNPS_R', 'NNP_L', 'NNP_R', 'NNS_L', 'NNS_R', 'NN_L', 'NN_R', 'NP_L', 'NP_R', 'NX_L', 'NX_R', 'PDT_L', 'POS_L', 'POS_R', 'PP_L', 'PP_R', 'PRN_L', 'PRN_R', 'PRP$_L', 'PRT_L', 'PRT_R', 'QP_L', 'QP_R', 'RBR_L', 'RBR_R', 'RBS_L', 'RB_L', 'RB_R', 'RP_L', 'RP_R', 'RRC_L', 'RRC_R', 'SBARQ_L', 'SBARQ_R', 'SBAR_L', 'SBAR_R', 'SINV_L', 'SINV_R', 'SQ_L', 'SQ_R', 'SYM_L', 'SYM_R', 'S_L', 'S_R', 'TO_L', 'TO_R', 'UCP_L', 'UCP_R', 'UH_L', 'V', 'VBD_L', 'VBD_R', 'VBG_L', 'VBG_R', 'VBN_L', 'VBN_R', 'VBP_L', 'VBP_R', 'VBZ_L', 'VBZ_R', 'VB_L', 'VB_R', 'VP_L', 'VP_R', 'WDT_L', 'WHADJP_L', 'WHADVP_L', 'WHADVP_R', 'WHNP_L', 'WHNP_R', 'WHPP_L', 'WP$_L', 'WP_L', 'WRB_L', 'WRB_R', 'X_L', 'X_R', '``_L', '``_R']
    label_list = get_syn_list_json(train_data_path)
    label_map = {v: k+1 for k, v in enumerate(label_list)}
    label_map["OTHER"] = 0
    return label_map
    

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)
