import sys

# sys[1]: input file path
# sys[2]: output file path

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]


def same_sentence(sent1, sent2):
    if not len(sent1) == len(sent2):
        return False
    for w1, w2 in zip(sent1, sent2):
        if not w1 == w2:
            return False
    return True


def get_eval_format(all_sentence_lists, all_label_lists):
    all_sent = []
    for sentence, label_list in zip(all_sentence_lists, all_label_lists):
        verb_index, all_args = get_args(sentence, label_list)
        if len(all_sent) == 0 or not same_sentence(all_sent[-1]['sent'], sentence):
            all_sent.append(
                {'sent': sentence, 'verb': [verb_index], 'args': [all_args]}
            )
        else:
            all_sent[-1]['verb'].append(verb_index)
            all_sent[-1]['args'].append(all_args)

    all_lines = []
    for item in all_sent:
        sentence = item['sent']
        verb_list = item['verb']
        args_list = item['args']
        verb_num = len(verb_list)

        for i in range(len(sentence)):
            line_items = ['-'] + ['*'] * verb_num
            for j in range(len(verb_list)):
                if i == verb_list[j]:
                    line_items[0] = sentence[i]
                    line_items[j+1] = '(V*)'
                for arg in args_list[j]:
                    if i == arg[1]:
                        line_items[j+1] = '(%s' % arg[0] + line_items[j+1]
                    if i == arg[2]:
                        line_items[j+1] = line_items[j+1] + ')'
            all_lines.append('\t'.join(line_items))

        all_lines.append('')

    return all_lines


def write_all_lines(file_path, all_lines):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in all_lines:
            f.write(line)
            f.write('\n')


def readfile(filename):
    all_sentence = []
    all_label = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()

    sentence = []
    labels = []
    for line in lines:
        line = line.strip()
        if line == '':
            if len(sentence) > 0:
                all_sentence.append(sentence)
                all_label.append(labels)
                sentence = []
                labels = []
            continue
        splits = line.split()
        sentence.append(splits[0])
        sr = splits[1]
        labels.append(sr)
    if len(sentence) > 0:
        all_sentence.append(sentence)
        all_label.append(labels)
    return all_sentence, all_label


def get_args(sentence, label_list):
    in_args = False
    current_args = None
    args_start = None
    verb_index = None

    all_args = []

    for i, (w, l) in enumerate(zip(sentence, label_list)):
        if l.startswith('B-'):
            if in_args:
                args_end = i - 1
                all_args.append([current_args, args_start, args_end])
            current_args = l[2:]
            in_args = True
            args_start = i
            continue

        if in_args and l == 'O':
            in_args = False
            args_end = i - 1
            all_args.append([current_args, args_start, args_end])
            continue

        if l == 'V':
            if in_args:
                args_end = i - 1
                in_args = False
                all_args.append([current_args, args_start, args_end])
            verb_index = i
            continue

    if in_args:
        args_end = len(sentence) - 1
        all_args.append([current_args, args_start, args_end])

    assert verb_index is not None

    return verb_index, all_args


all_sent, all_label = readfile(input_file_path)
all_lines = get_eval_format(all_sent, all_label)
write_all_lines(output_file_path, all_lines)
