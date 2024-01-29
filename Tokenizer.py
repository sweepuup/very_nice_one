import re
import torch

vocabulary = {}
token_vocabulary = {}
# vocabulary_length = ['<EOS>']

with open('cl100k_base_vocab_list.txt', 'r', encoding='utf-8') as file:
    for line_count, line in enumerate(file):
        line = line.rstrip('\n')
        if (line.startswith('\'') and line.endswith('\'')) or (line.startswith('\"') and line.endswith('\"')):
            line = line[1:-1]
            vocabulary[line] = line_count
        else:
            vocabulary[line] = line_count
token_vocabulary = {v: k for k, v in vocabulary.items()}

def get_vocabulary():
    return vocabulary


def get_token_vocabulary():
    return token_vocabulary

# def check_vocabulary_length(word):
#     append_length = True
#     for vocab in vocabulary_length:
#         if word == vocab:
#             append_length = False
#             break
#     if append_length == True:
#         vocabulary_length.append(word)
#
# def return_vocabulary_length():
#     return vocabulary_length

def tokenize_sequence(sentence):
    # tokenized_seq = [vocabulary.get('<SOS>')]
    tokenized_seq = []
    regex = r'(\s+\w+|\S+)'
    words = re.split(regex, sentence)
    for word in words:
        if word in vocabulary:
            tokenized_seq.append(vocabulary.get(word, vocabulary.get('<UNK>')))
        else:
            i = 0
            while i < len(word):
                subword_len = 1
                for j in range(len(word), i - 1, -1):
                    subword = word[i:j]
                    if subword in vocabulary:
                        tokenized_seq.append(vocabulary.get(subword, vocabulary.get('<UNK>')))
                        subword_len = len(subword)
                        break
                    if j - i == 1:
                        tokenized_seq.append(vocabulary.get('<UNK>'))
                        break
                i += subword_len
    tokenized_seq.append(vocabulary.get('<EOS>'))
    return tokenized_seq


def detokenize_sequence(tokenized_seq):
    decoded_sentence = ''
    for token in tokenized_seq:
        decoded_sentence += token_vocabulary[token]
    return decoded_sentence


def pad_to_length(seq, length):
    padded_seq = torch.full((length,), fill_value=0, dtype=torch.long)
    padded_seq[:len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded_seq
