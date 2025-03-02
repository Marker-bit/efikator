import numpy as np

CONTEXT_SIZE = 4
alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя*_'

def get_char(line, ind):
    if ind < 0 or ind >= len(line):
        return "_"
    return line[ind]


def cut_context(s: int, line: str):
    r = []
    for ind, i in enumerate(line):
        # print(ind, i)
        if i == "е":
            r.append(
                (
                    _cut_context(s, line, ind),
                    ind,
                )
            )
    return r


def _cut_context(s: int, line: str, pos: int):
    line = line.replace("\n", "")
    # line = f"*{line}*"
    # ind = pos + 1
    ind = pos
    r = ""
    for i in range(s):
        r += get_char(line, ind - (s - i))
    for i in range(1, s + 1):
        r += get_char(line, ind + i)
    return r


def one_hot_encode(letter, alphabet):
    vec = np.zeros(len(alphabet))
    if letter in alphabet:
        vec[alphabet.index(letter)] = 1
    return vec

# Function to encode a 4-letter context
def encode_context(context, alphabet):
    encoded = []
    for letter in context:
        encoded.extend(one_hot_encode(letter, alphabet))
    return np.array(encoded)