import re
import collections

PATH = 'cipher.txt'


def func(e):
    return e[1]


def to_list_fre(txt):
    txt = re.sub('\s', '', txt)
    c = len(txt)
    l = collections.Counter(txt).most_common();
    for i, k in enumerate(l):
        y = list(k)
        y[1] = y[1] / c
        l[i] = tuple(y)
    l.sort(key=func, reverse=True)
    return l


def reconstruct_key(txt):
    l = to_list_fre(txt)
    temp = []
    for i in l:
        temp.append(i[0])

    letters_ordered_by_freq = ["E", "T", "A", "O", "I", "N", "S", "R", "H", "D", "L",
                               "U", "C", "M", "F", "Y", "W", "G", "P", "B", "V", "K", "X", "Q", "J", "Z"]
    return dict(zip(temp, letters_ordered_by_freq[:len(temp)]))


def decode_raw(txt):
    dict = reconstruct_key(txt)
    ans = ''
    for i in txt:
        if (i == ' '):
            ans = ans + ' '
            continue
        if (i == '\n'):
            ans = ans + '         \n'
            continue
        ans = ans + dict[i]
    return ans


def decode(txt):
    data = [
        ('r', 'E'), ('b', 'T'), ('m', 'A'),
        ('k', 'N'), ('j', 'O'), ('w', 'I'),
        ('i', 'S'), ('p', 'H'), ('u', 'R'),
        ('d', 'D'), ('h', 'L'), ('v', 'C'),
        ('x', 'F'), ('y', 'M'), ('n', 'U'),
        ('s', 'P'), ('t', 'Y'), ('l', 'B'),
        ('o', 'G'), ('q', 'K'), ('a', 'X'),
        ('c', 'W'), ('e', 'V'), ('g', 'Z'),
        ('f', 'Q')
    ]
    dict = {}
    for item in data:
        dict[item[0]] = item[1]
    ans = ''
    for i in txt:
        if (i == ' '):
            ans = ans + ' '
            continue
        if (i == '\n'):
            ans = ans + '\n'
            continue
        ans = ans + dict[i]
    return ans


if __name__ == '__main__':
    f = open(PATH, 'r')
    txt = f.read()
    print(to_list_fre(txt))
    print(encode(txt))


