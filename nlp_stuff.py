
from collections import defaultdict

def void_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    breaker = False
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                term_present = True
                breaker = True
                break
        if not breaker:
            c_list.append(sentence.strip())
        breaker = False
    if return_bool:
        return not term_present
    else:
        return c_list

def validate_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                c_list.append(sentence.strip())
                term_present = True
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                break
    if return_bool:
        return term_present
    else:
        return c_list


def word_salad(pd_series, n_word_split = 1, walk_interval = 1):
    word_c = defaultdict(int)

    useless_words = 'the, of, and, with, is, for, a, in, to, as, there, be, an, or, if, at, are'.split(', ')
    useless_words.append('')

    # Only after splitting the impression by a period to get a list.
    for row in pd_series:
        for item in row:
            split_words = item.split(' ')
            split_words = [' '.join(split_words[i: i + n_word_split]) for i in range(0, len(split_words) - (n_word_split - walk_interval), walk_interval)]
            for i_word in split_words:
                if i_word not in useless_words:
                    word_c[i_word] += 1

    sorted_word_salad = sorted(word_c.items(), key=lambda x: x[1], reverse=True)

    return sorted_word_salad

def non_empty_list(x):
    return len(x) > 0

