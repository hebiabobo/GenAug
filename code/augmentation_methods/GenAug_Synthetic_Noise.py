#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
main_path = "data"
os.chdir(main_path)
#!nvidia-smi


# In[ ]:


import io, json, collections, pprint, time
import random
import string
import re
import unicodedata
from string import punctuation
string.ascii_lowercase


def add_letters(word, probability):
    k = len(word)
    i = 0
    while i < k:
        if random.random() < probability:
            word = word[:i] + random.choice(string.ascii_lowercase) + word[i:]
            i += 1
            k += 1
        i += 1
    if random.random() < probability:
        word = word[:len(word)] + random.choice(string.ascii_lowercase)
    return word


def remove_letters(word, probability):
    for i in range(len(word)-1):
        if random.random() < probability:
            word = word[:i] + " " + word[i+1:]
    if random.random() < probability:
        word = word[:len(word)-1]
    word = re.sub(" ", "", word)
    return word


def swap_letters(word, probability):
    k = len(word)
    if k < 2:
        return word
    else:
        i = 0
        skip = False
        for i in range(k-2):
            if skip == False and random.random() < probability:
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
                skip = True
            else:
                skip = False
        if skip == False and random.random() < probability:
            word = word[:k-2] + word[k-1] + word[k-2]
    return word


# add noise without swapping (only insertions and deletions)
def add_noise(word, probability):
    word = remove_letters(word, probability/3)
    word = add_letters(word, probability/3)
    return word


# add noise with swapping
def add_noise_swaps(word, probability):
    word = swap_letters(word, probability/3)
    word = remove_letters(word, probability/3)
    word = add_letters(word, probability/3)
    return word


# add noise with swapping but ignore first and last char of every word (this version used in the paper)
def add_noise_swaps_v2(word, probability):
    new_word = swap_letters(word[1:-1], probability/3)
    new_word = remove_letters(new_word, probability/3)
    new_word = add_letters(new_word, probability/3)
    return word[0] + new_word + word[-1]


#Example:
random.seed(54321)
word = "subreddit"
probability = 0.10
new_word = add_noise(word, probability)
print(new_word)
new_word_2 = add_noise_swaps(word, probability)
print(new_word_2)
new_word_3 = add_noise_swaps_v2(word, probability)
print(new_word_3)


# In[ ]:


def synthetic_noise_main(prompt, prob_lst, fnc):
    new_prompt_lst = []
    noise_words = prompt.split('\t')[0].split()
    clean_string = prompt.split('\t')[1]
    for prob in prob_lst:
        new_words = []
        for word in noise_words:
            if len(word) > 1:
                new_word = fnc(word, prob)
                new_words.append(new_word)
            else:
                new_words.append(word)
        new_prompt = ' '.join(new_words) + ' ' + clean_string
        new_prompt_lst.append(new_prompt)
    return new_prompt_lst


def main(input_file, output_file, prob_lst, fnc):
    main_prompt_lst = []
    with open(input_file) as in_f:
        input_prompts = in_f.readlines()
    counter = 0
    for prompt in input_prompts:
        new_prompt_lst = synthetic_noise_main(prompt.strip('\n'), prob_lst, fnc)
        if new_prompt_lst is not None:
            main_prompt_lst.append('\t'.join(new_prompt_lst))
        counter += 1
    with open(output_file, 'w') as out_f:
        out_f.write('\n'.join(main_prompt_lst))
    print("Lines written to file")
    return main_prompt_lst


# In[ ]:


random.seed(54321)
fnc = add_noise_swaps_v2
prob_lst = [0.05, 0.10, 0.15]
input_file = 'yelp_train_0.5.txt'
output_file = 'yelp_train_synthetic-noise.txt'

main_prompt_lst = main(input_file, output_file, prob_lst, fnc)

