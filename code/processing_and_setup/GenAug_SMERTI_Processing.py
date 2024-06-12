#!/usr/bin/env python
# coding: utf-8

# # FOR TRAINING:

# ## Setup SMERTI Train/Val Data

# ### Total 30-word windows

# In[ ]:


import os
import io
main_path = "data"
os.chdir(main_path)


# In[ ]:


import math
import random
import io


def get_smerti_data(file, amount):
    f = open(file, 'r')
    lines = f.readlines()
    chosen_lines = random.sample(lines, amount)
    total_lines = []
    for l in chosen_lines:
        words = l.strip('\n').split()
        if len(words) <= 25:
            total_lines.append(l)
            continue
        else:
            num_chunks = math.ceil((len(words)-10)/20)+1
            if len(words) % 20 <= 5: # and len([x for x in words[-(len(words)) % 20] if x not in punctuation]) 
                num_chunks = num_chunks-1
        for i in range(0, num_chunks):
            if i == 0:
                chunk = words[0:20]
                total_lines.append(' '.join(chunk)+'\n')
            else:
                if len(words[10+(i-1)*20+30:]) <= 5:
                    chunk = words[10+(i-1)*20:]
                else:
                    chunk = words[10+(i-1)*20:10+(i-1)*20+30]
                chunk_text = ' '.join(chunk[10:])
                if len(chunk) > 10 and len(chunk_text.strip()) > 0:
                    total_lines.append(' '.join(chunk)+'\n')
    random.shuffle(total_lines)
    return total_lines


def write_lines(lines, path):
    f = io.open(path, "w", encoding = 'utf-8')
    print("Currently writing lines to file ...")
    f.writelines(lines)
    f.close()
    print("Lines successfully written to file!")


# In[ ]:


random.seed(12345)

input_file = 'yelp_val.txt'
output_file = 'yelp_val_SMERTI.txt'
lines = get_smerti_data(input_file, 7500)
write_lines(lines, output_file)

input_file = 'yelp_train.txt'
output_file = 'yelp_train_SMERTI.txt'
lines = get_smerti_data(input_file, 25000)
write_lines(lines, output_file)


# ## SMERTI Masking Data

# In[ ]:


# -*- coding: utf-8 -*-
import io, json, os, collections, pprint, time
import re
from string import punctuation
import random
from random import sample
import math
import unicodedata
import sys


def mask_all(path):
    mask_list = []
    f = io.open(path, encoding = 'utf-8')
    lines = f.readlines()
    total = len(lines)
    counter = 0
    print("Currently reading lines from file ...")
    for l in lines:
        if 0 <= counter <= int(round(total / 3)):
            masked_text = mask_text(l, 0.20)
        elif int(round(total / 3)) < counter <= int(round((total * 2) / 3)):
            masked_text = mask_text(l, 0.40)
        else:
            masked_text = mask_text(l, 0.60)
        mask_list.append(masked_text + '\n')
        counter += 1
    return mask_list


def mask_text(line, value):
    word_list = (line.rstrip()).split()
    num = int(round(len(word_list) * value))
    mask_locs = set(sample(range(len(word_list)), num))
    masked = list(('[mask]' if i in mask_locs and word_list[i] not in punctuation else c for i,c in enumerate(word_list)))
    masked_groups = mask_groupings(masked)
    masked_text = ' '.join(masked_groups)
    return masked_text


def mask_groupings(masked_list):
    masked_group_list = []
    previous_element = ""
    for element in masked_list:
        if element != "[mask]":
            masked_group_list.append(element)
        elif element == "[mask]":
            if element != previous_element:
                masked_group_list.append(element)
        previous_element = element
    return masked_group_list


def write_file(lst, path):
    f = io.open(path, "w", encoding = 'utf-8')
    print("Currently writing lines to file ...")
    f.writelines(lst)
    f.close()
    print("Lines successfully written to file!")


# In[ ]:


random.seed(12345)

input_path = 'yelp_train_SMERTI.txt'
output_path = 'yelp_train_SMERTI_masked.txt'
mask_lst = mask_all(input_path)
write_file(mask_lst, output_path)


# # FOR INFERENCE:

# ## Split Training Data into Batches of 1000 and Combine SMERTI Outputs

# ### This is necessary as SMERTI inference takes a while to run, so we split the data into batches of 1000 examples and then recombine the outputs afterwards

# In[ ]:


import math

def chunk_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    num_chunks = math.ceil(len(lines)/1000)
    print("Writing chunks to files...")
    for i in range(num_chunks):
        chunk = lines[i*1000:(i+1)*1000]
        out_f = open('{}_p{}_SMERTI.txt'.format(path, i+1), 'w')
        out_f.writelines(chunk)
        out_f.close()
    print("Chunks written to files")


# In[ ]:


input_path = 'yelp_train.txt'
chunk_file(input_path)


# In[ ]:


def combine_files(num_input_files, output_path):
    final_lines = []
    files = []
    for i in range(num_input_files):
        input_file = open('yelp_train_p{}_SMERTI_outputs.txt'.format(i+1),'r')
        files.append(input_file)
    for f in files:
        lines = f.readlines()
        for l in lines:
            final_lines.append(l.strip('\n')+'\n')
    output_file = open(output_path, 'w')
    output_file.writelines(final_lines)


# In[ ]:


num_input_files = 50
output_path = 'yelp_train_SMERTI_outputs.txt'
combine_files(num_input_files, output_path)


# ## Get Training Vocab and Nouns

# In[ ]:


# Spacy for POS tagging
get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
import spacy
nlp = spacy.load("en_core_web_sm") 


# In[ ]:


def extract_POS(word):
    tokenized_word = nlp(word)
    for token in tokenized_word:
        if token.pos_ == "NOUN":
            return True
        else:
            return False

#Example:
prompt = "i hate this restaurant . the food is horrible !"
word = 'cooking'
POS_dict = extract_POS(word)
print(POS_dict)


# In[ ]:


from collections import defaultdict
from string import punctuation

def get_vocab(file):
    word_dict = defaultdict(int)
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        for w in [i for i in words if i not in punctuation]:
            word_dict[w] += 1
    print(len(word_dict))
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    return word_dict


def get_nouns(vocab_file):
    word_dict = defaultdict(int)
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    counter = 0
    for line in lines:
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        elements = line.split('\t')
        word = elements[0]
        count = int(elements[1].strip())
        if extract_POS(word) == True:
            word_dict[word] = count
    print(len(word_dict))
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    return word_dict


def write_words(word_dict, file):
    f = open(file, 'w')
    for word, count in word_dict.items():
        f.write(word + '\t' + str(count) + '\n')
    f.close()


# In[ ]:


input_file = 'SMERTI/yelp_train_SMERTI.txt'
output_file_vocab = 'SMERTI/yelp_train_SMERTI_vocab.txt'
output_file_nouns = 'SMERTI/yelp_train_SMERTI_nouns.txt'

word_dict = get_vocab(input_file)
write_words(word_dict, output_file_vocab)

noun_dict = get_nouns(output_file_vocab)
write_words(noun_dict, output_file_nouns)

