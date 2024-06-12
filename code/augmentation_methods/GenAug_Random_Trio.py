#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
main_path = "data"
os.chdir(main_path)
#!nvidia-smi


# In[ ]:


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

import io, json, collections, pprint, time
import random
import string
import re
import unicodedata
from string import punctuation
string.ascii_lowercase


# # Random Trio

# ### Note: based on the code for EDA (Wei and Kou, 2019) (https://arxiv.org/abs/1901.11196) found here: https://github.com/jasonwei20/eda_nlp

# In[ ]:


import random
from random import shuffle
random.seed(54321)

#stopwords list
def get_stopwords(path):
    f = open(path, 'r')
    stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    return stopwords


# ## Random Deletion

# In[ ]:


def random_deletion(words, p):
    
	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words


# ## Random Swap

# In[ ]:


def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words


# ## Random Insertion

# In[ ]:


def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)


# In[ ]:


def random_insertion(words, n, stopwords):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, stopwords)
    return new_words

def add_word(new_words, stopwords):
    new_words_2 = [word for word in new_words if word not in stopwords]
    if len(new_words_2) == 0:
        return []
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words_2[random.randint(0, len(new_words_2)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


# ## Main Random Trio Code

# In[ ]:


def random_trio(sentence, alpha, stopwords):
	
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    n1 = max(1, int(alpha[0]*num_words))
    n2 = max(1, int(alpha[1]*num_words))

    #ri
    a_words = random_insertion(words, n1, stopwords)
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        insertion_sentence_1 = '<blank>'
    else:
        insertion_sentence_1 = ' '.join(a_words)
        insertion_sentence_1 = re.sub(' +', ' ', insertion_sentence_1)
    a_words = random_insertion(words, n2, stopwords)
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        insertion_sentence_2 = '<blank>'
    else:
        insertion_sentence_2 = ' '.join(a_words)
        insertion_sentence_2 = re.sub(' +', ' ', insertion_sentence_2)
    insertion_sentences = insertion_sentence_1 + '\t' + insertion_sentence_2

    #rs
    a_words = random_swap(words, n1)
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        swap_sentence_1 = '<blank>'
    else:
        swap_sentence_1 = ' '.join(a_words)
    a_words = random_swap(words, n2)
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        swap_sentence_2 = '<blank>'
    else:
        swap_sentence_2 = ' '.join(a_words)
    swap_sentences = swap_sentence_1 + '\t' + swap_sentence_2

    #rd
    a_words = random_deletion(words, alpha[0])
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        deletion_sentence_1 = '<blank>'
    else:
        deletion_sentence_1 = ' '.join(a_words)
    a_words = random_deletion(words, alpha[1])
    if len(a_words) == 0 or ' '.join(a_words) == sentence:
        deletion_sentence_2 = '<blank>'
    else:
        deletion_sentence_2 = ' '.join(a_words)
    deletion_sentences = deletion_sentence_1 + '\t' + deletion_sentence_2

    return insertion_sentences, swap_sentences, deletion_sentences


# In[ ]:


def main_random_trio(input_file, alpha, stopwords):
    f = open(input_file, 'r')
    sentences = f.readlines()
    sentences = [s.strip() for s in sentences]
    insertion_lst = []
    swap_lst = []
    deletion_lst = []
    counter = 0
    for sentence in sentences:
        counter += 1
        insertion_sentence, swap_sentence, deletion_sentence = random_trio(sentence, alpha, stopwords)
        insertion_lst.append(insertion_sentence)
        swap_lst.append(swap_sentence)
        deletion_lst.append(deletion_sentence)
        if counter % 1000 == 0:
            print(counter)
            print("Sentence: ", sentence)
            print("Insertion_sentence: ", insertion_sentence)
            print("Swap_sentence: ", swap_sentence)
            print("Deletion_sentence: ", deletion_sentence,'\n')
    #print(len([x for x in insertion_lst if x != "<blank>\t<blank>"]), len([x for x in swap_lst if x != "<blank>\t<blank>"]), len([x for x in deletion_lst if x != "<blank>\t<blank>"]))
    return insertion_lst, swap_lst, deletion_lst


# In[ ]:


def write_random_prompts(insertion_lst, swap_lst, deletion_lst, output_file_lst):
    files_lst = [open(output_file_lst[i], 'w') for i in range(len(output_file_lst))]
    print("Writing output prompts to files...")
    files_lst[0].write('\n'.join(insertion_lst))
    files_lst[1].write('\n'.join(swap_lst))
    files_lst[2].write('\n'.join(deletion_lst))
    for i in range(len(files_lst)):
        files_lst[i].close()
    print("Output prompts written to files")


# ## Execution Code

# In[ ]:


import time
random.seed(54321)

stopwords_path = 'stopwords.txt'
stopwords = get_stopwords(stopwords_path)

alpha = [0.05, 0.10]
input_file = 'yelp_train.txt'
output_files = ['yelp_train_random_insert.txt','yelp_train_random_swap.txt','yelp_train_random_delete.txt']

start = time.time()
insertion_lst, swap_lst, deletion_lst = main_random_trio(input_file, alpha, stopwords)
write_random_prompts(insertion_lst, swap_lst, deletion_lst, output_files)
end = time.time()
print(end - start)

