#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
main_path = "data"
os.chdir(main_path)


# ## Keyword Extraction & POS Tagging

# In[ ]:


#!wget https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip
#!unzip stanford-postagger-2018-10-16.zip
#!mv stanford-postagger-2018-10-16.zip stanford-postagger
#!python


# In[ ]:


# Import Stanford POS Tagger
from nltk.tag.stanford import StanfordPOSTagger
_path_to_model = 'stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
_path_to_jar = 'stanford-postagger-2018-10-16/stanford-postagger.jar'
st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Install RAKE for keyword extraction
# get_ipython().system('pip3 install python-rake')
import RAKE
Rake = RAKE.Rake("stopwords.txt")


# In[ ]:


# Import Stanford POS Tagger
import nltk
nltk.download('punkt')
nltk.download('tagsets')
from nltk.tag.stanford import StanfordPOSTagger
_path_to_model = 'stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
_path_to_jar = 'stanford-postagger-2018-10-16/stanford-postagger.jar'
stanford_tagger = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)
from collections import defaultdict
import numpy as np
from nltk.data import load
tagdict = load('help/tagsets/upenn_tagset.pickle')


# In[ ]:


def extract_keywords_and_POS(prompt):
    POS_dict = {}
    try:
        tagged_prompt = st.tag(prompt.split())
    except:
        print("ERROR PROMPT: ", prompt)
        return False
    else:
        for pair in tagged_prompt:
            POS_dict[pair[0]] = pair[1]
        keywords_dict = {}
        #format: Rake.run(prompt, minCharacters = X, maxWords = Y, minFrequency = Z)
        keywords = Rake.run(prompt)
        for pair in keywords:
            words = pair[0].split()
            for word in words:
                try:
                    keywords_dict[word] = POS_dict[word]
                except:
                    pass
        return keywords_dict

#Example:
prompt = "first thing we do , let's fight all the lawyers"
keywords_dict = extract_keywords_and_POS(prompt)
print(keywords_dict)


# ## WordNet: Hypernyms

# In[ ]:


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def get_hypernyms(word, pos):
    hypers_lst = []
    try:
        s = wordnet.synsets(word, pos)[0]
    except:
        try:
            s = wordnet.synsets(word)[0]
        except:
            return hypers_lst
    if s.name() == 'restrain.v.01':
        print("RESTRAIN ENCOUNTERED (hypers)")
        return hypers_lst
    hypers = lambda s:s.hypernyms()
    hypers = list(s.closure(hypers))
    for syn in hypers:
        for l in syn.lemmas():
            if l.name().lower() != word:
                hypers_lst.append(l.name().lower())
    return list(dict.fromkeys(hypers_lst))

#Example:
print(get_hypernyms("person", "n"))


# In[ ]:


import random
import re

def single_prompt_helper(keywords_lst, keywords_dict, fnc, chosen_nums):
    counter = 1
    chosen_keywords_lst = []
    chosen_replacements_lst = []
    for i in range(0,len(keywords_lst)):
        if counter <= max(chosen_nums):
            keyword = keywords_lst[i]
            keyword_pos = keywords_dict[keyword][0].lower()
            if keyword_pos == 'j':
                keyword_pos = 'a'
            candidates = fnc(keyword, keyword_pos)
            if len(candidates) != 0:
                counter += 1
                chosen_keywords_lst.append(keyword)
                chosen_replacement = candidates[0]
                chosen_replacements_lst.append(chosen_replacement)
        else:
            return chosen_keywords_lst, chosen_replacements_lst
    return chosen_keywords_lst, chosen_replacements_lst


def single_prompt_wordnet(prompt, nums_lst):
    original_prompt = prompt
    hypernyms_prompt_lst = []
    keywords_dict = extract_keywords_and_POS(prompt)
    if keywords_dict == False:
        return []
    keywords_lst = list(keywords_dict.keys())
    num_keywords = len(keywords_lst)
    prompt_hypernym = original_prompt
    chosen_keywords, chosen_hypernyms = single_prompt_helper(keywords_lst, keywords_dict, get_hypernyms, nums_lst)    
    counter = 1
    for chosen_word, chosen_hyper in zip(chosen_keywords, chosen_hypernyms):
        prompt_hypernym = re.sub(r"\b%s\b" % chosen_word, chosen_hyper, prompt_hypernym)
        if counter in nums_lst:
            hypernyms_prompt_lst.append(re.sub('_',' ',prompt_hypernym))
        counter += 1
    return hypernyms_prompt_lst


#Example:
nums_lst = [1,2,3]
prompt = "an immortal being is explaining"
hypers_lst = single_prompt_wordnet(prompt,nums_lst)
print(hypers_lst)


# In[ ]:


def main_wordnet(input_file, output_file, nums_lst):
    hypers_prompt_lst = []
    hypers_counter = 0
    with open(input_file) as in_f:
        input_prompts = in_f.readlines()
    counter = 0
    for prompt in input_prompts:
        hypers_lst = single_prompt_wordnet(prompt.strip('\n'), nums_lst)
        if hypers_lst is not None and len(hypers_lst) > 0:
            hypers_counter += 1
            hypers_prompt_lst.append('\t'.join(hypers_lst)+'\n')
        else:
            hypers_prompt_lst.append('<blank>\n')
        if counter % 100 == 0:
            print(counter)
            write_wordnet_prompts(hypers_prompt_lst, output_file)
            hypers_prompt_lst = []
        counter += 1
    write_wordnet_prompts(hypers_prompt_lst, output_file)
    print("Final hypernym lines: ", hypers_counter)
    return hypers_prompt_lst


def write_wordnet_prompts(hypers_lst, output_file):
    f = open(output_file, 'a')
    print("Writing output prompts to file...")
    f.writelines(hypers_lst)
    f.close()
    print("Output prompts written to file\n")


# In[ ]:


import time

random.seed(54321)
nums_lst = [1,2,3]
input_file = 'yelp_train.txt'
output_file = 'yelp_train_hypernyms.txt'

start = time.time()
hypers_lst = main_wordnet(input_file, output_file, nums_lst)
end = time.time()
print(end - start)

