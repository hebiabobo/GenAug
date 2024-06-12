#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
main_path = "data"
os.chdir(main_path)
#!nvidia-smi


# ## Keyword Extraction

# In[ ]:


# Install RAKE for keyword extraction
get_ipython().system('pip3 install python-rake')
import RAKE
Rake = RAKE.Rake("stopwords.txt")


# In[ ]:


def extract_keywords_len(prompt):
    #format: Rake.run(prompt, minCharacters = X, maxWords = Y, minFrequency = Z)
    keywords = Rake.run(prompt)
    length = len(keywords)
    return length

#Example:
prompt = "first thing we do , let's fight all the lawyers"
keywords_len = extract_keywords_len(prompt)
print(keywords_len)


# ## Yelp Splits

# In[ ]:


import io, json, os, collections, pprint, time
import re
from string import punctuation
import unicodedata
import random


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def process_text(s):
    s = unicodeToAscii(s.lower().strip()) 
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)
    s = re.sub('\$+', '$', s)        
    s = re.sub("[^a-zA-Z0-9$.!?,'']+", ' ', s)
    for p in punctuation:
        if p != "'":
            s = s.replace(p, " " + p + " ")       
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


# In[ ]:


def filter_reviews(path):
    review_list = []
    one_star_list = []
    two_star_list = []
    three_star_list = []
    four_star_list = []
    five_star_list = []
    review_list_2 = []
    one_star_list_2 = []
    two_star_list_2 = []
    three_star_list_2 = []
    four_star_list_2 = []
    five_star_list_2 = []
    total_count = 0
    one_star_count = 0
    two_star_count = 0
    three_star_count = 0
    four_star_count = 0
    five_star_count = 0
    f = io.open(path, encoding = 'utf-8')
    counter = 0
    print("Currently reading lines from file ...")
    for l in f:
        if counter % 100000 == 0:
            print("Read in {%d} lines" % counter)
        jline = json.loads(l)
        if jline['text'] != '' and isEnglish(jline['text'])             and 'http' not in jline['text'].lower() and 'www' not in jline['text'].lower():
            clean_line = re.sub('\s+', ' ', jline['text']).strip()
            clean_line_final = process_text(clean_line)
            total_count += 1
            if jline['stars'] == 1.0:
                one_star_count += 1
            elif jline['stars'] == 2.0:
                two_star_count += 1
            elif jline['stars'] == 3.0:
                three_star_count += 1
            elif jline['stars'] == 4.0:
                four_star_count += 1
            elif jline['stars'] == 5.0:
                five_star_count += 1
            if len(clean_line_final.split()) <= 40:
                num_keywords = extract_keywords_len(clean_line_final)
                if num_keywords >= 4:
                    if jline['stars'] == 1.0:
                        one_star_list.append(clean_line_final + '\t' + '0' + '\n')
                    elif jline['stars'] == 2.0:
                        two_star_list.append(clean_line_final + '\t' + '0.25' + '\n')
                    elif jline['stars'] == 3.0:
                        three_star_list.append(clean_line_final + '\t' + '0.5' + '\n')
                    elif jline['stars'] == 4.0:
                        four_star_list.append(clean_line_final + '\t' + '0.75' + '\n')
                    elif jline['stars'] == 5.0:
                        five_star_list.append(clean_line_final + '\t' + '1' + '\n')
                    review_list.append(clean_line_final + '\t' + str((jline['stars']-1)/4) + '\n')
                else:
                    if jline['stars'] == 1.0:
                        one_star_list_2.append(clean_line_final + '\t' + '0' + '\n')
                    elif jline['stars'] == 2.0:
                        two_star_list_2.append(clean_line_final + '\t' + '0.25' + '\n')
                    elif jline['stars'] == 3.0:
                        three_star_list_2.append(clean_line_final + '\t' + '0.5' + '\n')
                    elif jline['stars'] == 4.0:
                        four_star_list_2.append(clean_line_final + '\t' + '0.75' + '\n')
                    elif jline['stars'] == 5.0:
                        five_star_list_2.append(clean_line_final + '\t' + '1' + '\n')
                    review_list_2.append(clean_line_final + '\t' + str((jline['stars']-1)/4) + '\n')
        counter += 1
    return review_list, one_star_list, two_star_list, three_star_list, four_star_list, five_star_list, review_list_2, one_star_list_2, two_star_list_2, three_star_list_2, four_star_list_2, five_star_list_2, one_star_count, two_star_count, three_star_count, four_star_count, five_star_count, total_count


# In[ ]:


def write_lines(lst, output_path_1, output_path_2):
    f1 = io.open(output_path_1, "w", encoding = 'utf-8')
    print("Currently writing lines to file1 ...")
    f1.writelines(lst)
    f1.close()
    print("Lines successfully written to file1!")

    f2 = io.open(output_path_2, "w", encoding = 'utf-8')
    print("Currently writing lines to file2 ...")
    for line in lst:
        new_line = line.split('\t')[0] + '\n'
        f2.write(new_line)
    f2.close()
    print("Lines successfully written to file2!")


# # Execution Code

# In[ ]:


random.seed(12345)
input_path = os.path.join(main_path, 'review.json') #this is the original Yelp Reviews dataset file containing all reviews

import time
start = time.time()
review_list, one_star_list, two_star_list, three_star_list, four_star_list, five_star_list, review_list_2, one_star_list_2, two_star_list_2, three_star_list_2, four_star_list_2, five_star_list_2, one_star_count, two_star_count, three_star_count, four_star_count, five_star_count, total_count = filter_reviews(input_path)
end = time.time()
print(end-start)


# In[ ]:


print("Number of total reviews: ", total_count)
print("Number of one star reviews: ", one_star_count)
print("Number of two star reviews: ", two_star_count)
print("Number of three star reviews: ", three_star_count)
print("Number of four star reviews: ", four_star_count)
print("Number of five star reviews: ", five_star_count)


# ## For Yelp Low-Resource (YLR)

# In[ ]:


total_reviews = len(review_list)
print("Number of total reviews: ", total_reviews)

one_star_reviews = len(one_star_list)
print("Number of one star reviews: ", one_star_reviews)

two_star_reviews = len(two_star_list)
print("Number of two star reviews: ", two_star_reviews)

three_star_reviews = len(three_star_list)
print("Number of three star reviews: ", three_star_reviews)

four_star_reviews = len(four_star_list)
print("Number of four star reviews: ", four_star_reviews)

five_star_reviews = len(five_star_list)
print("Number of five star reviews: ", five_star_reviews)

random.shuffle(one_star_list)
random.shuffle(two_star_list)
random.shuffle(three_star_list)
random.shuffle(four_star_list)
random.shuffle(five_star_list)


# In[ ]:


import math
random.seed(12345)

train_one_star = one_star_list[:math.ceil(50000*(one_star_count/total_count))]
train_two_star = two_star_list[:math.ceil(50000*(two_star_count/total_count))]
train_three_star = three_star_list[:math.ceil(50000*(three_star_count/total_count))]
train_four_star = four_star_list[:math.ceil(50000*(four_star_count/total_count))]
train_five_star = five_star_list[:math.ceil(50000*(five_star_count/total_count))]

valid_one_star = one_star_list[math.ceil(50000*(one_star_count/total_count)):math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))]
valid_two_star = two_star_list[math.ceil(50000*(two_star_count/total_count)):math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))]
valid_three_star = three_star_list[math.ceil(50000*(three_star_count/total_count)):math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))]
valid_four_star = four_star_list[math.ceil(50000*(four_star_count/total_count)):math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))]
valid_five_star = five_star_list[math.ceil(50000*(five_star_count/total_count)):math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))]

valid_sent_one_star = one_star_list[math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count)):math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))]
valid_sent_two_star = two_star_list[math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count)):math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))]
valid_sent_three_star = three_star_list[math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count)):math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))]
valid_sent_four_star = four_star_list[math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count)):math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))]
valid_sent_five_star = five_star_list[math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count)):math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))]

test_one_star = one_star_list[math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count)):math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(5000*(one_star_count/total_count))]
test_two_star = two_star_list[math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count)):math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(5000*(two_star_count/total_count))]
test_three_star = three_star_list[math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count)):math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(5000*(three_star_count/total_count))]
test_four_star = four_star_list[math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count)):math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(5000*(four_star_count/total_count))]
test_five_star = five_star_list[math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count)):math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(5000*(five_star_count/total_count))]

end_index_one = math.ceil(50000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(15000*(one_star_count/total_count))+math.ceil(5000*(one_star_count/total_count))
end_index_two = math.ceil(50000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(15000*(two_star_count/total_count))+math.ceil(5000*(two_star_count/total_count))
end_index_three = math.ceil(50000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(15000*(three_star_count/total_count))+math.ceil(5000*(three_star_count/total_count))
end_index_four = math.ceil(50000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(15000*(four_star_count/total_count))+math.ceil(5000*(four_star_count/total_count))
end_index_five = math.ceil(50000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(15000*(five_star_count/total_count))+math.ceil(5000*(five_star_count/total_count))

final_train_reviews = train_one_star + train_two_star + train_three_star + train_four_star + train_five_star
final_valid_reviews = valid_one_star + valid_two_star + valid_three_star + valid_four_star + valid_five_star
final_valid_sent_reviews = valid_sent_one_star + valid_sent_two_star + valid_sent_three_star + valid_sent_four_star + valid_sent_five_star
final_test_reviews = test_one_star + test_two_star + test_three_star + test_four_star + test_five_star

random.shuffle(final_train_reviews)
random.shuffle(final_valid_reviews)
random.shuffle(final_valid_sent_reviews)
random.shuffle(final_test_reviews)


# In[ ]:


final_train_reviews = final_train_reviews[:50000]
final_valid_reviews = final_valid_reviews[:15000]
final_valid_sent_reviews = final_valid_sent_reviews[:15000]
final_test_reviews = final_test_reviews[:5000]

print(len(final_train_reviews))
print(len(final_valid_reviews))
print(len(final_valid_sent_reviews))
print(len(final_test_reviews))


# In[ ]:


#files with stars and "yelp_val_sent" are for finetuning the BERT sentiment regressor

train_path_1 = os.path.join(main_path, 'yelp_train_stars.txt')
train_path_2 = os.path.join(main_path, 'yelp_train.txt')
valid_path_1 = os.path.join(main_path, 'yelp_val_stars.txt')
valid_path_2 = os.path.join(main_path, 'yelp_val.txt')
test_path_1 = os.path.join(main_path, 'yelp_test_stars.txt')
test_path_2 = os.path.join(main_path, 'yelp_test.txt')
valid_sent_path_1 = os.path.join(main_path, 'yelp_val_sent_stars.txt')
valid_sent_path_2 = os.path.join(main_path, 'yelp_val_sent.txt')

write_lines(final_train_reviews, train_path_1, train_path_2)
write_lines(final_valid_reviews, valid_path_1, valid_path_2)
write_lines(final_test_reviews, test_path_1, test_path_2)
write_lines(final_valid_sent_reviews, valid_sent_path_1, valid_sent_path_2)


# ## For Finetuning GPT-2 for Perplexity/SLOR Evaluation (2 million reviews)

# In[ ]:


def filter_reviews_full(path):
    review_list = []
    f = io.open(path, encoding = 'utf-8')
    counter = 0
    print("Currently reading lines from file ...")
    for l in f:
        if counter % 100000 == 0:
            print("Read in {%d} lines" % counter)
        jline = json.loads(l)
        if jline['text'] != '' and isEnglish(jline['text'])             and 'http' not in jline['text'].lower() and 'www' not in jline['text'].lower():
            clean_line = re.sub('\s+', ' ', jline['text']).strip()
            clean_line_final = process_text(clean_line)
            if len(clean_line_final.strip()) != 0:
                review_list.append(clean_line_final + '\t' + str((jline['stars']-1)/4))
        counter += 1
    print("Length of review_list: ", len(review_list))
    return review_list


# In[ ]:


def write_lines_full(lst, output_path_1, output_path_2):
    f1 = io.open(output_path_1, "w", encoding = 'utf-8')
    print("Currently writing lines to file1 ...")
    f1.write('\n'.join(lst))
    f1.close()
    print("Lines successfully written to file1!")

    f2 = io.open(output_path_2, "w", encoding = 'utf-8')
    print("Currently writing lines to file2 ...")
    for line in lst:
        new_line = line.split('\t')[0] + '\n'
        f2.write(new_line)
    f2.close()
    print("Lines successfully written to file2!")


# In[ ]:


input_path = 'review.json' #original Yelp Reviews dataset (over 6 million reviews)
output_path_train_stars = 'yelp_train_full_stars.txt'
output_path_train = 'yelp_train_full.txt'
output_path_val_stars = 'yelp_val_full_stars.txt'
output_path_val = 'yelp_val_full.txt'

import time
start = time.time()
#review_list = filter_reviews_full(input_path)
end = time.time()
print(end-start)


# In[ ]:


random.seed(12345)
random.shuffle(review_list)
train_lst = review_list[:2000000]
val_lst = review_list[2000000:2500000]
#write_lines_full(train_lst, output_path_train_stars, output_path_train)
#write_lines_full(val_lst, output_path_val_stars, output_path_val)


# ## Get Unigram Frequencies

# In[ ]:


###This will be used for SLOR normalization (on yelp_full (2 million reviews)) and rare_words metric (on YLR)


# In[ ]:


get_ipython().system('pip install transformers==2.5.1')


# In[ ]:


import io, json, os, collections, pprint, time
import re
from string import punctuation
import unicodedata
import random
from collections import defaultdict
from transformers import GPT2Tokenizer

lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_unigrams_test(lst):
    unigrams_dict = defaultdict(lambda: [0,0])
    counter = 0
    for l in lst:
        tokenized_line = lm_tokenizer.tokenize(l)
        print("tokenized_line: ", tokenized_line)
        for token in tokenized_line:
            unigrams_dict[token][0] += 1
        counter += 1
    print("Length of unigrams_dict: ", len(unigrams_dict))
    total_freq = 0
    for k,v in unigrams_dict.items():
        total_freq += v[0]
    print("Total_freq: ", total_freq)
    for k,v in unigrams_dict.items():
        v[1] = v[0]/total_freq
    unigrams_dict = {k: v for k, v in sorted(unigrams_dict.items(), key=lambda item: item[1][1], reverse=True)}
    return unigrams_dict


def get_unigrams(path):
    unigrams_dict = defaultdict(lambda: [0,0])
    f = io.open(path, encoding = 'utf-8')
    counter = 0
    print("Currently getting unigrams from file ...")
    for l in f:
        if counter % 100000 == 0:
            print("Read in {%d} lines" % counter)
        tokenized_line = lm_tokenizer.tokenize(l)
        for token in tokenized_line:
            unigrams_dict[token][0] += 1
        counter += 1
    print("Length of unigrams_dict: ", len(unigrams_dict))
    total_freq = 0
    for k,v in unigrams_dict.items():
        total_freq += v[0]
    print("Total_freq: ", total_freq)
    for k,v in unigrams_dict.items():
        v[1] = v[0]/total_freq
    unigrams_dict = {k: v for k, v in sorted(unigrams_dict.items(), key=lambda item: item[1][1], reverse=True)}
    return unigrams_dict


# In[ ]:


def write_unigrams(unigrams_dict, output_path):
    f = io.open(output_path, "w", encoding = 'utf-8')
    print("Currently writing unigrams to file ...")
    for key, value in unigrams_dict.items():
        f.write(key + '\t' + str(value[0]) + '\t' + str(value[1]) + '\n')
    f.close()
    print("Unigrams successfully written to file!")


# In[ ]:


#examples
lst = ['hello i am steven',
       'hello who are hello?',
       'wow this this this']
unigrams_dict = get_unigrams_test(lst)
print(unigrams_dict)


# In[ ]:


input_path = 'yelp_train_full.txt'
output_path = 'yelp_train_full_unigrams.txt'

import time
start = time.time()
#unigrams_dict = get_unigrams(input_path)
#write_unigrams(unigrams_dict, output_path)
end = time.time()
print(end-start)

