#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
main_path = "data"
os.chdir(main_path)


# # Prepare Final Training Data

# ## Amount Experiments (Together)

# In[ ]:


def write_lst(lst, output_file):
    out_f = open(output_file, 'w')
    print("Writing lines to file...")
    out_f.writelines(lst)
    out_f.close()
    print("Lines written to files")


# In[ ]:


def get_final_train_lst_together(files_lst):
    files = [open(files_lst[i], 'r', encoding='utf-8') for i in range(len(files_lst))]
    final_train_lst = []
    gold = files[0].readlines()
    noise = files[1].readlines()
    syns = files[2].readlines()
    hypos = files[3].readlines()
    hypers = files[4].readlines()
    STE = files[5].readlines()

    final_variation_lst_4x = []
    final_variation_lst_3x = []
    final_variation_lst_2x = []

    variation_lst = [x.strip('\n') for x in STE[:16667]] + [x.strip('\n') for x in noise[16667:33333]] +     [x.strip('\n') for x in hypers[33333:38889]] + [x.strip('\n') for x in syns[38889:44445]] + [x.strip('\n') for x in hypos[44445:50000]]
    for x in variation_lst:
        choices = x.split('\t')
        for element in choices:
            final_variation_lst_4x.append(element.strip('\n') + '\n')    
    counter = 0
    noise_counter = 0
    STE_counter = 0
    syns_counter = 0
    hypos_counter = 0
    hypers_counter = 0
    for x in variation_lst:
        if len(x.split('\t')) >= 2:
            if counter < 16667:
                STE_counter += 1
            elif counter >= 16667 and counter < 33333:
                noise_counter += 1
            elif counter >= 33333 and counter < 38889:
                hypers_counter += 1
            elif counter >= 38889 and counter < 44445:
                syns_counter += 1
            elif counter >= 44445 and counter < 50000:
                hypos_counter += 1
            choices = random.sample(x.split('\t'),2)
            choice = random.choice(choices)
            final_variation_lst_2x.append(choice.strip('\n') + '\n')
            for element in choices:
                final_variation_lst_3x.append(element.strip('\n') + '\n')
        counter += 1

    final_variation_lst_15x = final_variation_lst_2x[:8333] + final_variation_lst_2x[16667:25000] + final_variation_lst_2x[33334:36112] +     final_variation_lst_2x[38890:41668] + final_variation_lst_2x[44446:47224]
    
    final_train_lst_2x = gold + final_variation_lst_2x
    final_train_lst_3x = gold + final_variation_lst_3x
    final_train_lst_4x = gold + final_variation_lst_4x
    final_train_lst_15x = gold + final_variation_lst_15x

    random.shuffle(final_train_lst_2x)
    random.shuffle(final_train_lst_3x)
    random.shuffle(final_train_lst_4x)
    random.shuffle(final_train_lst_15x)

    print(len(final_train_lst_2x))
    print(len(final_train_lst_3x))
    print(len(final_train_lst_4x))
    print(len(final_train_lst_15x))

    return final_train_lst_2x, final_train_lst_3x, final_train_lst_4x, final_train_lst_15x


# In[ ]:


random.seed(54321)
files_lst = ['yelp_train.txt','yelp_train_noise.txt','WordNet/yelp_train_synonyms.txt',             'yelp_train_hyponyms.txt','yelp_train_hypernyms.txt',             'SMERTI/yelp_train_SMERTI_outputs.txt']
final_train_lst_2x, final_train_lst_3x, final_train_lst_4x, final_train_lst_15x = get_final_train_lst_together(files_lst)
          
output_file_2x = 'yelp_train_2x.txt'
output_file_3x = 'yelp_train_3x.txt'
output_file_4x = 'yelp_train_4x.txt'
output_file_15x = 'yelp_train_1.5x.txt'

#write_lst(final_train_lst_2x, output_file_2x)
#write_lst(final_train_lst_3x, output_file_3x)
#write_lst(final_train_lst_4x, output_file_4x)
#write_lst(final_train_lst_15x, output_file_15x)


# ## Augmentation Method Experiments (Single Variations - Excluding Random Trio)

# In[ ]:


def get_final_train_lst(file1, file2, limit=1000000):
    f1 = open(file1, 'r', encoding='utf-8')
    f2 = open(file2, 'r', encoding='utf-8')
    gold = f1.readlines()
    variation = f2.readlines()
    blank_count = sum([1 for x in variation if x.strip() == '<blank>'])
    final_train_lst = []
    variation_lst = []
    counter = 0
    for line in variation:
        if counter < limit:
            if line.strip() != '<blank>' and len(line.split('\t')) >= 1:
                choices = line.split('\t')
                chosen = random.sample(choices, 1)
                for item in chosen:
                    variation_lst.append(item.strip('\n')+'\n')
                counter += 1
    final_train_lst = gold + variation_lst
    random.shuffle(final_train_lst)
    print(len(final_train_lst))
    return final_train_lst

def write_lst(lst, output_file):
    out_f = open(output_file, 'w')
    print("Writing lines to file...")
    out_f.writelines(lst)
    out_f.close()
    print("Lines written to files")


# In[ ]:


#Example execution code for SMERTI:
random.seed(54321)
file1 = 'yelp_train.txt'
file2 = 'SMERTI/yelp_train_SMERTI_outputs.txt'
output_file = 'yelp_train_SMERTI_final.txt'
final_train_lst = get_final_train_lst(file1, file2)
#write_lst(final_train_lst, output_file)


# ## Random Trio

# In[ ]:


def get_final_train_lst_random(files_lst):
    files = [open(files_lst[i], 'r', encoding='utf-8') for i in range(len(files_lst))]
    final_train_lst = []
    final_variation_lst = []
    gold = files[0].readlines()
    insertion = files[1].readlines()
    swap = files[2].readlines()
    deletion = files[3].readlines()
    counter = 0
    for line1, line2, line3 in zip(insertion, swap, deletion):
        counter += 1
        try:
            chosen_line = random.choice([x for x in [line1.strip('\n'),line2.strip('\n'),line3.strip('\n')] if x.split('\t')[0] != '<blank>' or x.split('\t')[1] != '<blank>'])
        except:
            continue
        if chosen_line.split('\t')[0] != '<blank>' and chosen_line.split('\t')[1] != '<blank>':
            chosen = random.choice(chosen_line.split('\t'))
        elif chosen_line.split('\t')[0] != '<blank>':
            chosen = chosen_line.split('\t')[0]
        else:
            chosen = chosen_line.split('\t')[1]
        final_variation_lst.append(chosen.strip('\n')+'\n')
    final_train_lst = gold + final_variation_lst
    random.shuffle(final_train_lst)
    print(len(final_train_lst))
    return final_train_lst


def write_lst(lst, output_file):
    out_f = open(output_file, 'w')
    print("Writing lines to file...")
    out_f.writelines(lst)
    out_f.close()
    print("Lines written to files")


# In[ ]:


random.seed(54321)
files_lst = ['yelp_train.txt','yelp_train_random_insert.txt',             'yelp_train_random_swap.txt','yelp_train_random_delete.txt']
output_file = 'yelp_train_random_final.txt'

final_train_lst = get_final_train_lst_random(files_lst)
#write_lst(final_train_lst, output_file)

