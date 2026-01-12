import os
import re
import ast
import json
import nltk
import shutil
import pymupdf
import platform
import numpy as np
import pandas as pd
import phonetics as ph
from tqdm.auto import tqdm
from pypdf import PdfReader
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'
#%%

doc = pymupdf.open(f'{RAW}/family_name_dict/family_name_dict.pdf')
#%%

# Pages to Skip
vol_title = re.compile(r'^The Oxford Dictionary')
copy_page = re.compile(r'^[0-9]\n')
ed_page = re.compile(r'^Editors and contributors\n')
toc_page = re.compile(r'^Contents\n')

letter_header = re.compile(r'^[A-Z]\n$')

missed_space = re.compile(r'([A-Za-z]+)([A-Z]+)')
missed_space2 = re.compile(r'([0-9]+)([A-Za-z]+)')
missed_space3 = re.compile(r'(:)([A-Za-z]+)')


start_name_block = re.compile(r'^\n?([A-Z][\'-. ]?[A-Za-z\'-. ]+)\n\.\.\.')
mid_name_block = re.compile(r'\n?([A-Z][\'-. ]?[A-Za-z\'-. ]+)\n\.\.\.')
variants = re.compile(r'Variants:\s?((?:[A-Za-z\']+,?\s?)+)')
#%%
def fix_unicode_numbers(text: str) -> str:
    return ''.join(
        chr(ord(c) - 0xF700) if '\uf730' <= c <= '\uf739' else c
        for c in text
    )


def fix_ligatures(text):
    ligature_map = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st"
    }
    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)
    return text

#%%
text_list = []
for page_num in tqdm(range(122, 3138)):
    test_page = doc[page_num]
    text_blocks = test_page.get_text('blocks')
    if len(text_blocks) == 0:
        continue
    text_blocks.sort(key=lambda x: x[-2])
    text_blocks = [x[4] for x in text_blocks]
    text_blocks = [fix_unicode_numbers(x) for x in text_blocks]
    text_blocks = [fix_ligatures(x) for x in text_blocks]
    text_blocks = [x.replace('’', "'") for x in text_blocks]

    if re.search(vol_title, text_blocks[0]) or re.search(copy_page, text_blocks[0]) or re.search(ed_page, text_blocks[0]) or re.search(toc_page, text_blocks[0]):
        continue
    if re.search(letter_header, text_blocks[0]):
        text_blocks = text_blocks[1:-1]
    else:
        text_blocks = text_blocks[:-2]
    text_list.extend(text_blocks)
huh = ['Delve', 'Lukehurst', 'Luckhurst', 'Lovick', 'Delves', 'Look', 'Leffek', 'Lucus', 'Delph', 'Livick', 'Lowick', 'Lukas', 'Deluca', 'Mc Lucas', 'Lucas', 'Loake', 'Dolph', 'Loukes', 'Levick', 'Locke', 'Locks', 'Lock', 'Luck', 'Delf', 'Livock']

new_list = []
for text in text_list:

    if new_list and not re.search(start_name_block, text):
        new_list[-1] += '\n' + text  # Append to the last element
    else:
        new_list.append(text)

text_list = new_list

#%%
num_shit = 0
new_list = []
for text in text_list:
    if len(re.findall(mid_name_block, text)) > 1:
        num_shit += len(re.findall(mid_name_block, text))
        text = re.sub(mid_name_block, r'\n\n\n\1', text)
        new_text_blocks = text.split('\n\n\n')
        for new_block in new_text_blocks[1:]:
            print(new_block)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('0000000000000000000000000000000000000000000000000000000000000000')
        new_list += new_text_blocks[1:]
        continue
    new_list += [text]
print(num_shit)


#%%

permitted_origins = ['English:',
                     'French:',
                     'Welsh:',
                     'Cornish:',
                     'Norman:']

see_other = [x + ' see' for x in permitted_origins]
variant_of = [re.compile(rf'{x}[a-z\s,()]+variant\sof\s[A-Z][\'a-z]+[\'\-A-Za-z\s]\.') for x in permitted_origins]

occupational_surname_pattern = re.compile(
    r"[Oo]ccupational name [A-Za-z(),\s]+('[A-Za-z\s]+')"
)
occupational_surname = re.compile(r'[Oo]ccupational name')
locative_surname_pattern = re.compile(
    r'[Ll]ocative name from\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
    r'(?: in\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*))?'
    r' \((\w+)\)'
)
devon_pattern = re.compile(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*) \(Devon\)')
locative_surname = re.compile(r'[Ll]ocative name')
topo_surname = re.compile(r'(toponym|topograph)')
nickname = re.compile(r'[Nn]ickname from')
relationship_name = re.compile(r'[Rr]elationship name')

new_list = []
for text in text_list:
    if any(x in text for x in see_other):
        continue
    if any(re.search(x, text) for x in variant_of):
        continue
    new_list.append(text)

text_list = new_list
name_info_df = pd.DataFrame()
basic_name_lists = []
for text in tqdm(text_list):
    text = text.replace('\n\n', '\n')
    text = re.sub(missed_space, r'\1 \2', text)
    text = re.sub(missed_space2, r'\1 \2', text)
    text = re.sub(missed_space3, r'\1 \2', text)
    if not any(x in text for x in permitted_origins):
        continue


    name = re.search(start_name_block, text).group(1)

    if re.search(variants, text):
        variants_text = re.search(variants, text).group(1)
        variants_list = variants_text.split(',')
        variants_list = [x.strip() for x in variants_list]
    else:
        variants_list = []
    new_name_list = [name] + variants_list
    basic_name_lists.append(new_name_list)

    origins = re.split(r'\n\d ', text)
    origins = origins[1:]
    occup = 0
    occupation = ''
    loc = 0
    topo = 0
    nick = 0
    rel = 0
    location = ''
    location_in = ''
    county = ''
    for section_text in origins:
        section_text = re.split(r'\nEarly Bearers:', section_text)[0]
        section_text = re.split(r'\nReferences:', section_text)[0]
        if re.search(occupational_surname, section_text):
            occup = 1
            if re.search(occupational_surname_pattern, section_text):
                occupation = re.search(occupational_surname, section_text).group(1)
        if re.search(locative_surname, section_text):
            loc = 1

            if re.search(locative_surname_pattern, section_text):
                location = re.search(locative_surname_pattern, section_text).group(1)
                if re.search(locative_surname_pattern, section_text).group(2):
                    location_in = re.search(locative_surname_pattern, section_text).group(2)
                county = re.search(locative_surname_pattern, section_text).group(3)
        if re.search(topo_surname, section_text):
            topo = 1
        if re.search(nickname, section_text):
            nick = 1
        if re.search(relationship_name, section_text):
            rel = 1
    for name in new_name_list:
        new_row_dict = {'surname': name,
                        'occup': occup,
                        'occupation': occupation,
                        'loc': loc,
                        'location': location,
                        'location_in': location_in,
                        'county': county,
                        'topo': topo,
                        'nick': nick,
                        'rel': rel}
        for origin in permitted_origins:
            if origin in text:
                new_row_dict[origin] = 1
            else:
                new_row_dict[origin] = 0
        new_row = pd.DataFrame(new_row_dict, index=[0])
        name_info_df = pd.concat([name_info_df, new_row], ignore_index=True)


name_info_df = name_info_df.sort_values(by='surname')
name_info_df.to_csv(f'{PROCESSED}/surname_info.csv', index=False)

#%%

basic_name_sets = [set(x) for x in basic_name_lists]
non_combined_name_lists = [x.copy() for x in basic_name_lists]
# Combine all sets with any elements in common
fixedPoint = False
iteration = 0
while not fixedPoint:
    fixedPoint = True
    print('Iteration: ' + str(iteration))
    iteration += 1
    for i, name_set in enumerate(basic_name_sets):
        for name_set2 in basic_name_sets[i+1:]:
            if name_set & name_set2:
                basic_name_sets.remove(name_set)
                basic_name_sets.remove(name_set2)
                basic_name_sets.append(name_set | name_set2)
                fixedPoint = False
                break

combined_name_lists = [list(x) for x in basic_name_sets]
combined_name_lists.sort()
non_combined_name_lists.sort()
#%%
max = 0
for name_list in combined_name_lists:
    for name in name_list:
        if len(name) > max:
            max = len(name)
            longest = name
    if len(name_list) > 20:
        print(name_list)
print(f'Max Name Length: {max}, Name: {longest}')
#%%
with open(f'{PROCESSED}/combined_surnames.json', 'w', encoding='utf-8') as f:
    json.dump(combined_name_lists, f, indent=4)

with open(f'{PROCESSED}/non_combined_surnames.json', 'w', encoding='utf-8') as f:
    json.dump(non_combined_name_lists, f, indent=4)

# #%%
# # Substitutions in regex form so I can specify end-of-string etc.
# substitutions = [
#     [r'([bcdfghjklmnpqrstvwxyz])e$',  # Trailing E
#      r'([bcdfghjklmnpqrstvwxyz])'],

#     [r'([bcdfghjklmnpqrstvwxyz])eigh$',   # "ay"/"ee" sound at end of name
#      r'([bcdfghjklmnpqrstvwxyz])egh$',
#      r'([bcdfghjklmnpqrstvwxyz])ey$',
#      r'([bcdfghjklmnpqrstvwxyz])y$'],
#     r'([bcdfghjklmnpqrstvwxyz])ay$',

#     [r'^gui',   # "guy" sound at beginning of name
#      r'^guy'],

#     [r'y',  # Interchangeable Y and I
#      r'i'],

#     [r'^s', # Leading S or Z
#      r'^z'],

#     [r'([bcdfgklmnprstvwz]{2})', # Double consonants
#      r'([bcdfgklmnprstvwz])'],

#     [r'bury$',  # "Bury" sound at end of name
#      r'berry$',
#      r'bery$',
#      r'borough$',
#      r'boro$',
#      r'brow$',],

#     [r'w([aeiouy])',  # W followed by a vowel
#      r'wh[aeiouy]'],

#     [r'ool',    # "ool" sound
#      r'oll',],

#     [r'le$',    # "ul" sound at end of name
#      r'ell$',
#      r'el$'],

#     [r'tch',    # "ch" sound
#      r'ch'],

#     [r'dg',     # "dge" sound
#      r'g'],



# ]