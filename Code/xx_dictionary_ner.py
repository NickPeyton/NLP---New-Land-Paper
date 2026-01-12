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

import spacy

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
nlp = spacy.load('en_core_web_trf')
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

early_bearers = re.compile(r'\nEarly bearers:')
references = re.compile(r'\nReferences:')


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
    if page_num > 125:
        break
    test_page = doc[page_num]
    text_blocks = test_page.get_text('blocks')
    if len(text_blocks) == 0:
        continue
    text_blocks.sort(key=lambda x: x[-2])
    text_blocks = [x[4] for x in text_blocks]
    text_blocks = [fix_unicode_numbers(x) for x in text_blocks]
    text_blocks = [fix_ligatures(x) for x in text_blocks]
    text_blocks = [x.replace('’', "'") for x in text_blocks]

    for block in text_blocks:
        print(block)
        print('='*20)
    print('+'*50)



