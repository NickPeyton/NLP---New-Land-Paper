import os
import re

import numpy as np
np.bool = np.bool_
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

import spacy
nlp = spacy.load('en_core_web_sm')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import platform
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

PROCESSED_DATA = 'Data/Processed'
RAW_DATA = 'Data/Raw'

#%% Loading

tdf = pd.read_csv(f'{RAW_DATA}/tithe_apportionments_1834_1852.csv')
tdf.columns = [x.lower() for x in tdf.columns]

#%% Little adjustments

tdf['area_perches'] = tdf['acres'] * 160 + tdf['roods'] * 40 + tdf['perches']
tdf = tdf[tdf['area_perches'] > 0]
tdf.rename(columns={'cultivat_1': 'cultivation'}, inplace=True)
tdf = tdf[['parish', 'landowner', 'occupier', 'cultivation', 'area_perches']]

#%% Grabbing landowner names

def name_grabber(row):
    owner_forename = ''
    owner_surname = ''
    owner_text = row['landowner'].strip()
    owner_text_list = owner_text.split(' ')
    owner_doc = nlp(owner_text)
    owner_ents = owner_doc.ents
    owner_ents = [x for x in owner_ents if x.label_ == 'PERSON']
    if len(owner_ents) != 0:
        if len(owner_ents) == 1:
            name_list = owner_ents[0].text.split(' ')
            owner_forename = name_list[0]
            owner_surname = name_list[-1]
        else:
            for token in owner_doc:
                if token.text == 'for' or token.text == 'under':
                    name = owner_ents[-1].text
                    name_list = name.split(' ')
                    owner_surname = name_list[-1]
                    owner_forename = name_list[0]
    else:
        owner_forename = ''
        owner_surname = ''
    row['owner_forename'] = owner_forename
    row['owner_surname'] = owner_surname

    occupier_text = row['occupier'].strip()
    occupier_text_list = occupier_text.split(' ')
    occupier_doc = nlp(occupier_text)
    occupier_ents = occupier_doc.ents
    occupier_ents = [x for x in occupier_ents if x.label_ == 'PERSON']
    if len(occupier_ents) != 0:
        name_list = occupier_ents[0].text.split(' ')
        occupier_forename = name_list[0]
        occupier_surname = name_list[-1]
    else:
        occupier_forename = ''
        occupier_surname = ''
    row['occupier_forename'] = occupier_forename
    row['occupier_surname'] = occupier_surname

    return row

tdf = tdf.progress_apply(name_grabber, axis=1)


#%%
tdf.to_csv(f'{PROCESSED_DATA}/tithe_landowners.csv', index=False, encoding='utf-8')