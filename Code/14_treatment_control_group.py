import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import phonetics as ph
import nltk
import random
from tqdm.auto import tqdm
tqdm.pandas()
import os
import ast
import shutil
import platform
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

#%% Loading
dsdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks.csv', encoding='utf-8')
hsdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks_by_hundred.csv', encoding='utf-8')
psdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks_by_parish.csv', encoding='utf-8')


#%% Getting surnames with similar wealth and size in 1524, 1543
for recip_surname in [
    'recipient_surname',
    'recipient_surname_43',
    'recipient_surname_post_43'
]:
    if recip_surname == 'recipient_surname':
        subsidy = '1524'
        recipient_surname_df = dsdf[dsdf[recip_surname] == 1].copy()
        non_recipient_df = dsdf[dsdf[recip_surname] == 0].copy()
    elif recip_surname == 'recipient_surname_43':
        subsidy = '1524'
        recipient_surname_df = dsdf[dsdf[recip_surname] == 1].copy()
        non_recipient_df = dsdf[(dsdf[recip_surname] == 0) & (dsdf['recipient_surname_post_43'] == 0)].copy()
    elif recip_surname == 'recipient_surname_post_43':
        subsidy = '1543'
        recipient_surname_df = dsdf[dsdf[recip_surname] == 1].copy()
        non_recipient_df = dsdf[(dsdf[recip_surname] == 0) & (dsdf['recipient_surname_43'] == 0)].copy()

    recipient_control_df = pd.DataFrame()

    for i, row in tqdm(recipient_surname_df.iterrows(), total=len(recipient_surname_df)):
        u_id = row['id']
        count = row[f'count_{subsidy}']
        tot_val = row[f'tot_val_{subsidy}']
        avg_val = row[f'avg_val_{subsidy}']
        max_val = row[f'max_val_{subsidy}']
        if np.isnan(count) or np.isnan(tot_val) or count == 0 or tot_val == 0:
            continue
        candidates_df = pd.DataFrame()
        pct = .01
        while len(candidates_df) == 0:
            candidates_df = non_recipient_df[(non_recipient_df[f'count_{subsidy}'] >= (1 - pct) * count) & (
                        non_recipient_df[f'count_{subsidy}'] <= (1 + pct) * count)]
            candidates_df = candidates_df[(candidates_df[f'tot_val_{subsidy}'] >= (1 - pct) * tot_val) & (
                        candidates_df[f'tot_val_{subsidy}'] <= (1 + pct) * tot_val)]
            candidates_df = candidates_df[(candidates_df[f'avg_val_{subsidy}'] >= (1 - pct) * avg_val) & (
                        candidates_df[f'avg_val_{subsidy}'] <= (1 + pct) * avg_val)]
            pct += .01
        new_row = candidates_df.sample(1)
        recipient_control_df = pd.concat([recipient_control_df, new_row])
        non_recipient_df = non_recipient_df.loc[~non_recipient_df['id'].isin(new_row['id'])]

        if recip_surname == 'recipient_surname':
            recipient_surname_df.to_csv(f'{PROCESSED}/surname_treatment_group.csv', encoding='utf-8', index=False)
            recipient_control_df.to_csv(f'{PROCESSED}/surname_control_group.csv', encoding='utf-8', index=False)
            rt_list = recipient_surname_df['id'].tolist()
            rc_list = recipient_control_df['id'].tolist()
        elif recip_surname == 'recipient_surname_43':
            recipient_surname_df.to_csv(f'{PROCESSED}/surname_treatment_group_43.csv', encoding='utf-8', index=False)
            recipient_control_df.to_csv(f'{PROCESSED}/surname_control_group_43.csv', encoding='utf-8', index=False)
            rt_list_1543 = recipient_surname_df['id'].tolist()
            rc_list_1543 = recipient_control_df['id'].tolist()
        elif recip_surname == 'recipient_surname_post_43':
            recipient_surname_df.to_csv(f'{PROCESSED}/surname_treatment_group_post_43.csv', encoding='utf-8', index=False)
            recipient_control_df.to_csv(f'{PROCESSED}/surname_control_group_post_43.csv', encoding='utf-8', index=False)
            rt_list_post_1543 = recipient_surname_df['id'].tolist()
            rc_list_post_1543 = recipient_control_df['id'].tolist()


rando_df = non_recipient_df.loc[~non_recipient_df['id'].isin(recipient_control_df['id'])]
rando_df = rando_df.sample(len(recipient_control_df))
rando_list = rando_df['id'].tolist()

#%% Tagging control group in county, hundred, parish, master, and tithe data

for df_file in [
f'{PROCESSED}/tithe_landowners_final.csv',
f'{PROCESSED}/master_subsidy_data_final.csv',
f'{PROCESSED}/devon_surname_ranks.csv',
f'{PROCESSED}/devon_surname_ranks_by_hundred.csv',
f'{PROCESSED}/devon_surname_ranks_by_parish.csv',
f'{RAW}/freeholders_list_1713_1780_final.csv',
f'{RAW}/bank_returns_1845_1880_final.csv',
f'{RAW}/bankrupts_list_1800_1820_final.csv',
f'{RAW}/bankrupts_list_1820_1843_final.csv',
f'{RAW}/indictable_offenses_1745_1782_final.csv',
f'{RAW}/monumental_brasses_final.csv',
f'{RAW}/victuallers_list_1651_1828_final.csv',
f'{RAW}/workhouse_list_1861_final.csv',
f'{PROCESSED}/ukda_pcc_wills_final.csv',
]:
    df = pd.read_csv(df_file, encoding='utf-8')
    df['recipient_treatment_group'] = df['id'].apply(lambda x: 1 if x in rt_list else 0)
    df['recipient_treatment_group_43'] = df['id'].apply(lambda x: 1 if x in rt_list_1543 else 0)
    df['recipient_treatment_group_post_43'] = df['id'].apply(lambda x: 1 if x in rt_list_post_1543 else 0)
    df['recipient_control_group'] = df['id'].apply(lambda x: 1 if x in rc_list else 0)
    df['recipient_control_group_43'] = df['id'].apply(lambda x: 1 if x in rc_list_1543 else 0)
    df['recipient_control_group_post_43'] = df['id'].apply(lambda x: 1 if x in rc_list_post_1543 else 0)
    df['rando_group'] = df['id'].apply(lambda x: 1 if x in rando_list else 0)
    df.to_csv(df_file, encoding='utf-8', index=False)

