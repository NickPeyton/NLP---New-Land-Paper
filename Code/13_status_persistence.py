#%%

import os
import shutil
import re
import ast
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import phonetics as ph
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
# %% Load Data
mdf = pd.read_csv(f'{PROCESSED}/master_subsidy_data_final.csv', encoding='utf-8')
if 'gemini_surname' in mdf.columns:
    mdf.rename(columns={'gemini_surname': 'surname',
                        'gemini_parish': 'parish',
                        'gemini_value': 'value'}, inplace=True)

mdf = mdf.loc[~mdf['surname'].isna()]
mdf = mdf[mdf['surname'].str.istitle()]
phdf = pd.read_csv(f'{RAW}/parishes_hundreds.csv', encoding='utf-8')
tdf = pd.read_csv(f'{PROCESSED}/tithe_landowners_final.csv')

cdf = pd.read_csv(f'{RAW}/calendar_of_particulars/calendar_of_particulars.csv', encoding='utf-8')

with open(f'{PROCESSED}/soundex_metaphone_id_dict.json', 'r') as j:
    s_m_id_dict = json.loads(j.read())

with open(f'{PROCESSED}/parish_correction.json', 'r') as j:
    parish_name_correction = json.loads(j.read())
# %% Recipient Surnames

pddf = cdf[cdf['in_devon'].isin(['0.5', '1'])]
pddf = pddf.loc[~pddf['surname'].isna()]
pddf = pddf.groupby('grant_num').agg({'surname': 'first', 'surname_2': 'first', 'year': 'first'}).reset_index()
pddf['surname'] = pddf['surname'].str.replace('St Hill', 'Sainthill')
pddf_2 = pddf[pddf['surname_2'] != '']
pddf_2 = pddf_2.loc[~pddf_2['surname_2'].isna()]
pddf_2 = pddf_2[['grant_num', 'surname_2', 'year']]
pddf_2.columns = ['grant_num', 'surname', 'year']
pddf = pddf[['grant_num', 'surname', 'year']]
pddf = pd.concat([pddf, pddf_2])
pddf.reset_index(drop=True, inplace=True)
pddf['metaphone'] = pddf['surname'].progress_apply(lambda x: ph.metaphone(x.title()))
pddf['soundex'] = pddf['surname'].progress_apply(lambda x: ph.soundex(x.title()))
pddf['s_m'] = pddf['soundex'] + ' ' + pddf['metaphone']
if 'id' not in pddf.columns:
    pddf['id'] = np.nan
pddf.loc[pddf['id'].isna(), 'id'] = pddf['s_m'].map(s_m_id_dict)
pddf = pddf[~pddf['surname'].isin(['Bellingham', 'Strangman'])]
pddf_43 = pddf[pddf['year'] <= 1543].copy()
pddf_post_43 = pddf[pddf['year'] > 1543].copy()

recipient_surnames = list(set(pddf['id'].unique()))
recipient_surnames_43 = list(set(pddf_43['id'].unique()))
recipient_surnames_post_43 = list(set(pddf_post_43['id'].unique()))

mdf.loc[mdf['id'].isin(recipient_surnames), 'recipient_surname'] = 1
mdf.loc[mdf['id'].isin(recipient_surnames_43), 'recipient_surname_43'] = 1
mdf.loc[mdf['id'].isin(recipient_surnames_post_43), 'recipient_surname_post_43'] = 1

tdf.loc[tdf['id'].isin(recipient_surnames), 'recipient_surname'] = 1
tdf.loc[tdf['id'].isin(recipient_surnames_43), 'recipient_surname_43'] = 1
tdf.loc[tdf['id'].isin(recipient_surnames_post_43), 'recipient_surname_post_43'] = 1

# %% Correcting hundreds

hundo_dict = dict(zip(phdf['parish'], phdf['hundred']))
hundo_dict = hundo_dict | {'Burlescombe': 'Bampton',
                           'Brixham': 'Haytor',
                           'Uplowman': 'Tiverton',
                           'Hawkchurch': 'Axminster',
                           'Somerset': 'Somerset',
                           'Middlesex': 'Middlesex',
                           'Sussex': 'Sussex',
                           'Dorset': 'Dorset', }
mdf['parish'] = mdf['parish'].str.title()
mdf['parish'] = mdf['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)
mdf = mdf.loc[~mdf['parish'].isin(['Northcott Hamlet'])]

tdf['parish'] = tdf['parish'].str.title()
tdf['parish'] = tdf['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)

mdf['hundred'] = mdf['parish'].apply(lambda x: hundo_dict[x] if x in hundo_dict else '')
tdf['hundred'] = tdf['parish'].apply(lambda x: hundo_dict[x] if x in hundo_dict else '')

mdf_parish_list = []

mdf.to_csv(f'{PROCESSED}/master_subsidy_data.csv', encoding='utf-8', index=False)
tdf.to_csv(f'{PROCESSED}/tithe_landowners.csv', encoding='utf-8', index=False)

# %%
# Get a dictionary of ids to the list of surnames they go to
id_list = mdf.id.unique().tolist()
id_list.sort()
id_dict = {}
for id in id_list:
    id_dict[id] = mdf[mdf['id'] == id]['surname'].unique().tolist()

# %% Surname ranks in Devon overall

devon_surname_df = pd.DataFrame({'id': mdf['id'].unique()})
mdf['count'] = 1
mdf['max_val'] = mdf['value']

for subsidy in [
    1524,
    1543,
    1581,
    1674
]:
    gdf = mdf[mdf['year'] == subsidy].groupby('id').agg({'count': 'sum',
                                                              'value': 'sum',
                                                              'max_val': 'max', }).reset_index()
    gdf.rename(columns={'value': 'tot_val'}, inplace=True)
    gdf = gdf.sort_values('tot_val', ascending=False)
    gdf['tot_val_rank'] = gdf['tot_val'].rank(ascending=False)
    gdf['tot_val_pctile'] = gdf['tot_val'].rank(ascending=True) / len(gdf)
    gdf['count_rank'] = gdf['count'].rank(ascending=False)
    gdf['count_pctile'] = gdf['count'].rank(ascending=True) / len(gdf)
    gdf['avg_val'] = gdf['tot_val'] / gdf['count']
    gdf['avg_val_rank'] = gdf['avg_val'].rank(ascending=False)
    gdf['avg_val_pctile'] = gdf['avg_val'].rank(ascending=True) / len(gdf)
    gdf['max_val_rank'] = gdf['max_val'].rank(ascending=False)
    gdf['max_val_pctile'] = gdf['max_val'].rank(ascending=True) / len(gdf)

    devon_surname_df['count_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['count'])))
    devon_surname_df['tot_val_rank_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['tot_val_rank'])))
    devon_surname_df['tot_val_pctile_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['tot_val_pctile'])))
    devon_surname_df['count_rank_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['count_rank'])))
    devon_surname_df['count_pctile_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['count_pctile'])))
    devon_surname_df['avg_val_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['avg_val'])))
    devon_surname_df['avg_val_rank_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['avg_val_rank'])))
    devon_surname_df['avg_val_pctile_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['avg_val_pctile'])))
    devon_surname_df['max_val_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['max_val'])))
    devon_surname_df['max_val_rank_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['max_val_rank'])))
    devon_surname_df['max_val_pctile_' + str(subsidy)] = devon_surname_df['id'].map(
        dict(zip(gdf['id'], gdf['max_val_pctile'])))
    devon_surname_df['tot_val_' + str(subsidy)] = devon_surname_df['id'].map(dict(zip(gdf['id'], gdf['tot_val'])))

tdf['count'] = 1
tdf['max_val'] = tdf['area_perches']
gtdf = tdf.groupby('id').agg({'count': 'sum',
                                'area_perches': 'sum',
                                'max_val': 'max'}).reset_index()
gtdf.rename(columns={'area_perches': 'tot_val'}, inplace=True)
gtdf = gtdf.sort_values('tot_val', ascending=False)
gtdf['tot_val_rank'] = gtdf['tot_val'].rank(ascending=False)
gtdf['tot_val_pctile'] = gtdf['tot_val'].rank(ascending=True) / len(gtdf)
gtdf['count_rank'] = gtdf['count'].rank(ascending=False)
gtdf['count_pctile'] = gtdf['count'].rank(ascending=True) / len(gtdf)
gtdf['avg_val'] = gtdf['tot_val'] / gtdf['count']
gtdf['avg_val_rank'] = gtdf['avg_val'].rank(ascending=False)
gtdf['avg_val_pctile'] = gtdf['avg_val'].rank(ascending=True) / len(gtdf)
gtdf['max_val_rank'] = gtdf['max_val'].rank(ascending=False)
gtdf['max_val_pctile'] = gtdf['max_val'].rank(ascending=True) / len(gtdf)

devon_surname_df['count_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['count'])))
devon_surname_df['tot_val_rank_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['tot_val_rank'])))
devon_surname_df['tot_val_pctile_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['tot_val_pctile'])))
devon_surname_df['count_rank_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['count_rank'])))
devon_surname_df['count_pctile_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['count_pctile'])))
devon_surname_df['avg_val_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['avg_val'])))
devon_surname_df['avg_val_rank_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['avg_val_rank'])))
devon_surname_df['avg_val_pctile_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['avg_val_pctile'])))
devon_surname_df['max_val_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['max_val'])))
devon_surname_df['max_val_rank_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['max_val_rank'])))
devon_surname_df['max_val_pctile_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['max_val_pctile'])))
devon_surname_df['tot_val_1840'] = devon_surname_df['id'].map(dict(zip(gtdf['id'], gtdf['tot_val'])))

year_list = [1524, 1543, 1581, 1674, 1840]
for year1 in year_list:
    for year2 in year_list[year_list.index(year1) + 1:]:
        devon_surname_df[f'tot_val_rank_corr_{year1}_{year2}'] = devon_surname_df['tot_val_rank_' + str(year1)].corr(
            devon_surname_df['tot_val_rank_' + str(year2)])
        devon_surname_df[f'tot_val_pctile_corr{year1}_{year2}'] = devon_surname_df['tot_val_pctile_' + str(year1)].corr(
            devon_surname_df['tot_val_pctile_' + str(year2)])
        devon_surname_df[f'avg_val_rank_corr_{year1}_{year2}'] = devon_surname_df['avg_val_rank_' + str(year1)].corr(
            devon_surname_df['avg_val_rank_' + str(year2)])
        devon_surname_df[f'avg_val_pctile_corr_{year1}_{year2}'] = devon_surname_df['avg_val_pctile_' + str(year1)].corr(
            devon_surname_df['avg_val_pctile_' + str(year2)])
        devon_surname_df[f'max_val_rank_corr_{year1}_{year2}'] = devon_surname_df['max_val_rank_' + str(year1)].corr(
            devon_surname_df['max_val_rank_' + str(year2)])
        devon_surname_df[f'max_val_pctile_corr_{year1}_{year2}'] = devon_surname_df['max_val_pctile_' + str(year1)].corr(
            devon_surname_df['max_val_pctile_' + str(year2)])

        devon_surname_df[f'tot_val_rank_{year1}_{year2}'] = devon_surname_df['tot_val_rank_' + str(year1)] - devon_surname_df[
            'tot_val_rank_' + str(year2)]
        devon_surname_df[f'tot_val_pctile_{year1}_{year2}'] = devon_surname_df['tot_val_pctile_' + str(year1)] - \
                                                            devon_surname_df['tot_val_pctile_' + str(year2)]
        devon_surname_df[f'avg_val_rank_{year1}_{year2}'] = devon_surname_df['avg_val_rank_' + str(year1)] - \
                                                            devon_surname_df['avg_val_rank_' + str(year2)]
        devon_surname_df[f'avg_val_pctile_{year1}_{year2}'] = devon_surname_df['avg_val_pctile_' + str(year1)] - \
                                                            devon_surname_df['avg_val_pctile_' + str(year2)]
        devon_surname_df[f'max_val_rank_{year1}_{year2}'] = devon_surname_df['max_val_rank_' + str(year1)] - \
                                                            devon_surname_df['max_val_rank_' + str(year2)]
        devon_surname_df[f'max_val_pctile_{year1}_{year2}'] = devon_surname_df['max_val_pctile_' + str(year1)] - \
                                                            devon_surname_df['max_val_pctile_' + str(year2)]

devon_surname_df['recipient_surname'] = 0
devon_surname_df.loc[devon_surname_df['id'].isin(recipient_surnames), 'recipient_surname'] = 1
devon_surname_df['recipient_surname_43'] = 0
devon_surname_df.loc[devon_surname_df['id'].isin(recipient_surnames_43), 'recipient_surname_43'] = 1
devon_surname_df['recipient_surname_post_43'] = 0
devon_surname_df.loc[devon_surname_df['id'].isin(recipient_surnames_post_43), 'recipient_surname_post_43'] = 1
devon_surname_df.to_csv(f'{PROCESSED}/devon_surname_ranks.csv', encoding='utf-8', index=False)

# %% Surname ranks in Devon by hundred
hsdf = mdf.groupby(['id', 'hundred']).agg({'count': 'sum', })
hsdf.reset_index(inplace=True)
hsdf.drop(columns=['count'], inplace=True)

for subsidy in [
    1524,
    1543,
    1581,
    1674
]:
    gdf = mdf[mdf['year'] == subsidy].groupby(['id', 'hundred']).agg({'count': 'count',
                                                                           'value': 'sum',
                                                                           'max_val': 'max', }).reset_index()
    gdf.rename(columns={'value': 'tot_val'}, inplace=True)

    for hundred in gdf.hundred.unique():
        hdf = gdf[gdf['hundred'] == hundred]
        hdf = hdf.sort_values('tot_val', ascending=False)
        hdf['tot_val_rank'] = hdf['tot_val'].rank(ascending=False)
        hdf['tot_val_pctile'] = hdf['tot_val'].rank(ascending=True) / len(hdf)
        hdf['count_rank'] = hdf['count'].rank(ascending=False)
        hdf['count_pctile'] = hdf['count'].rank(ascending=True) / len(hdf)
        hdf['avg_val'] = hdf['tot_val'] / hdf['count']
        hdf['avg_val_rank'] = hdf['avg_val'].rank(ascending=False)
        hdf['avg_val_pctile'] = hdf['avg_val'].rank(ascending=True) / len(hdf)
        hdf['max_val_rank'] = hdf['max_val'].rank(ascending=False)
        hdf['max_val_pctile'] = hdf['max_val'].rank(ascending=True) / len(hdf)
        hsdf.loc[hsdf['hundred'] == hundred, f'count_{subsidy}'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['count'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_{subsidy}'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_rank_{subsidy}'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_rank'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_pctile_{subsidy}'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_pctile'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'count_rank_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['count_rank'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'count_pctile_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['count_pctile'])))

        hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_rank_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val_rank'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_pctile_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val_pctile'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'max_val_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'max_val_rank_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val_rank'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'max_val_pctile_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val_pctile'])))
        hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_{subsidy}'] = hsdf['id'].map(
            dict(zip(hdf['id'], hdf['tot_val'])))


gtdf = tdf.groupby(['id', 'hundred']).agg({'count': 'sum',
                                             'area_perches': 'sum',
                                             'max_val': 'max'}).reset_index()
gtdf.rename(columns={'area_perches': 'tot_val'}, inplace=True)
gtdf = gtdf.sort_values('tot_val', ascending=False)
gtdf['tot_val_rank'] = gtdf['tot_val'].rank(ascending=False)
gtdf['tot_val_pctile'] = gtdf['tot_val'].rank(ascending=True) / len(gtdf)
gtdf['count_rank'] = gtdf['count'].rank(ascending=False)
gtdf['count_pctile'] = gtdf['count'].rank(ascending=True) / len(gtdf)
gtdf['avg_val'] = gtdf['tot_val'] / gtdf['count']
gtdf['avg_val_rank'] = gtdf['avg_val'].rank(ascending=False)
gtdf['avg_val_pctile'] = gtdf['avg_val'].rank(ascending=True) / len(gtdf)
gtdf['max_val_rank'] = gtdf['max_val'].rank(ascending=False)
gtdf['max_val_pctile'] = gtdf['max_val'].rank(ascending=True) / len(gtdf)

for hundred in gtdf.hundred.unique():
    hdf = gtdf[gtdf['hundred'] == hundred]
    hdf = hdf.sort_values('tot_val', ascending=False)
    hdf['tot_val_rank'] = hdf['tot_val'].rank(ascending=False)
    hdf['tot_val_pctile'] = hdf['tot_val'].rank(ascending=True) / len(hdf)
    hdf['count_rank'] = hdf['count'].rank(ascending=False)
    hdf['count_pctile'] = hdf['count'].rank(ascending=True) / len(hdf)
    hdf['avg_val'] = hdf['tot_val'] / hdf['count']
    hdf['avg_val_rank'] = hdf['avg_val'].rank(ascending=False)
    hdf['avg_val_pctile'] = hdf['avg_val'].rank(ascending=True) / len(hdf)
    hdf['max_val_rank'] = hdf['max_val'].rank(ascending=False)
    hdf['max_val_pctile'] = hdf['max_val'].rank(ascending=True) / len(hdf)

    hsdf.loc[hsdf['hundred'] == hundred, f'count_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_rank'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_rank_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_rank'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_pctile_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_pctile'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'count_rank_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['count_rank'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'count_pctile_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['count_pctile'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['avg_val'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_rank_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['avg_val_rank'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_pctile_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['avg_val_pctile'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['tot_val'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'max_val_1840'] = hsdf['id'].map(dict(zip(hdf['id'], hdf['max_val'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'max_val_rank_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['max_val_rank'])))
    hsdf.loc[hsdf['hundred'] == hundred, f'max_val_pctile_1840'] = hsdf['id'].map(
        dict(zip(hdf['id'], hdf['max_val_pctile'])))

hsdf.sort_values(['hundred', 'tot_val_1524'], inplace=True)
hsdf['recipient_surname'] = 0
hsdf.loc[hsdf['id'].isin(recipient_surnames), 'recipient_surname'] = 1
hsdf['recipient_surname_43'] = 0
hsdf.loc[hsdf['id'].isin(recipient_surnames_43), 'recipient_surname_43'] = 1
hsdf['recipient_surname_post_43'] = 0
hsdf.loc[hsdf['id'].isin(recipient_surnames_post_43), 'recipient_surname_post_43'] = 1
hsdf.to_csv(f'{PROCESSED}/devon_surname_ranks_by_hundred.csv', encoding='utf-8', index=False)

# %% Same shit, by parish

psdf = mdf.groupby(['id', 'parish']).agg({'count': 'sum'})
psdf.reset_index(inplace=True)
psdf.drop(columns=['count'], inplace=True)

for subsidy in [
    1524,
    1543,
    1581,
    1674
]:
    gdf = mdf[mdf['year'] == subsidy].groupby(['id', 'parish']).agg({'count': 'count',
                                                                          'value': 'sum',
                                                                          'max_val': 'max', }).reset_index()
    gdf.rename(columns={'value': 'tot_val'}, inplace=True)

    for parish in gdf.parish.unique():
        hdf = gdf[gdf['parish'] == parish]
        hdf = hdf.sort_values('tot_val', ascending=False)
        hdf['tot_val_rank'] = hdf['tot_val'].rank(ascending=False)
        hdf['tot_val_pctile'] = hdf['tot_val'].rank(ascending=True) / len(hdf['tot_val'])
        hdf['count_rank'] = hdf['count'].rank(ascending=False)
        hdf['count_pctile'] = hdf['count'].rank(ascending=True) / len(hdf['count'])
        hdf['avg_val'] = hdf['tot_val'] / hdf['count']
        hdf['avg_val_rank'] = hdf['avg_val'].rank(ascending=False)
        hdf['avg_val_pctile'] = hdf['avg_val'].rank(ascending=True) / len(hdf['avg_val'])
        hdf['max_val_rank'] = hdf['max_val'].rank(ascending=False)
        hdf['max_val_pctile'] = hdf['max_val'].rank(ascending=True) / len(hdf['max_val'])
        psdf.loc[psdf['parish'] == parish, f'count_{subsidy}'] = psdf['id'].map(dict(zip(hdf['id'], hdf['count'])))
        psdf.loc[psdf['parish'] == parish, f'tot_val_{subsidy}'] = psdf['id'].map(dict(zip(hdf['id'], hdf['tot_val'])))
        psdf.loc[psdf['parish'] == parish, f'tot_val_rank_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['tot_val_rank'])))
        psdf.loc[psdf['parish'] == parish, f'tot_val_pctile_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['tot_val_pctile'])))
        psdf.loc[psdf['parish'] == parish, f'count_rank_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['count_rank'])))
        psdf.loc[psdf['parish'] == parish, f'count_pctile_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['count_pctile'])))
        psdf.loc[psdf['parish'] == parish, f'avg_val_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val'])))
        psdf.loc[psdf['parish'] == parish, f'avg_val_rank_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val_rank'])))
        psdf.loc[psdf['parish'] == parish, f'avg_val_pctile_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['avg_val_pctile'])))
        psdf.loc[psdf['parish'] == parish, f'max_val_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val'])))
        psdf.loc[psdf['parish'] == parish, f'max_val_rank_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val_rank'])))
        psdf.loc[psdf['parish'] == parish, f'max_val_pctile_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['max_val_pctile'])))
        psdf.loc[psdf['parish'] == parish, f'tot_val_{subsidy}'] = psdf['id'].map(
            dict(zip(hdf['id'], hdf['tot_val'])))

tdf['count'] = 1
gtdf = tdf.groupby(['id', 'parish']).agg({'count': 'sum',
                                            'area_perches': 'sum',
                                            'max_val': 'max'}).reset_index()
gtdf.rename(columns={'area_perches': 'tot_val'}, inplace=True)
gtdf = gtdf.sort_values('tot_val', ascending=False)
gtdf['tot_val_rank'] = gtdf['tot_val'].rank(ascending=False)
gtdf['tot_val_pctile'] = gtdf['tot_val'].rank(ascending=True) / len(gtdf)
gtdf['count_rank'] = gtdf['count'].rank(ascending=False)
gtdf['count_pctile'] = gtdf['count'].rank(ascending=True) / len(gtdf)
gtdf['avg_val'] = gtdf['tot_val'] / gtdf['count']
gtdf['avg_val_rank'] = gtdf['avg_val'].rank(ascending=False)
gtdf['avg_val_pctile'] = gtdf['avg_val'].rank(ascending=True) / len(gtdf)
gtdf['max_val_rank'] = gtdf['max_val'].rank(ascending=False)
gtdf['max_val_pctile'] = gtdf['max_val'].rank(ascending=True) / len(gtdf)

for parish in gtdf.parish.unique():
    hdf = gtdf[gtdf['parish'] == parish]
    hdf = hdf.sort_values('tot_val', ascending=False)
    hdf['tot_val_rank'] = hdf['tot_val'].rank(ascending=False)
    hdf['tot_val_pctile'] = hdf['tot_val'].rank(ascending=True) / len(hdf)
    hdf['count_rank'] = hdf['count'].rank(ascending=False)
    hdf['count_pctile'] = hdf['count'].rank(ascending=True) / len(hdf)
    hdf['avg_val'] = hdf['tot_val'] / hdf['count']
    hdf['avg_val_rank'] = hdf['avg_val'].rank(ascending=False)
    hdf['avg_val_pctile'] = hdf['avg_val'].rank(ascending=True) / len(hdf)
    hdf['max_val_rank'] = hdf['max_val'].rank(ascending=False)
    hdf['max_val_pctile'] = hdf['max_val'].rank(ascending=True) / len(hdf)

    psdf.loc[psdf['parish'] == parish, f'count_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_rank'])))
    psdf.loc[psdf['parish'] == parish, f'tot_val_rank_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_rank'])))
    psdf.loc[psdf['parish'] == parish, f'tot_val_pctile_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['tot_val_pctile'])))
    psdf.loc[psdf['parish'] == parish, f'count_rank_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['count_rank'])))
    psdf.loc[psdf['parish'] == parish, f'count_pctile_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['count_pctile'])))
    psdf.loc[psdf['parish'] == parish, f'avg_val_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['avg_val'])))
    psdf.loc[psdf['parish'] == parish, f'avg_val_rank_1840'] = psdf['id'].map(
        dict(zip(hdf['id'], hdf['avg_val_rank'])))
    psdf.loc[psdf['parish'] == parish, f'avg_val_pctile_1840'] = psdf['id'].map(
        dict(zip(hdf['id'], hdf['avg_val_pctile'])))
    psdf.loc[psdf['parish'] == parish, f'max_val_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['max_val'])))
    psdf.loc[psdf['parish'] == parish, f'max_val_rank_1840'] = psdf['id'].map(
        dict(zip(hdf['id'], hdf['max_val_rank'])))
    psdf.loc[psdf['parish'] == parish, f'max_val_pctile_1840'] = psdf['id'].map(
        dict(zip(hdf['id'], hdf['max_val_pctile'])))
    psdf.loc[psdf['parish'] == parish, f'tot_val_1840'] = psdf['id'].map(dict(zip(hdf['id'], hdf['tot_val'])))

psdf.sort_values(['parish', 'tot_val_1524'], inplace=True)
psdf['recipient_surname'] = 0
psdf.loc[psdf['id'].isin(recipient_surnames), 'recipient_surname'] = 1
psdf['recipient_surname_43'] = 0
psdf.loc[psdf['id'].isin(recipient_surnames_43), 'recipient_surname_43'] = 1
psdf['recipient_surname_post_43'] = 0
psdf.loc[psdf['id'].isin(recipient_surnames_post_43), 'recipient_surname_post_43'] = 1

psdf.to_csv(f'{PROCESSED}/devon_surname_ranks_by_parish.csv', encoding='utf-8', index=False)

