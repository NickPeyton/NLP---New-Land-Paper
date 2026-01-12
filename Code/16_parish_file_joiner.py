import os

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

tqdm.pandas()
import os
import ast
import json
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

# %% Load parish csv and hundred csv
phdf = pd.read_csv(f'{RAW}/parishes_hundreds.csv')
phdf_p = phdf['parish'].tolist()
phdf_h = phdf['hundred'].tolist()
ph_dict = dict(zip(phdf['parish'], phdf['hundred']))

psdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks_by_parish.csv', encoding='utf-8')

pdf = pd.read_csv(f'{PROCESSED}/devon_parish_flows.csv')

with open(f'{PROCESSED}/parish_correction.json', 'r') as j:
    parish_name_correction = json.loads(j.read())
pdf = pdf[['PAR', 'hundred', 'terrain', 'mean_elev', 'mean_slope', 'wheatsuit', 'lspc1332', 'agsh1370',
           'indsh1370',
           'ind_1831', 'agr_share', 'agr_1831', 'ind_share', 'mills_1400', 'gent_1400', 'NrPatents', 'copys_1850', 'mills',
           'NrGentry', 'area', 'landOwned', 'WheatYield', 'copys_1516', 'hrv_land', 'lspc1525',
           'distriver', 'distmkt', 'distcoal', 'latitude', 'longitude']]

pdf['hrv_land'] = pdf['hrv_land'] * 240
pdf['hrv_dums'] = 0
pdf.loc[pdf['hrv_land'] > 0, 'hrv_dums'] = 1



# %% Correlations in surname ranks
assessment_pair_list = []
year_list = [1524, 1543, 1581, 1674, 1840]
for i, year in enumerate(year_list):
    for year2 in year_list[i + 1:]:
        assessment_pair_list.append([year, year2])
# %%
pdf.rename(columns={'PAR': 'parish'}, inplace=True)
pdf['parish'] = pdf['parish'].str.replace('\'', '')
pdf['parish'] = pdf['parish'].str.title()
pdf['parish'] = pdf['parish'].apply(lambda x: x.strip().replace(',', '').replace('Exeter ', ''))
pdf['parish'] = pdf['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)


# %% Correcting hundreds
hundred_correction_dict = {'Bampton (Devon)': 'Bampton',
                           'Exeter City': 'Wonford',
                           'Sherwill': 'Shirwell',
                           'Plymouth Borough': 'Roborough',
                           }
pdf['hundred'] = pdf['hundred'].apply(lambda x: x.strip().title() if type(x) == str else x)
pdf['hundred'] = pdf['parish'].apply(lambda x: ph_dict[x] if x in ph_dict else x)
pdf['hundred'] = pdf['hundred'].apply(lambda x: hundred_correction_dict[x] if x in hundred_correction_dict else x)
remove_list = ['Beaminster forum', 'Stratton', 'Whitchurch Canonicorum']
pdf = pdf[~pdf['hundred'].isin(remove_list)]


#%% Calculating correlations between total, average, and max value ranks

year_list = [1524, 1543, 1581, 1674, 1840]

for parish in psdf['parish'].unique():
    parish_df = psdf[psdf['parish'] == parish].copy()



    for i, year in enumerate(year_list):

        parish_df[f'recipient_count_{year}'] = parish_df['recipient_treatment_group'] * parish_df[f'count_{year}']
        parish_df[f'recipient_count_share_{year}'] = parish_df[f'recipient_count_{year}'] / parish_df[f'count_{year}'].sum()

        parish_df[f'control_count_{year}'] = parish_df['recipient_control_group'] * parish_df[f'count_{year}']
        parish_df[f'control_count_share_{year}'] = parish_df[f'control_count_{year}'] / parish_df[f'count_{year}'].sum()

        parish_df[f'recipient_tot_val_{year}'] = parish_df['recipient_treatment_group'] * parish_df[f'tot_val_{year}']
        parish_df[f'recipient_tot_val_share_{year}'] = parish_df[f'recipient_tot_val_{year}'] / parish_df[f'tot_val_{year}'].sum()
        parish_df[f'control_tot_val_{year}'] = parish_df['recipient_control_group'] * parish_df[f'tot_val_{year}']
        parish_df[f'control_tot_val_share_{year}'] = parish_df[f'control_tot_val_{year}'] / parish_df[f'tot_val_{year}'].sum()

        for col in [f'recipient_count_{year}', f'control_count_{year}', f'recipient_count_share_{year}', f'control_count_share_{year}',
                    f'recipient_tot_val_{year}', f'control_tot_val_{year}', f'recipient_tot_val_share_{year}', f'control_tot_val_share_{year}']:
            psdf.loc[psdf['parish'] == parish, col] = parish_df[col].sum()
        for year2 in year_list[i + 1:]:
            if year >= year2:
                continue
            for col in ['tot_val_rank', 'avg_val_rank', 'max_val_rank', 'tot_val_pctile', 'avg_val_pctile', 'max_val_pctile']:
                col1 = f'{col}_{year}'
                col2 = f'{col}_{year2}'
                corr = parish_df[col1].corr(parish_df[col2])
                psdf.loc[psdf['parish'] == parish, f'{col}_{year}_{year2}_corr'] = corr
#%%
parish_agg_dict = {x: 'mean' for x in psdf.columns if 'corr' in x or 'rank' in x or 'group' in x or 'count' in x or 'share' in x}
for year in year_list:
    parish_agg_dict[f'count_{year}'] = 'sum'
    parish_agg_dict[f'tot_val_{year}'] = 'sum'
    parish_agg_dict[f'avg_val_{year}'] = 'mean'
    parish_agg_dict[f'max_val_{year}'] = 'mean'
    parish_agg_dict[f'avg_val_pctile_{year}'] = 'mean'
    parish_agg_dict[f'max_val_pctile_{year}'] = 'mean'
    parish_agg_dict[f'tot_val_pctile_{year}'] = 'mean'


gpsdf = psdf.groupby('parish').agg(parish_agg_dict)


#%% Merging pdf and psdf
pdf['lland'] = np.log(pdf['landOwned'] + 1)
pdf = pdf.merge(gpsdf, on='parish', how='left')

#%%
pdf.to_csv(f'{PROCESSED}/parish_data.csv', index=False)
