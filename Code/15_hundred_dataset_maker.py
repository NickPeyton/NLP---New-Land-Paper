import os
import ast
import json
import shutil
import platform
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
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
parish_df = pd.read_csv(f'{PROCESSED}/devon_parish_flows.csv')
parish_df = parish_df.rename(columns={'PAR': 'parish'})
gini_df = pd.read_csv(f'{PROCESSED}/pred_ginis.csv',
                  encoding='utf-8')
phdf = pd.read_csv(f'{RAW}/parishes_hundreds.csv')
ph_dict = dict(zip(phdf['parish'], phdf['hundred']))


with open(f'{PROCESSED}/parish_correction.json', 'r') as j:
    parish_name_correction = json.loads(j.read())
    
hsdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks_by_hundred.csv', encoding='utf-8')
censusdf = pd.read_csv(f'{RAW}/devon_hundreds_census.csv', encoding='utf-8')

#%% Damage in 1674
damage_dict = {
    'Axminster': 1,
    'Bampton': 2,
    'Black Torrington': 1,
    'Braunton': 1,
    'Cliston': 1,
    'Coleridge': 2,
    'Colyton': 2,
    'Crediton': 1,
    'East Budleigh': 0,
    'Ermington': 2,
    'Exminster': 1,
    'Fremington': 1,
    'Halberton': 1,
    'Hartland': 2,
    'Hayridge': 2,
    'Haytor': 1,
    'Hemyock': 2,
    'Lifton': 1,
    'North Tawton': 1,
    'Ottery St Mary': 0,
    'Plympton': 1,
    'Roborough': 1,
    'Shebbear': 1,
    'Shirwell': 1,
    'South Molton': 1,
    'Stanborough': 1,
    'Tavistock': 2,
    'Teignbridge': 1,
    'Tiverton': 2,
    'West Budleigh': 0,
    'Witheridge': 1,
    'Wonford': 1,
}

hsdf['damage_1674'] = hsdf['hundred'].map(damage_dict)

#%% Correlations in surname ranks
assessment_pair_list = []
year_list = [1524, 1543, 1581, 1674, 1840]
for i, year in enumerate(year_list):
    for year2 in year_list[i+1:]:
        assessment_pair_list.append([year, year2])
print(assessment_pair_list)
for hundred in hsdf.hundred.unique().tolist():
    for year_pair in assessment_pair_list:
        year1 = year_pair[0]
        year2 = year_pair[1]
        sub_df = hsdf[hsdf['hundred'] == hundred].copy()
        sub_df.dropna(subset=[f'tot_val_rank_{year1}', f'tot_val_rank_{year2}'], inplace=True)
        if len(sub_df) < 2:
            continue
        corr = sub_df[f'tot_val_rank_{year1}'].corr(sub_df[f'tot_val_rank_{year2}'])
        avg_val_corr = sub_df[f'avg_val_rank_{year1}'].corr(sub_df[f'avg_val_rank_{year2}'])
        max_val_corr = sub_df[f'max_val_rank_{year1}'].corr(sub_df[f'max_val_rank_{year2}'])
        if np.isnan(corr):
            continue
        hsdf.loc[hsdf['hundred'] == hundred, f'tot_val_rank_corr_{year1}_{year2}'] = corr
        hsdf.loc[hsdf['hundred'] == hundred, f'avg_val_rank_corr_{year1}_{year2}'] = avg_val_corr
        hsdf.loc[hsdf['hundred'] == hundred, f'max_val_rank_corr_{year1}_{year2}'] = max_val_corr

#%% Share of largest 1, 2, 3 families

for hundred in hsdf.hundred.unique().tolist():
    for year in [1524, 1543, 1581, 1674, 1840]:
        sub_df = hsdf[hsdf['hundred'] == hundred].copy()
        total_wealth = sub_df[f'tot_val_{year}'].sum()
        if total_wealth == 0:
            print(f'{hundred} {year}')
        sub_df.sort_values(by=f'tot_val_{year}', ascending=False, inplace=True)
        sub_df['wealth_share'] = sub_df[f'tot_val_{year}'] / total_wealth
        for i in range(1, 4):
            hsdf.loc[hsdf['hundred'] == hundred, f'wealth_share_{i}_{year}'] = sub_df['wealth_share'].head(i).sum()


        rsdf = sub_df[sub_df['recipient_surname'] == 1]
        rsdf_43 = sub_df[sub_df['recipient_surname_43'] == 1]
        rsdf_post_43 = sub_df[sub_df['recipient_surname_post_43'] == 1]
        rs_total = rsdf[f'tot_val_{year}'].sum()
        rc_df = sub_df[sub_df['recipient_control_group'] == 1]
        rc_df_43 = sub_df[sub_df['recipient_control_group_43'] == 1]
        rc_df_post_43 = sub_df[sub_df['recipient_control_group_post_43'] == 1]
        rc_total = rc_df[f'tot_val_{year}'].sum()
        hsdf.loc[hsdf['hundred'] == hundred, f'wealth_share_recipient_{year}'] = rs_total / total_wealth
        hsdf.loc[hsdf['hundred'] == hundred, f'wealth_share_recipient_control_{year}'] = rc_total / total_wealth

year_list = [1524, 1543, 1581, 1674, 1840]

for hundred in hsdf['hundred'].unique():
    hundred_df = hsdf[hsdf['hundred'] == hundred].copy()
    for i, year in enumerate(year_list):
        hundred_df[f'recipient_count_{year}'] = hundred_df['recipient_treatment_group'] * hundred_df[f'count_{year}']
        hundred_df[f'recipient_count_share_{year}'] = hundred_df[f'recipient_count_{year}'] / hundred_df[f'count_{year}'].sum()
        hundred_df[f'control_count_{year}'] = hundred_df['recipient_control_group'] * hundred_df[f'count_{year}']
        hundred_df[f'control_count_share_{year}'] = hundred_df[f'control_count_{year}'] / hundred_df[f'count_{year}'].sum()
        hundred_df[f'recipient_tot_val_{year}'] = hundred_df['recipient_treatment_group'] * hundred_df[f'tot_val_{year}']
        hundred_df[f'recipient_tot_val_share_{year}'] = hundred_df[f'recipient_tot_val_{year}'] / hundred_df[f'tot_val_{year}'].sum()
        hundred_df[f'control_tot_val_{year}'] = hundred_df['recipient_control_group'] * hundred_df[f'tot_val_{year}']
        hundred_df[f'control_tot_val_share_{year}'] = hundred_df[f'control_tot_val_{year}'] / hundred_df[f'tot_val_{year}'].sum()
        for col in [f'recipient_count_{year}', f'control_count_{year}', f'recipient_count_share_{year}', f'control_count_share_{year}',
                    f'recipient_tot_val_{year}', f'control_tot_val_{year}', f'recipient_tot_val_share_{year}', f'control_tot_val_share_{year}']:
            hsdf.loc[hsdf['hundred'] == hundred, col] = hundred_df[col].sum()
        for year2 in year_list[i + 1:]:
            if year >= year2:
                continue
            for col in ['tot_val_rank', 'avg_val_rank', 'max_val_rank', 'tot_val_pctile', 'avg_val_pctile', 'max_val_pctile']:
                col1 = f'{col}_{year}'
                col2 = f'{col}_{year2}'
                corr = hundred_df[col1].corr(hundred_df[col2])
                hsdf.loc[hsdf['hundred'] == hundred, f'{col}_{year}_{year2}_corr'] = corr

        #%%
mean_list = [x for x in hsdf.columns if 'wealth_share' in x or 'control' in x or 'recipient' in x or 'found' in x or 'corr' in x]
agg_dict = {x: 'mean' for x in mean_list}

for year in year_list:
    agg_dict[f'count_{year}'] = 'sum'
    agg_dict[f'tot_val_{year}'] = 'sum'
    agg_dict[f'avg_val_{year}'] = 'mean'
    agg_dict[f'max_val_{year}'] = 'mean'
    agg_dict[f'avg_val_pctile_{year}'] = 'mean'
    agg_dict[f'max_val_pctile_{year}'] = 'mean'
    agg_dict[f'tot_val_pctile_{year}'] = 'mean'
hundred_df = hsdf.groupby('hundred').agg(agg_dict)

#%% Attaching census data

censusdf['hundred'] = censusdf['hundred'].str.title()
hundred_df = hundred_df.merge(censusdf, on='hundred', how='left')


#%% Attaching parish data

parish_agg_dict = {'mean_elev': 'mean',
                   'mean_slope': 'mean',
                   'wheatsuit': 'mean',
                   'lspc1332': 'mean',
                   'ind_1831': 'mean',
                   'agr_1831': 'mean',
                   'agr_share': 'mean',
                   'ind_share': 'mean',
                   'mills_1400': 'sum',
                   'gent_1400': 'sum',
                   'NrPatents': 'sum',
                   'copys_1850': 'sum',
                   'mills': 'sum',
                   'NrGentry': 'sum',
                   'area': 'sum',
                   'landOwned': 'sum',
                   'WheatYield': 'mean',
                   'copys_1516': 'sum',
                   'hrv_land': 'sum',
                   'lspc1525': 'mean',
                   'distriver': 'mean',
                   'distmkt': 'mean',
                   'distcoal': 'mean',
                   'hrv_dums': 'sum',
                   'latitude': 'mean',
                   'longitude': 'mean',
                   }


hundred_correction_dict = {'Bampton (Devon)': 'Bampton',
                           'Exeter City': 'Wonford',
                           'Sherwill': 'Shirwell',
                           'Plymouth Borough': 'Roborough',
                           }
parish_df['hrv_land'] = parish_df['hrv_land'] * 240
parish_df['hrv_dums'] = 0
parish_df.loc[parish_df['hrv_land'] > 0, 'hrv_dums'] = 1
parish_df['parish'] = parish_df['parish'].str.replace('\'', '')
parish_df['parish'] = parish_df['parish'].str.replace('Exeter, ', '')
parish_df['parish'] = parish_df['parish'].str.replace('Exeter ', '')
parish_df['parish'] = parish_df['parish'].str.replace('Plymouth, ', '')
parish_df['parish'] = parish_df['parish'].str.replace(',', '')
parish_df['parish'] = parish_df['parish'].str.title()
parish_df['parish'] = parish_df['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)
parish_df['hundred'] = parish_df['hundred'].apply(lambda x: x.strip().title() if type(x) == str else x)
parish_df['hundred'] = parish_df['parish'].apply(lambda x: ph_dict[x] if x in ph_dict else x)
parish_df['hundred'] = parish_df['hundred'].apply(lambda x: hundred_correction_dict[x] if x in hundred_correction_dict else x)
parish_df['hundred'] = parish_df['hundred'].apply(lambda x: 'Wonford' if x in ['Exeter All Hallows Goldsmith Street', 'Exeter All Hallows On The Walls', 'Exeter Bedford Circus', 'Exeter Bradninch Precinct', 'Exeter Castle Yard', 'Exeter Cathedral Close', 'Exeter St Edmund', 'Exeter St George The Martyr', 'Exeter St John', 'Exeter St Kerrian', 'Exeter St Lawrence', 'Exeter St Martin', 'Exeter St Mary Arches', 'Exeter St Mary Major', 'Exeter St Olave', 'Exeter St Pancras', 'Exeter St Paul', 'Exeter St Petrock', 'Exeter St Stephen', 'Exeter St Thomas The Apostle'] else x)

remove_list = ['Beaminster forum', 'Stratton', 'Whitchurch Canonicorum']
parish_df = parish_df[~parish_df['hundred'].isin(remove_list)]

aggdf = parish_df.groupby('hundred').agg(parish_agg_dict)
hundred_df = hundred_df.merge(aggdf, on='hundred', how='left')
hundred_df.to_csv(f'{PROCESSED}/hundred_data.csv', index=False)
