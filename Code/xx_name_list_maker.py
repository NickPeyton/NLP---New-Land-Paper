import os
import re

import numpy as np
np.bool = np.bool_
import pandas as pd

import phonetics as ph
import nltk

from tqdm.auto import tqdm
tqdm.pandas()

import statsmodels.api as sm

os.chdir('D:/PhD/DissolutionProgramming/LND---Land-Paper')

PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'

#%% Grabbing the data



sdf = pd.read_csv(f'{PROCESSED}/subsidy_master_data.csv')
subsidy_names = sdf['surname'].unique().tolist()
ndf = pd.read_csv(f'{PROCESSED}/surname_ids.csv')
ndf['surname_id'] = ndf['surname_id'].astype(int)

# create dictionary to assign id to each surname
surname_dict = dict(zip(ndf['surname'], ndf['surname_id']))

recipients_df = pd.read_csv(f'{PROCESSED}/recipient_matches.csv')
recipients_match_surnames = recipients_df['surname_id'].unique().tolist()

control_df = pd.read_csv(f'{PROCESSED}/control_individuals.csv')
control_match_surnames = control_df['match_surname_id'].unique().tolist()

recipient_surnames = pd.read_csv(f'{PROCESSED}/surname_matches.csv')
recipient_surnames = recipient_surnames['surname_id'].unique().tolist()

control_surnames = pd.read_csv(f'{PROCESSED}/control_surnames.csv')
control_surnames = control_surnames['match_surname_id'].unique().tolist()

prevalence_df = pd.DataFrame(columns=['indicator', 'recipient_match', 'control_match', 'match_t_test', 'match_p_val', 'recipient_surname', 'control_surname', 'surname_t_test', 'surname_p_val'])


#%%
def surname_id_assigner(row):
    surname = row['surname']
    surname = surname.capitalize()
    if not type(surname) == str:
        row['surname_id'] = np.nan
        return row
    if not surname.isalpha():
        row['surname_id'] = np.nan
        return row
    if surname in surname_dict:
        row['surname_id'] = int(surname_dict[surname])
    else:
        score_dict = {}
        for key in surname_dict:
            score_dict[key] = 0
            if surname == key:
                row['surname_id'] = surname_dict[key]
                break
            else:
                try:
                    if ph.soundex(surname) == ph.soundex(key):
                        score_dict[key] += 1
                    if ph.metaphone(surname) == ph.metaphone(key):
                        score_dict[key] += 1
                    if nltk.edit_distance(surname, key) < 2:
                        score_dict[key] += 1
                except:
                    pass
        if len(set(score_dict.values())) > 1:
            row['surname_id'] = int(surname_dict[max(score_dict, key=score_dict.get)])
        else:
            row['surname_id'] = int(max(surname_dict.values()) + 1)
            surname_dict[surname] = row['surname_id']

    return row

#%% Victuallers 1651-1828
vdf = pd.read_csv(f'{RAW}/victuallers_list_1651_1828.csv')
vdf_1600 = vdf[vdf['year'] < 1700]
vdf_1700 = vdf[vdf['year'] >= 1700 & (vdf['year'] < 1800)]
vdf_1800 = vdf[vdf['year'] >= 1800]

vdf_1600 = vdf_1600.progress_apply(surname_id_assigner, axis=1)
vdf_1700 = vdf_1700.progress_apply(surname_id_assigner, axis=1)
vdf_1800 = vdf_1800.progress_apply(surname_id_assigner, axis=1)

# Assign 1 to recipient_match var if surname_id is in recipients_match_surnames
# Assign 1 to control_match var if surname_id is in control_match_surnames
# Assign 1 to recipient_surname var if surname_id is in recipient_surnames
# Assign 1 to control_surname var if surname_id is in control_surnames

vdf_1600.loc[vdf_1600['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
vdf_1600.loc[vdf_1600['surname_id'].isin(control_match_surnames), 'control_match'] = 1
vdf_1700.loc[vdf_1700['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
vdf_1700.loc[vdf_1700['surname_id'].isin(control_match_surnames), 'control_match'] = 1
vdf_1800.loc[vdf_1800['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
vdf_1800.loc[vdf_1800['surname_id'].isin(control_match_surnames), 'control_match'] = 1

vdf_1600.loc[vdf_1600['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
vdf_1600.loc[vdf_1600['surname_id'].isin(control_surnames), 'control_surname'] = 1
vdf_1700.loc[vdf_1700['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
vdf_1700.loc[vdf_1700['surname_id'].isin(control_surnames), 'control_surname'] = 1
vdf_1800.loc[vdf_1800['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
vdf_1800.loc[vdf_1800['surname_id'].isin(control_surnames), 'control_surname'] = 1

vdf_1600[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = vdf_1600[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
vdf_1700[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = vdf_1700[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
vdf_1800[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = vdf_1800[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)

vdf_1600_match_t_test, vdf_1600_match_p_val, _ = sm.stats.ttest_ind(vdf_1600['recipient_match'], vdf_1600['control_match'])
vdf_1700_match_t_test, vdf_1700_match_p_val, _ = sm.stats.ttest_ind(vdf_1700['recipient_match'], vdf_1700['control_match'])
vdf_1800_match_t_test, vdf_1800_match_p_val, _ = sm.stats.ttest_ind(vdf_1800['recipient_match'], vdf_1800['control_match'])

vdf_1600_surname_t_test, vdf_1600_surname_p_val, _ = sm.stats.ttest_ind(vdf_1600['recipient_surname'], vdf_1600['control_surname'])
vdf_1700_surname_t_test, vdf_1700_surname_p_val, _ = sm.stats.ttest_ind(vdf_1700['recipient_surname'], vdf_1700['control_surname'])
vdf_1800_surname_t_test, vdf_1800_surname_p_val, _ = sm.stats.ttest_ind(vdf_1800['recipient_surname'], vdf_1800['control_surname'])

vdf_1600_recipient_surname = len(vdf_1600[vdf_1600['surname_id'].isin(recipient_surnames)])/len(vdf_1600)
vdf_1600_control_surname = len(vdf_1600[vdf_1600['surname_id'].isin(control_surnames)])/len(vdf_1600)
vdf_1700_recipient_surname = len(vdf_1700[vdf_1700['surname_id'].isin(recipient_surnames)])/len(vdf_1700)
vdf_1700_control_surname = len(vdf_1700[vdf_1700['surname_id'].isin(control_surnames)])/len(vdf_1700)
vdf_1800_recipient_surname = len(vdf_1800[vdf_1800['surname_id'].isin(recipient_surnames)])/len(vdf_1800)
vdf_1800_control_surname = len(vdf_1800[vdf_1800['surname_id'].isin(control_surnames)])/len(vdf_1800)

vdf_1600_recipient_match = len(vdf_1600[vdf_1600['surname_id'].isin(recipients_match_surnames)])/len(vdf_1600)
vdf_1600_control_match = len(vdf_1600[vdf_1600['surname_id'].isin(control_match_surnames)])/len(vdf_1600)
vdf_1700_recipient_match = len(vdf_1700[vdf_1700['surname_id'].isin(recipients_match_surnames)])/len(vdf_1700)
vdf_1700_control_match = len(vdf_1700[vdf_1700['surname_id'].isin(control_match_surnames)])/len(vdf_1700)
vdf_1800_recipient_match = len(vdf_1800[vdf_1800['surname_id'].isin(recipients_match_surnames)])/len(vdf_1800)
vdf_1800_control_match = len(vdf_1800[vdf_1800['surname_id'].isin(control_match_surnames)])/len(vdf_1800)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['victuallers_1600s'], 'recipient_match': [vdf_1600_recipient_match], 'control_match': [vdf_1600_control_match], 'match_t_test': [vdf_1600_match_t_test], 'match_p_val': [vdf_1600_match_p_val], 'recipient_surname': [vdf_1600_recipient_surname], 'control_surname': [vdf_1600_control_surname], 'surname_t_test': [vdf_1600_surname_t_test], 'surname_p_val': [vdf_1600_surname_p_val]})], ignore_index=True)
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['victuallers_1700s'], 'recipient_match': [vdf_1700_recipient_match], 'control_match': [vdf_1700_control_match],'match_t_test': [vdf_1700_match_t_test], 'match_p_val': [vdf_1700_match_p_val], 'recipient_surname': [vdf_1700_recipient_surname], 'control_surname': [vdf_1700_control_surname],  'surname_t_test': [vdf_1700_surname_t_test], 'surname_p_val': [vdf_1700_surname_p_val]})], ignore_index=True)
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['victuallers_1800s'], 'recipient_match': [vdf_1800_recipient_match], 'control_match': [vdf_1800_control_match],'match_t_test': [vdf_1800_match_t_test], 'match_p_val': [vdf_1800_match_p_val],  'recipient_surname': [vdf_1800_recipient_surname], 'control_surname': [vdf_1800_control_surname], 'surname_t_test': [vdf_1800_surname_t_test], 'surname_p_val': [vdf_1800_surname_p_val]})], ignore_index=True)

#%% Bankrupts 1800-1843
br1df = pd.read_csv(f'{RAW}/bankrupts_list_1800_1820.csv')
br1df = br1df.progress_apply(surname_id_assigner, axis=1)
br2df = pd.read_csv(f'{RAW}/bankrupts_list_1820_1843.csv')

for i, row in br2df.iterrows():
    text = row['text']
    surname = text.split(' ')[0].strip()
    br2df.at[i, 'surname'] = surname

br2df = br2df.progress_apply(surname_id_assigner, axis=1)

br1df.loc[br1df['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
br1df.loc[br1df['surname_id'].isin(control_match_surnames), 'control_match'] = 1
br2df.loc[br2df['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
br2df.loc[br2df['surname_id'].isin(control_match_surnames), 'control_match'] = 1

br1df.loc[br1df['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
br1df.loc[br1df['surname_id'].isin(control_surnames), 'control_surname'] = 1
br2df.loc[br2df['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
br2df.loc[br2df['surname_id'].isin(control_surnames), 'control_surname'] = 1

br1df[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = br1df[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
br2df[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = br2df[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)

br1_recipient_match_t_test, br1_recipient_match_p_val, _ = sm.stats.ttest_ind(br1df['recipient_match'], br1df['control_match'])
br1_recipient_surname_t_test, br1_recipient_surname_p_val, _ = sm.stats.ttest_ind(br1df['recipient_surname'], br1df['control_surname'])
br2_recipient_match_t_test, br2_recipient_match_p_val, _ = sm.stats.ttest_ind(br2df['recipient_match'], br2df['control_match'])
br2_recipient_surname_t_test, br2_recipient_surname_p_val, _ = sm.stats.ttest_ind(br2df['recipient_surname'], br2df['control_surname'])

br1_recipient_surname = len(br1df[br1df['surname_id'].isin(recipient_surnames)])/len(br1df)
br1_control_surname = len(br1df[br1df['surname_id'].isin(control_surnames)])/len(br1df)
br2_recipient_surname = len(br2df[br2df['surname_id'].isin(recipient_surnames)])/len(br2df)
br2_control_surname = len(br2df[br2df['surname_id'].isin(control_surnames)])/len(br2df)

br1_recipient_match = len(br1df[br1df['surname_id'].isin(recipients_match_surnames)])/len(br1df)
br1_control_match = len(br1df[br1df['surname_id'].isin(control_match_surnames)])/len(br1df)
br2_recipient_match = len(br2df[br2df['surname_id'].isin(recipients_match_surnames)])/len(br2df)
br2_control_match = len(br2df[br2df['surname_id'].isin(control_match_surnames)])/len(br2df)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['bankrupts_1800_20'], 'recipient_match': [br1_recipient_match], 'control_match': [br1_control_match], 'match_t_test': [br1_recipient_match_t_test], 'match_p_val': [br1_recipient_match_p_val],  'recipient_surname': [br1_recipient_surname], 'control_surname': [br1_control_surname], 'surname_t_test': [br1_recipient_surname_t_test], 'surname_p_val': [br1_recipient_surname_p_val]})], ignore_index=True)
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['bankrupts_1820_43'], 'recipient_match': [br2_recipient_match], 'control_match': [br2_control_match], 'match_t_test': [br2_recipient_match_t_test], 'match_p_val': [br2_recipient_match_p_val], 'recipient_surname': [br2_recipient_surname], 'control_surname': [br2_control_surname], 'surname_t_test': [br2_recipient_surname_t_test], 'surname_p_val': [br2_recipient_surname_p_val]})], ignore_index=True)

#%% Bankholders 1845-1880

bankdf = pd.read_csv(f'{RAW}/bank_returns_1845_1880.csv')
for i, row in bankdf.iterrows():
    name = row['name']
    surname = name.split(' ')[0].strip()
    bankdf.at[i, 'surname'] = surname

bankdf = bankdf.progress_apply(surname_id_assigner, axis=1)

bankdf.loc[bankdf['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
bankdf.loc[bankdf['surname_id'].isin(control_match_surnames), 'control_match'] = 1

bankdf.loc[bankdf['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
bankdf.loc[bankdf['surname_id'].isin(control_surnames), 'control_surname'] = 1

bankdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = bankdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)

bank_recipient_match_t_test, bank_recipient_match_p_val, _ = sm.stats.ttest_ind(bankdf['recipient_match'], bankdf['control_match'])
bank_recipient_surname_t_test, bank_recipient_surname_p_val, _ = sm.stats.ttest_ind(bankdf['recipient_surname'], bankdf['control_surname'])

bank_recipient_surname = len(bankdf[bankdf['surname_id'].isin(recipient_surnames)])/len(bankdf)
bank_control_surname = len(bankdf[bankdf['surname_id'].isin(control_surnames)])/len(bankdf)

bank_recipient_match = len(bankdf[bankdf['surname_id'].isin(recipients_match_surnames)])/len(bankdf)
bank_control_match = len(bankdf[bankdf['surname_id'].isin(control_match_surnames)])/len(bankdf)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['bankholders_1845_80'], 'recipient_match': [bank_recipient_match], 'control_match': [bank_control_match], 'match_t_test': [bank_recipient_match_t_test], 'match_p_val': [bank_recipient_match_p_val],'recipient_surname': [bank_recipient_surname], 'control_surname': [bank_control_surname], 'surname_t_test': [bank_recipient_surname_t_test], 'surname_p_val': [bank_recipient_surname_p_val]})], ignore_index=True)

#%% Indictable Offenses 1745-1782

indictdf = pd.read_csv(f'{RAW}/indictable_offenses_1745_1782.csv')
indictdf.dropna(inplace=True)
indictdf = indictdf[indictdf['person'] != '']
indictdf = indictdf[indictdf['person'] != np.nan]
indictdf = indictdf[~indictdf['person'].str.contains('DRO QS')]
for i, row in indictdf.iterrows():
    text = row['person']
    surname = text.split(' ')[1]
    indictdf.at[i, 'surname'] = surname

indictdf = indictdf.progress_apply(surname_id_assigner, axis=1)

indictdf.loc[indictdf['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
indictdf.loc[indictdf['surname_id'].isin(control_match_surnames), 'control_match'] = 1

indictdf.loc[indictdf['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
indictdf.loc[indictdf['surname_id'].isin(control_surnames), 'control_surname'] = 1
indictdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = indictdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
indict_recipient_match_t_test, indict_recipient_match_p_val, _ = sm.stats.ttest_ind(indictdf['recipient_match'], indictdf['control_match'])
indict_recipient_surname_t_test, indict_recipient_surname_p_val, _ = sm.stats.ttest_ind(indictdf['recipient_surname'], indictdf['control_surname'])

indict_recipient_surname = len(indictdf[indictdf['surname_id'].isin(recipient_surnames)])/len(indictdf)
indict_control_surname = len(indictdf[indictdf['surname_id'].isin(control_surnames)])/len(indictdf)

indict_recipient_match = len(indictdf[indictdf['surname_id'].isin(recipients_match_surnames)])/len(indictdf)
indict_control_match = len(indictdf[indictdf['surname_id'].isin(control_match_surnames)])/len(indictdf)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['indictments_1745_82'], 'recipient_match': [indict_recipient_match], 'control_match': [indict_control_match], 'match_t_test': [indict_recipient_match_t_test], 'match_p_val': [indict_recipient_match_p_val], 'recipient_surname': [indict_recipient_surname], 'control_surname': [indict_control_surname], 'surname_t_test': [indict_recipient_surname_t_test], 'surname_p_val': [indict_recipient_surname_p_val]})], ignore_index=True)

#%% Monumental Brasses
hdf = pd.read_csv(f'{RAW}/parishes_hundreds.csv')
parish_list = hdf['parish'].unique().tolist()
brassdf = pd.read_csv(f'{RAW}/monumental_brasses.csv')
for i, row in brassdf.iterrows():
    name = row['name']
    surname = name.split(',')[0]
    brassdf.at[i, 'surname'] = surname

brassdf = brassdf[~brassdf['surname'].isin(parish_list)]
brassdf = brassdf.progress_apply(surname_id_assigner, axis=1)

brassdf.loc[brassdf['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
brassdf.loc[brassdf['surname_id'].isin(control_match_surnames), 'control_match'] = 1

brassdf.loc[brassdf['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
brassdf.loc[brassdf['surname_id'].isin(control_surnames), 'control_surname'] = 1
brassdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = brassdf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
brass_recipient_match_t_test, brass_recipient_match_p_val, _ = sm.stats.ttest_ind(brassdf['recipient_match'], brassdf['control_match'])
brass_recipient_surname_t_test, brass_recipient_surname_p_val, _ = sm.stats.ttest_ind(brassdf['recipient_surname'], brassdf['control_surname'])

brass_recipient_surname = len(brassdf[brassdf['surname_id'].isin(recipient_surnames)])/len(brassdf)
brass_control_surname = len(brassdf[brassdf['surname_id'].isin(control_surnames)])/len(brassdf)

brass_recipient_match = len(brassdf[brassdf['surname_id'].isin(recipients_match_surnames)])/len(brassdf)
brass_control_match = len(brassdf[brassdf['surname_id'].isin(control_match_surnames)])/len(brassdf)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['monumental_brasses'], 'recipient_match': [brass_recipient_match], 'control_match': [brass_control_match], 'match_t_test': [brass_recipient_match_t_test], 'match_p_val': [brass_recipient_match_p_val], 'recipient_surname': [brass_recipient_surname], 'control_surname': [brass_control_surname],  'surname_t_test': [brass_recipient_surname_t_test], 'surname_p_val': [brass_recipient_surname_p_val]})], ignore_index=True)
#%% Workhouses 1861

workhousedf = pd.read_csv(f'{RAW}/workhouse_list_1861.csv')
for i, row in workhousedf.iterrows():
    name = row['Name']
    surname = name.split(' ')[-1]
    workhousedf.at[i, 'surname'] = surname

workhousedf = workhousedf.progress_apply(surname_id_assigner, axis=1)

workhousedf.loc[workhousedf['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
workhousedf.loc[workhousedf['surname_id'].isin(control_match_surnames), 'control_match'] = 1

workhousedf.loc[workhousedf['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
workhousedf.loc[workhousedf['surname_id'].isin(control_surnames), 'control_surname'] = 1
workhousedf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = workhousedf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)
workhouse_recipient_match_t_test, workhouse_recipient_match_p_val, _ = sm.stats.ttest_ind(workhousedf['recipient_match'], workhousedf['control_match'])
workhouse_recipient_surname_t_test, workhouse_recipient_surname_p_val, _ = sm.stats.ttest_ind(workhousedf['recipient_surname'], workhousedf['control_surname'])

workhouse_recipient_surname = len(workhousedf[workhousedf['surname_id'].isin(recipient_surnames)])/len(workhousedf)
workhouse_control_surname = len(workhousedf[workhousedf['surname_id'].isin(control_surnames)])/len(workhousedf)

workhouse_recipient_match = len(workhousedf[workhousedf['surname_id'].isin(recipients_match_surnames)])/len(workhousedf)
workhouse_control_match = len(workhousedf[workhousedf['surname_id'].isin(control_match_surnames)])/len(workhousedf)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'indicator': ['workhouse_inmates_1861'], 'recipient_match': [workhouse_recipient_match], 'control_match': [workhouse_control_match], 'match_t_test': [workhouse_recipient_match_t_test], 'match_p_val': [workhouse_recipient_match_p_val], 'recipient_surname': [workhouse_recipient_surname], 'control_surname': [workhouse_control_surname], 'surname_t_test': [workhouse_recipient_surname_t_test], 'surname_p_val': [workhouse_recipient_surname_p_val]})], ignore_index=True)

#%% Freeholders 1713-1780

weird_shit = re.compile(r'\{\w+\}')
freedf = pd.read_csv(f'{RAW}/freeholders_list_1713_1780.csv')
freedf = freedf.progress_apply(surname_id_assigner, axis=1)

freedf.loc[freedf['surname_id'].isin(recipients_match_surnames), 'recipient_match'] = 1
freedf.loc[freedf['surname_id'].isin(control_match_surnames), 'control_match'] = 1

freedf.loc[freedf['surname_id'].isin(recipient_surnames), 'recipient_surname'] = 1
freedf.loc[freedf['surname_id'].isin(control_surnames), 'control_surname'] = 1
freedf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']] = freedf[['recipient_match', 'control_match', 'recipient_surname', 'control_surname']].fillna(0)

free_recipient_match_t_test, free_recipient_match_p_val, _ = sm.stats.ttest_ind(freedf['recipient_match'], freedf['control_match'])
free_recipient_surname_t_test, free_recipient_surname_p_val, _ = sm.stats.ttest_ind(freedf['recipient_surname'], freedf['control_surname'])

free_recipient_surname = len(freedf[freedf['surname_id'].isin(recipient_surnames)])/len(freedf)
free_control_surname = len(freedf[freedf['surname_id'].isin(control_surnames)])/len(freedf)

free_recipient_match = len(freedf[freedf['surname_id'].isin(recipients_match_surnames)])/len(freedf)
free_control_match = len(freedf[freedf['surname_id'].isin(control_match_surnames)])/len(freedf)
#%%
prevalence_df = pd.concat([prevalence_df, pd.DataFrame({'Indicator': ['freeholders_1713_80'], 'recipient_match': [free_recipient_match], 'control_match': [free_control_match],  'match_t_test': [free_recipient_match_t_test], 'match_p_val': [free_recipient_match_p_val], 'recipient_surname': [free_recipient_surname], 'control_surname': [free_control_surname], 'surname_t_test': [free_recipient_surname_t_test], 'surname_p_val': [free_recipient_surname_p_val]})], ignore_index=True)

#%% Saving
prevalence_df.to_csv(f'{PROCESSED}/prevalence_ratios.csv', index=False)