import os
import ast
import json
import shutil
import platform
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col

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
IMAGES = 'Output/Images'
TABLES = 'Output/Tables'


with open(f'{PROCESSED}/pretty_dict.json', 'r', encoding='utf-8') as f:
    pretty_dict = json.load(f)

for measure in ['avg', 'max', 'tot']:
    pretty_measure = {'avg': 'Average', 'max': 'Maximum', 'tot': 'Total'}
    for year in [1524, 1543, 1581, 1674, 1840]:
        pretty_dict[f'{measure}_val_pctile_{year}'] = f'{pretty_measure[measure]} Value Percentile {year}'

with open(f'{PROCESSED}/pretty_dict.json', 'w', encoding='utf-8') as f:
    json.dump(pretty_dict, f, ensure_ascii=False, indent=4)
#%% Loading
df = pd.read_csv(f'{PROCESSED}/devon_surname_ranks.csv', encoding='utf-8')

vdf = pd.read_csv(f'{RAW}/victuallers_list_1651_1828_final.csv')
vdf['year'] = vdf['year'].astype(int)
vdf = vdf.loc[vdf['year'] >= 1730]
fdf = pd.read_csv(f'{RAW}/freeholders_list_1713_1780_final.csv')
fdf['year'] = fdf['year'].astype(int)
wdf = pd.read_csv(f'{PROCESSED}/ukda_pcc_wills_final.csv')
wdf['year'] = wdf['year'].astype(int)


#%% Dividing name lists into periods by year

for i in range(((1830-1730)//20)+1):
    vdf.loc[vdf['year'].between(1730+i*20, 1730+(i+1)*20), 'period'] = i
    pretty_dict[f'vlist_{int(i)}'] = f'{int(i*20+1730)}-{int((i+1)*20+1730)}'

for i in range(((1780-1710)//10)+1):
    fdf.loc[fdf['year'].between(1710+i*10, 1710+(i+1)*10), 'period'] = i
    pretty_dict[f'flist_{int(i)}'] = f'{int(i*10+1710)}-{int((i+1)*10+1710)}'

for i in range(((1860-1400)//20)+1):
    wdf.loc[wdf['year'].between(1400+i*20, 1400+(i+1)*20), 'period'] = i
    pretty_dict[f'wlist_{int(i)}'] = f'{int(i*20+1400)}-{int((i+1)*20+1400)}'


#%% Regressions on each period of freeholders

for period in vdf.period.unique():
    print(period)
print('========================')
for period in fdf.period.unique():
    print(period)
print('========================')
for period in wdf.period.unique():
    print(period)


#%% Assigning a "present in x period" variable to each name in the name list

def period_assigner(name_df):

    '''
    Assigns a "present in x list in y period" variable to each name in the name list.
    '''

    for period in vdf.period.unique():
        vdf_list = vdf.loc[vdf['period'] == period, 'id'].unique()
        name_df.loc[name_df['id'].isin(vdf_list), f'vlist_{int(period)}'] = 1
        name_df.loc[~name_df['id'].isin(vdf_list), f'vlist_{int(period)}'] = 0
    for period in fdf.period.unique():
        fdf_list = fdf.loc[fdf['period'] == period, 'id'].unique()
        name_df.loc[name_df['id'].isin(fdf_list), f'flist_{int(period)}'] = 1
        name_df.loc[~name_df['id'].isin(fdf_list), f'flist_{int(period)}'] = 0
    for period in wdf.period.unique():
        wdf_list = wdf.loc[wdf['period'] == period, 'id'].unique()
        name_df.loc[name_df['id'].isin(wdf_list), f'wlist_{int(period)}'] = 1
        name_df.loc[~name_df['id'].isin(wdf_list), f'wlist_{int(period)}'] = 0

    return name_df

#%%
df = period_assigner(df)
print(df)

#%% Victuallers List Regressions
vdf_results = []
vdf_periods = sorted(vdf.period.unique().tolist())
pretty_vdf_periods = [pretty_dict[f'vlist_{int(x)}'] for x in vdf_periods]
for period in vdf_periods:
    xvars = ['recipient_surname', 'recipient_control_group', 'avg_val_pctile_1524']
    rdf = df.copy()
    rdf = rdf.dropna(subset = xvars)
    y = rdf[f'vlist_{int(period)}']
    y.rename(pretty_dict, inplace=True)
    x = rdf[xvars]
    x.rename(columns=pretty_dict, inplace=True)

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit(cov_type='HC3',)
    vdf_results.append(model)
vdf_table = summary_col(vdf_results,
                        regressor_order = [pretty_dict[x] for x in xvars],
                        model_names=pretty_vdf_periods,
                        stars = True)
print(vdf_table)

#%% Freeholders List Regressions
fdf_results = []
fdf_periods = sorted(fdf.period.unique().tolist())
pretty_fdf_periods = [pretty_dict[f'flist_{int(x)}'] for x in fdf_periods]
for period in fdf_periods:
    xvars = ['recipient_surname', 'recipient_control_group', 'avg_val_pctile_1524']
    rdf = df.copy()
    rdf = rdf.dropna(subset = xvars)
    y = rdf[f'flist_{int(period)}']
    y.rename(pretty_dict, inplace=True)
    x = rdf[xvars]
    x.rename(columns=pretty_dict, inplace=True)

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit(cov_type='HC3',)
    fdf_results.append(model)
fdf_table = summary_col(fdf_results,
                        regressor_order = [pretty_dict[x] for x in xvars],
                        model_names=pretty_fdf_periods,
                        stars = True)
print(fdf_table)

#%% Wills List Regressions
wdf_results = []
wdf_periods = sorted(wdf.period.unique().tolist())
pretty_wdf_periods = [pretty_dict[f'wlist_{int(x)}'] for x in wdf_periods]
for period in wdf_periods:
    xvars = ['recipient_surname', 'recipient_control_group', 'avg_val_pctile_1524']
    rdf = df.copy()
    rdf = rdf.dropna(subset = xvars)
    y = rdf[f'wlist_{int(period)}']
    y.rename(pretty_dict, inplace=True)
    x = rdf[xvars]
    x.rename(columns=pretty_dict, inplace=True)

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit(cov_type='HC3',)
    wdf_results.append(model)
wdf_table = summary_col(wdf_results,
                        regressor_order = [pretty_dict[x] for x in xvars],
                        model_names=pretty_wdf_periods,
                        stars = True)
print(wdf_table)


