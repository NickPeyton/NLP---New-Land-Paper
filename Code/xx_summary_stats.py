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

#%%

mdf = pd.read_csv(f'{PROCESSED}/master_subsidy_data_final.csv', encoding='utf-8')

parishes_1524 = set(mdf.loc[mdf['year'] == 1524, 'gemini_parish'].unique().tolist())
parishes_1543 = set(mdf.loc[mdf['year'] == 1543, 'gemini_parish'].unique().tolist())
parishes_1581 = set(mdf.loc[mdf['year'] == 1581, 'gemini_parish'].unique().tolist())
overlap_parishes = parishes_1524.intersection(parishes_1543).intersection(parishes_1581)

mdf = mdf.loc[mdf['gemini_parish'].isin(overlap_parishes)]



cdf = mdf.loc[(((mdf['gemini_value'] >= 3) & (mdf['gemini_type'] == 'G')) |
               (mdf['gemini_type'] == 'L') |
               (mdf['gemini_type'] == 'W')) &
              (mdf['year'] < 1674)]
ldf = mdf.loc[(mdf['gemini_type'] == 'L') & (mdf['year'] < 1674)]
# Get mean, sum, sd, count for 1524, 43, 81
agg_dict = {
    'gemini_value': ['mean', 'median', 'std', 'sum', 'count'],

}

gdf = cdf.groupby('year').agg(agg_dict).reset_index()
gdf['year'] = gdf['year'].astype(int)
tdf = gdf.T
year_list = list(tdf.iloc[0])
year_list = [str(int(x)) for x in year_list]
tdf.columns = year_list
tdf = tdf[1:].round(2)
tdf = tdf.droplevel(level=0)
tdf.to_latex(f'{TABLES}/subsidy_summary_stats.tex', float_format="%.2f", escape=False)
sdf = tdf.copy()


gldf = ldf.groupby('year').agg(agg_dict).reset_index()
gldf['year'] = gldf['year'].astype(int)
gldf = gldf.T
year_list = list(gldf.iloc[0])
year_list = [str(int(x)) + ', \\pounds' for x in year_list]
gldf.columns = year_list
gldf = gldf[1:].round(2)
gldf = gldf.droplevel(level=0)
gldf.to_latex(f'{TABLES}/subsidy_land_summary_stats.tex', float_format="%.2f", escape=False)

htdf = mdf.loc[mdf['year'] == 1674]
ghdf = htdf.groupby('year').agg(agg_dict).reset_index()
ghdf = ghdf.T
year_list = list(ghdf.iloc[0])
year_list = [str(int(x)) + ' Hearths' for x in year_list]
ghdf.columns = year_list
ghdf = ghdf[1:].round(2)
ghdf = ghdf.droplevel(level=0)
ghdf.to_latex(f'{TABLES}/hearth_tax_summary_stats.tex', float_format="%.2f", escape=False)

#%% Tithe map data
agg_dict = {
    'area_acres': ['mean', 'median', 'std', 'sum', 'count'],
}
tdf = pd.read_csv(f'{PROCESSED}/tithe_landowners.csv', encoding='utf-8')

tdf['area_acres'] = tdf['area_perches'] / 160
print(tdf.area_acres.sum())
tdf['owner_name'] = tdf['owner_forename'] + ' ' + tdf['owner_surname']
tdf = tdf.groupby('owner_name').agg({'area_acres': 'sum'}).reset_index()
print(tdf.area_acres.sum())

#%%
tdf['cat'] = '1840 Acres'
gtdf = tdf.groupby('cat').agg(agg_dict).reset_index()
gtdf = gtdf.T
gtdf.columns = ['1840 Acres']
gtdf = gtdf.droplevel(level=0).round(2)[1:]
gtdf.to_latex(f'{TABLES}/tithe_summary_stats.tex', float_format="%.2f", escape=False)

#%% Join them all

alldf = pd.concat([sdf, ghdf, gtdf], axis=1)
alldf.to_latex(f'{TABLES}/summary_stats.tex', float_format="%.2f", escape=False)