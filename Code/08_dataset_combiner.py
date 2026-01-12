import numpy as np
import pandas as pd
import os
import platform
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

#%%
new_df = pd.DataFrame()
for subsidy in [
    1524,
    1543,
    1581,
    # 1642,
    # 1647,
    # 1660,
    # 1661,
    1674,
     ]:
    SUBSIDY_FOLDER = f'Data/Processed/subsidy{subsidy}'
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv')
    tdf = tdf.loc[tdf['gemini_value'] > 0]
    tdf['year'] = subsidy
    new_df = pd.concat([new_df, tdf])
new_df.to_csv('Data/Processed/all_taxpayers.csv', encoding='utf-8', index=False)

#%%

df = pd.read_csv('Data/Processed/all_taxpayers.csv')
df = df.loc[df['year'] != 1642]
df = df.loc[df['year'] != 1647]
df = df.loc[df['year'] != 1660]
df = df.loc[df['year'] != 1661]

df.to_csv('Data/Processed/master_subsidy_data.csv', encoding='utf-8', index=False)