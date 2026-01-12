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

fhdf = pd.read_csv(f'{RAW}/freeholders_list_1713_1780_final.csv', encoding='utf-8')

year_list = fhdf.year.unique().tolist()

yearly_df = pd.DataFrame()
for year in year_list:
    recipient_share = len(fhdf[(fhdf['year'] == year) & (fhdf['recipient_treatment_group'] == 1)]) / len(fhdf[fhdf['year'] == year])
    control_share = len(fhdf[(fhdf['year'] == year) & (fhdf['recipient_control_group'] == 1)]) / len(fhdf[fhdf['year'] == year])
    yearly_df = pd.concat([yearly_df, pd.DataFrame({'year': year, 'recipient_share': recipient_share, 'control_share': control_share}, index=[0])])

yearly_df['recipient_share_rolling'] = yearly_df['recipient_share'].rolling(window=5).mean()
yearly_df['control_share_rolling'] = yearly_df['control_share'].rolling(window=5).mean()
yearly_df.reset_index(drop=True, inplace=True)
# Plot the shares of recipients and controls over time
fig, ax = plt.subplots()
sns.lineplot(data=yearly_df, x='year', y='recipient_share_rolling', ax=ax, label='Recipient Share')
sns.lineplot(data=yearly_df, x='year', y='control_share_rolling', ax=ax, label='Control Share')
ax.set_title('Recipient and Control Share of Freeholders')
ax.set_xlabel('Year')
ax.set_ylabel('Share')
plt.savefig(f'{IMAGES}/freeholder_list_shares.png')
plt.show()
plt.close()

#%%

wdf = pd.read_csv(f'{RAW}/victuallers_list_1651_1828_final.csv', encoding='utf-8')
wdf = wdf.loc[(wdf['year'] <= 1760) & (wdf['year'] >= 1730)]
year_list = wdf.year.unique().tolist()

yearly_df = pd.DataFrame()
for year in year_list:
    recipient_share = len(wdf[(wdf['year'] == year) & (wdf['recipient_treatment_group'] == 1)]) / len(wdf[wdf['year'] == year])
    control_share = len(wdf[(wdf['year'] == year) & (wdf['recipient_control_group'] == 1)]) / len(wdf[wdf['year'] == year])
    yearly_df = pd.concat([yearly_df, pd.DataFrame({'year': year, 'recipient_share': recipient_share, 'control_share': control_share}, index=[0])])

yearly_df['recipient_share_rolling'] = yearly_df['recipient_share'].rolling(window=10).mean()
yearly_df['control_share_rolling'] = yearly_df['control_share'].rolling(window=10).mean()
yearly_df.reset_index(drop=True, inplace=True)
# Plot the shares of recipients and controls over time
fig, ax = plt.subplots()
sns.lineplot(data=yearly_df, x='year', y='recipient_share_rolling', ax=ax, label='Recipient Share')
sns.lineplot(data=yearly_df, x='year', y='control_share_rolling', ax=ax, label='Control Share')
ax.set_title('Recipient and Control Share of Licensed Victuallers')
ax.set_xlabel('Year')
ax.set_ylabel('Share')
plt.savefig(f'{IMAGES}/victualler_shares.png')
plt.show()
plt.close()

#%%

wdf = pd.read_csv(f'{PROCESSED}/ukda_pcc_wills_final.csv', encoding='utf-8')

year_list = wdf.year.unique().tolist()

yearly_df = pd.DataFrame()
for year in year_list:
    recipient_share = len(wdf[(wdf['year'] == year) & (wdf['recipient_treatment_group'] == 1)]) / len(wdf[wdf['year'] == year])
    control_share = len(wdf[(wdf['year'] == year) & (wdf['recipient_control_group'] == 1)]) / len(wdf[wdf['year'] == year])
    yearly_df = pd.concat([yearly_df, pd.DataFrame({'year': year, 'recipient_share': recipient_share, 'control_share': control_share}, index=[0])])

yearly_df['recipient_share_rolling'] = yearly_df['recipient_share'].rolling(window=20).mean()
yearly_df['control_share_rolling'] = yearly_df['control_share'].rolling(window=20).mean()
yearly_df.reset_index(drop=True, inplace=True)
# Plot the shares of recipients and controls over time
fig, ax = plt.subplots()
sns.lineplot(data=yearly_df, x='year', y='recipient_share_rolling', ax=ax, label='Recipient Share')
sns.lineplot(data=yearly_df, x='year', y='control_share_rolling', ax=ax, label='Control Share')
ax.set_title('Recipient and Control Share of PCC Wills')
ax.set_xlabel('Year')
ax.set_ylabel('Share')
plt.savefig(f'{IMAGES}/pcc_will_shares.png')
plt.show()
plt.close()

#%%
# Now let's do the same for surname status in 1524, 43, 81, 1674, and 1840

mdf = pd.read_csv(f'{PROCESSED}/master_subsidy_data_final.csv', encoding='utf-8')

subsidy_list = [1524, 1543, 1581]

subsidyly_df = pd.DataFrame()

for subsidy in subsidy_list:
    recipient_share = len(mdf[(mdf['year'] == subsidy) & (mdf['recipient_treatment_group'] == 1)]) / len(mdf[mdf['year'] == subsidy])
    control_share = len(mdf[(mdf['year'] == subsidy) & (mdf['recipient_control_group'] == 1)]) / len(mdf[mdf['year'] == subsidy])
    subsidyly_df = pd.concat([subsidyly_df, pd.DataFrame({'year': subsidy, 'recipient_share': recipient_share, 'control_share': control_share}, index=[0])])

# Plot the shares of recipients and controls over time
fig, ax = plt.subplots()
sns.lineplot(data=subsidyly_df, x='year', y='recipient_share', ax=ax, label='Recipient Share', marker='o')
sns.lineplot(data=subsidyly_df, x='year', y='control_share', ax=ax, label='Control Share', marker='o')
ax.set_title('Recipient and Control Share of Taxable Surnames')
ax.set_xlabel('year')
ax.set_ylabel('Share')
plt.savefig(f'{IMAGES}/surname_subsidy_shares.png')
plt.show()


#%% Let's do the same for shares of wealth

dsdf = pd.read_csv(f'{PROCESSED}/devon_surname_ranks.csv', encoding='utf-8')
dsdf['wealth_share_1524'] = dsdf['tot_val_1524'] / dsdf['tot_val_1524'].sum()
dsdf['wealth_share_1543'] = dsdf['tot_val_1543'] / dsdf['tot_val_1543'].sum()
dsdf['wealth_share_1581'] = dsdf['tot_val_1581'] / dsdf['tot_val_1581'].sum()
dsdf['wealth_share_1674'] = dsdf['tot_val_1674'] / dsdf['tot_val_1674'].sum()

yearly_df = pd.DataFrame()
for year in [1524, 1543, 1581]:
    recipient_share = dsdf[dsdf['recipient_treatment_group'] == 1][f'wealth_share_{year}'].sum()
    control_share = dsdf[dsdf['recipient_control_group'] == 1][f'wealth_share_{year}'].sum()
    print(f'{year}: Recipient Share: {recipient_share:.2%}, Control Share: {control_share:.2%}')
    yearly_df = pd.concat([yearly_df, pd.DataFrame({'year': year, 'recipient_share': recipient_share, 'control_share': control_share}, index=[0])])

# Plot the shares of recipients and controls over time
fig, ax = plt.subplots()
sns.lineplot(data=yearly_df, x='year', y='recipient_share', ax=ax, label='Recipient Share')
sns.lineplot(data=yearly_df, x='year', y='control_share', ax=ax, label='Control Share')
ax.set_title('Recipient and Control Share of Wealth')
ax.set_xlabel('Year')
ax.set_ylabel('Share')
plt.savefig(f'{IMAGES}/wealth_shares.png')
plt.show()


