import re
import os
import csv
import numpy as np
import pandas as pd
import platform
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

WILLS = 'Data/Raw/pcc_wills'
UKDA = 'Data/Raw/ukda_pcc_wills'
#%% Regexes

see_other = re.compile(r'[A-Z]+ see [A-Z]+')
parenthetical = re.compile(r'[A-Z]+ (\([A-Z]+\))')

#%% Processing the pcc wills data
big_df = pd.DataFrame()
for file in os.listdir(WILLS):
    print('==================================================')
    print(file)
    print('------------')
    if 'wills' not in file:
        continue
    df = pd.read_csv(os.path.join(WILLS, file))
    if file == 'wills_1630_1630.csv':
        df['YEAR'] = 1630
    for col in df.columns:
        df.rename(columns={col: col.strip()}, inplace=True)
        if 'OCCPN' in col or 'STATUS' in col:
            df.rename(columns={col.strip(): 'OCCPN'}, inplace=True)
        if col.strip() == 'NAME':
            df.rename(columns={col.strip(): 'SURNAME'}, inplace=True)
    df = df[['SURNAME', 'FORENAME', 'OCCPN', 'PLACE', 'YEAR']].copy()
    new_df = pd.DataFrame()
    print(len(df))
    for i, row in df.iterrows():
        added_row = False
        if (row['SURNAME'] == '' or pd.isna(row['SURNAME'])) and pd.isna(row['YEAR']) and (row['PLACE'] == '' or pd.isna(row['PLACE'])):
            continue

        if type(row['SURNAME']) == str:
            if re.search(see_other, row['SURNAME']):
                continue
        if (row['SURNAME'] == '' or pd.isna(row['SURNAME'])) and not (row['PLACE'] == '' or pd.isna(row['PLACE'])):
            new_df.loc[len(new_df)-1, 'PLACE'] = str(new_df.loc[len(new_df)-1, 'PLACE']) + ' ' + str(row['PLACE'])
            added_row = True
        if (row['SURNAME'] == '' or pd.isna(row['SURNAME'])) and pd.isna(new_df.iloc[len(new_df)-1]['YEAR']) and not pd.isna(row['YEAR']):
            new_df.loc[len(new_df)-1, 'YEAR'] = row['YEAR']
            added_row = True

        if len(new_df) > 0:
            if not pd.isna(new_df.iloc[len(new_df)-1]['YEAR']) and not pd.isna(row['YEAR']) and pd.isna(row['SURNAME']):
                added_row = True
        if added_row:
            continue
        new_row = row.copy()
        if pd.isna(new_row['PLACE']):
            new_row['PLACE'] = ''
        if re.search(parenthetical, new_row['SURNAME']):
            new_row['SURNAME'] = re.search(parenthetical, new_row['SURNAME']).group(1)
        new_row['SURNAME'] = new_row['SURNAME'].title()
        if type(new_row['YEAR']) == str:
            try:
                new_row['YEAR'] = new_row['YEAR'].replace('?', '').replace('(sic)', '').strip()
                if 'p' in new_row['YEAR']:
                    new_row['YEAR'] = new_row['YEAR'][:4]
                new_row['YEAR'] = int(new_row['YEAR'])
            except:
                print(new_row)
                continue
        if new_row['YEAR'] < 1300:
            new_row['YEAR'] = np.nan
        new_df = pd.concat([new_df, pd.DataFrame(new_row).T], ignore_index=True)
    print(len(new_df))
    big_df = pd.concat([big_df, new_df], ignore_index=True)
print(len(big_df))
big_df = big_df.loc[~pd.isna(big_df['SURNAME'])]
print(len(big_df))
big_df = big_df.loc[~pd.isna(big_df['YEAR'])]
print(len(big_df))

big_df['YEAR'] = big_df['YEAR'].astype(int)

big_df = big_df.loc[big_df['YEAR'] > 1300]
big_df.sort_values(['YEAR', 'SURNAME'], inplace=True)
big_df.columns = [x.lower() for x in big_df.columns]
big_df.to_csv(f'{PROCESSED}/pcc_wills.csv', index=False)

#%% UKDA wills
big_df = pd.DataFrame()
for file in os.listdir(UKDA):
    print(file)
    with open(os.path.join(UKDA, file)) as tabfile:
        reader = csv.reader(tabfile, delimiter='\t')
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
    df = pd.DataFrame(data, columns=header)
    df = df[['surname', 'forename', 'title', 'occupation', 'parish_place', 'county_country', 'year']].copy()
    big_df = pd.concat([big_df, df], ignore_index=True)

# Select only rows with "Devon" in the 'county_country' column
big_df = big_df.loc[~pd.isna(big_df['surname'])]
big_df = big_df.loc[~pd.isna(big_df['year'])]
big_df = big_df.loc[big_df['county_country'].str.contains('Devon', na=False)]
big_df.drop(columns='county_country', inplace=True)
big_df.sort_values(['year', 'surname', 'forename'], inplace=True)
big_df.to_csv(f'{PROCESSED}/ukda_pcc_wills.csv', index=False)
