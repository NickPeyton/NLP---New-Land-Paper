import os
import json
import platform
import pandas as pd

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

#%% Loading census data
pdf = pd.read_csv(f'{PROCESSED}/parish_data.csv')
phdf = pd.read_csv(f'{RAW}/parishes_hundreds.csv')
cdf = pd.read_csv(f'{RAW}/census_1801.csv', encoding='utf-8')
cdf.columns = [c.lower() for c in cdf.columns]
for col in cdf.columns:
    if cdf[col].dtype == 'O' or cdf[col].dtype == 'object':
        cdf[col] = cdf[col].str.replace('%', '', regex=False)
        cdf[col] = cdf[col].str.replace(',', '', regex=False)
        cdf[col] = cdf[col].str.replace('nan', 'np.nan', regex=False)
        cdf[col] = cdf[col].str.replace('£', '', regex=False)
        cdf[col] = cdf[col].str.strip()
        try:
            cdf[col] = pd.to_numeric(cdf[col])
        except:
            print('==================================================')
            print(f'{col} is not numeric')
            for line in cdf[col]:
                print(line)
            pass
with open(f'{PROCESSED}/census_parish_correction.json', 'r') as j:
    parish_name_correction = json.loads(j.read())

cdf['parish'] = cdf['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)
cdf['parish'] = cdf['parish'].apply(lambda x: x.strip().title())
pdf['parish'] = pdf['parish'].apply(lambda x: parish_name_correction[x] if x in parish_name_correction else x)
pdf['parish'] = pdf['parish'].apply(lambda x: x.strip().title())
# Need to combine: "Rewe (part, incl Up Exe)" and "Rewe (part, excl Up Exe)"

not_in_pdf = list(set(cdf['parish']) - set(pdf['parish']))
not_in_cdf = list(set(pdf['parish']) - set(cdf['parish']))


print(not_in_pdf)
print(not_in_cdf)
