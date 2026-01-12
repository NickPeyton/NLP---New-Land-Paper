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
doc_list = [
f'{RAW}/monumental_brasses_final.csv',
f'{RAW}/victuallers_list_1651_1828_final.csv',
f'{RAW}/freeholders_list_1713_1780_final.csv',
f'{PROCESSED}/ukda_pcc_wills_final.csv',
f'{RAW}/bank_returns_1845_1880_final.csv',
f'{RAW}/bankrupts_list_1800_1820_final.csv',
f'{RAW}/bankrupts_list_1820_1843_final.csv',
f'{RAW}/indictable_offenses_1745_1782_final.csv',
f'{RAW}/workhouse_list_1861_final.csv',
    ]
desc_list = [
    'Monumental Brasses',
    'Victuallers 1651-1828',
    'Freeholders List 1713-1780',
    'PCC Wills 1558-1858',
    'Bank Returns 1845-1880',
    'Bankrupts List 1800-1820',
    'Bankrupts List 1820-1843',
    'Indictments 1745-1782',
    'Workhouse List 1861',
]
results_df = pd.DataFrame(columns=['Indicator', 'Recipient', 'Control', 'Ratio', 'P-Value'])
for doc_file, description in zip(doc_list, desc_list):
    df = pd.read_csv(doc_file, encoding='utf-8')
    if 'ukda_pcc_wills_final' in doc_file:
        df = df.loc[df['year'] >= 1558]
    indicator = description
    recipient_share = len(df[df['recipient_treatment_group'] == 1]) / len(df)
    control_share = len(df[df['recipient_control_group'] == 1]) / len(df)
    ratio = recipient_share / control_share
    t_test = sm.stats.ttest_ind(df['recipient_treatment_group'], df['recipient_control_group'])
    p_value = t_test[1]
    results_df = pd.concat([results_df, pd.DataFrame({'Indicator': indicator, 'Recipient': recipient_share, 'Control': control_share, 'Ratio': ratio, 'P-Value': p_value}, index=[0])])

# Export to Latex, with some formatting
for col in ['Recipient', 'Control']:
    results_df[col] = results_df[col].apply(lambda x: f'{x:.2%}')
results_df['Ratio'] = results_df['Ratio'].apply(lambda x: f'{x:.2f}')
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f'{x:.3f}')
# Add some stars
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f'{x}***' if float(x) < .001 else f'{x}**' if float(x) < .01 else f'{x}*' if float(x) < .05 else x)

latex_text = results_df.to_latex(index=False)
latex_text = '\n\\begin{table}[H]\n\\centering\n\\resizebox{\\textwidth}{!}{\n' + latex_text + '\n}\n\\end{table}\n'

with open(f'{TABLES}/status_indicators.tex', 'w', encoding='utf-8') as f:
    f.write(latex_text)

print(results_df)