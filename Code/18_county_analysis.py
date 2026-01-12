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

TABLES = 'Output/Tables'
# %% Load data
df = pd.read_csv(f'{PROCESSED}/devon_surname_ranks.csv', encoding='utf-8')

pretty_dict = {
    'rank_1524': 'Total Val Rank 1524',
    'rank_1543': 'Total Val Rank 1543',
    'rank_1581': 'Total Val Rank 1581',
    'rank_1674': 'Total Val Rank 1674',
    'rank_1840': 'Total Val Rank 1840',
    'avg_val_rank_1524': 'Avg Val Rank 1524',
    'avg_val_rank_1543': 'Avg Val Rank 1543',
    'avg_val_rank_1581': 'Avg Val Rank 1581',
    'avg_val_rank_1674': 'Avg Val Rank 1674',
    'avg_val_rank_1840': 'Avg Val Rank 1840',
    'max_val_rank_1524': 'Max Val Rank 1524',
    'max_val_rank_1543': 'Max Val Rank 1543',
    'max_val_rank_1581': 'Max Val Rank 1581',
    'max_val_rank_1674': 'Max Val Rank 1674',
    'max_val_rank_1840': 'Max Val Rank 1840',
    'tot_val_pctile_1524': 'Total Val Pctile 1524',
    'tot_val_pctile_1543': 'Total Val Pctile 1543',
    'tot_val_pctile_1581': 'Total Val Pctile 1581',
    'tot_val_pctile_1674': 'Total Val Pctile 1674',
    'tot_val_pctile_1840': 'Total Val Pctile 1840',
    'avg_val_pctile_1524': 'Avg Val Pctile 1524',
    'avg_val_pctile_1543': 'Avg Val Pctile 1543',
    'avg_val_pctile_1581': 'Avg Val Pctile 1581',
    'avg_val_pctile_1674': 'Avg Val Pctile 1674',
    'avg_val_pctile_1840': 'Avg Val Pctile 1840',
    'max_val_pctile_1524': 'Max Val Pctile 1524',
    'max_val_pctile_1543': 'Max Val Pctile 1543',
    'max_val_pctile_1581': 'Max Val Pctile 1581',
    'max_val_pctile_1674': 'Max Val Pctile 1674',
    'max_val_pctile_1840': 'Max Val Pctile 1840',
    'recipient_surname': 'Recipient Surname',
    'recipient_surname_43': 'Pre-1543 Recipient',
    'recipient_surname_post_43': 'Post-1543 Recipient',
    'recipient_control_group': 'Control Surname',
    'recipient_control_group_43': 'Pre-1543 Control',
    'recipient_control_group_post_43': 'Post-1543 Control',
    'const': 'Constant',
    'recipient_interaction': 'Recipient * Rarity',
    'control_interaction': 'Control * Rarity',
}

for year in [1524, 1543]:
    df = df[df[f'count_{year}'].notnull()]
    df = df[df[f'count_{year}'] > 0]
    print(len(df))

# %% Regression time!
result_list = []
# Regression on changes in average value rank
for depvar in ['avg_val_pctile_1524_1581',
               'avg_val_pctile_1581_1674',
               'avg_val_pctile_1674_1840',
               'avg_val_pctile_1524_1840',
               'avg_val_pctile_1581_1840']:
    xvars = ['avg_val_pctile_1524', 'recipient_surname', 'recipient_control_group']
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
avg_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1581', '1581-1674', '1674-1840', '1524-1840', '1581-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(avg_val_reg)


# Regression on changes in total wealth rank
result_list = []
for depvar in ['tot_val_pctile_1524_1581',
               'tot_val_pctile_1581_1674',
               'tot_val_pctile_1674_1840',
               'tot_val_pctile_1524_1840',
               'tot_val_pctile_1581_1840']:
    xvars = ['tot_val_pctile_1524', 'recipient_surname', 'recipient_control_group']
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)

total_wealth_reg = summary_col(result_list, stars=True,
                               float_format='%0.2f',
                               model_names=['1524-1581', '1581-1674', '1674-1840', '1524-1840', '1581-1840'],
                               info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                          'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                               ).as_latex()
print(total_wealth_reg)

result_list = []
# Regression on changes in max value rank
for depvar in ['max_val_pctile_1524_1581',
               'max_val_pctile_1581_1674',
               'max_val_pctile_1674_1840',
               'max_val_pctile_1524_1840',
               'max_val_pctile_1581_1840']:
    xvars = ['max_val_pctile_1524', 'recipient_surname', 'recipient_control_group']
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
max_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1581', '1581-1674', '1674-1840', '1524-1840', '1581-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(max_val_reg)

for table, filename in zip([avg_val_reg, total_wealth_reg, max_val_reg], ['avg_val_reg', 'total_wealth_reg', 'max_val_reg']):
    table = table.replace('\\begin{center}', '\\begin{center}\n\\resizebox{\\textwidth}{!}{')
    table = table.replace('\\end{tabular}', '\\end{tabular}\n}')
    with open(f'{TABLES}/{filename}.tex', 'w', encoding='utf-8') as f:
        f.write(table)

# %% 1543 regression time!
result_list = []
# Regression on changes in average value rank
for depvar in ['avg_val_pctile_1524_1543',
               'avg_val_pctile_1543_1581',
               'avg_val_pctile_1581_1674',
               'avg_val_pctile_1674_1840',
               'avg_val_pctile_1524_1840',
               'avg_val_pctile_1543_1840']:
    xvars = ['avg_val_pctile_1524', 'recipient_surname_43', 'recipient_control_group_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_post_43'] == 1) & (reg_df['recipient_surname_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
avg_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1543-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(avg_val_reg)


# Regression on changes in total wealth rank
result_list = []
for depvar in ['tot_val_pctile_1524_1543',
               'tot_val_pctile_1543_1581',
               'tot_val_pctile_1581_1674',
               'tot_val_pctile_1674_1840',
               'tot_val_pctile_1524_1840',
               'tot_val_pctile_1543_1840']:
    xvars = ['tot_val_pctile_1524', 'recipient_surname_43', 'recipient_control_group_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_post_43'] == 1) & (reg_df['recipient_surname_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)

total_wealth_reg = summary_col(result_list, stars=True,
                               float_format='%0.2f',
                               model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1543-1840'],
                               info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                          'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                               ).as_latex()
print(total_wealth_reg)

result_list = []
# Regression on changes in max value rank
for depvar in ['max_val_pctile_1524_1543',
               'max_val_pctile_1543_1581',
               'max_val_pctile_1581_1674',
               'max_val_pctile_1674_1840',
               'max_val_pctile_1524_1840',
               'max_val_pctile_1543_1840']:
    xvars = ['max_val_pctile_1524', 'recipient_surname_43', 'recipient_control_group_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_post_43'] == 1) & (reg_df['recipient_surname_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
max_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1543-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(max_val_reg)

for table, filename in zip([avg_val_reg, total_wealth_reg, max_val_reg], ['avg_val_reg_43', 'total_wealth_reg_43', 'max_val_reg_43']):
    table = table.replace('\\begin{center}', '\\begin{center}\n\\resizebox{\\textwidth}{!}{')
    table = table.replace('\\end{tabular}', '\\end{tabular}\n}')
    with open(f'{TABLES}/{filename}.tex', 'w', encoding='utf-8') as f:
        f.write(table)

# %% post-1543 regression time!
result_list = []
# Regression on changes in average value rank
for depvar in ['avg_val_pctile_1524_1543',
               'avg_val_pctile_1543_1581',
               'avg_val_pctile_1581_1674',
               'avg_val_pctile_1674_1840',
               'avg_val_pctile_1524_1840',
               'avg_val_pctile_1581_1840']:
    xvars = ['avg_val_pctile_1543', 'recipient_surname_post_43', 'recipient_control_group_post_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_43'] == 1) & (reg_df['recipient_surname_post_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
avg_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1581-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(avg_val_reg)


# Regression on changes in total wealth rank
result_list = []
for depvar in ['tot_val_pctile_1524_1543',
               'tot_val_pctile_1543_1581',
               'tot_val_pctile_1581_1674',
               'tot_val_pctile_1674_1840',
               'tot_val_pctile_1524_1840',
               'tot_val_pctile_1581_1840']:
    xvars = ['tot_val_pctile_1543', 'recipient_surname_post_43', 'recipient_control_group_post_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_43'] == 1) & (reg_df['recipient_surname_post_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)

total_wealth_reg = summary_col(result_list, stars=True,
                               float_format='%0.2f',
                               model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1581-1840'],
                               info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                          'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                               ).as_latex()
print(total_wealth_reg)

result_list = []
# Regression on changes in max value rank
for depvar in ['max_val_pctile_1524_1543',
               'max_val_pctile_1543_1581',
               'max_val_pctile_1581_1674',
               'max_val_pctile_1674_1840',
               'max_val_pctile_1524_1840',
               'max_val_pctile_1581_1840']:
    xvars = ['max_val_pctile_1543', 'recipient_surname_post_43', 'recipient_control_group_post_43']
    reg_df = df.copy()
    reg_df = reg_df[~(reg_df['recipient_surname_43'] == 1) & (reg_df['recipient_surname_post_43'] == 0)]
    reg_df = reg_df.dropna(subset=[depvar] + xvars)

    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
max_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1524-1543', '1543-1581', '1581-1674', '1674-1840', '1524-1840', '1543-1840'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.2f}".format(x.rsquared)},
                          regressor_order=xvars
                          ).as_latex()
print(max_val_reg)

for table, filename in zip([avg_val_reg, total_wealth_reg, max_val_reg], ['avg_val_reg_post_43', 'total_wealth_reg_post_43', 'max_val_reg_post_43']):
    table = table.replace('\\begin{center}', '\\begin{center}\n\\resizebox{\\textwidth}{!}{')
    table = table.replace('\\end{tabular}', '\\end{tabular}\n}')
    with open(f'{TABLES}/{filename}.tex', 'w', encoding='utf-8') as f:
        f.write(table)

#%% Avg Val Reg With Interaction

df['recipient_interaction'] = (df['recipient_surname'] * df['count_rank_1524']) / df['count_rank_1524'].max()
df['control_interaction'] = (df['recipient_control_group'] * df['count_rank_1524']) / df['count_rank_1524'].max()


result_list = []
depvar_list = ['avg_val_pctile_1524',
    'avg_val_pctile_1581',
    'avg_val_pctile_1674',
    'avg_val_pctile_1840']
always_xvars = ['recipient_interaction', 'control_interaction']
var_names = always_xvars + depvar_list[:-1]
var_names = [pretty_dict[x] for x in var_names]

for i, depvar in enumerate(depvar_list):
    if depvar == 'avg_val_pctile_1524':
        continue
    xvars = [depvar_list[i-1]] + always_xvars
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)
    # reg_df[[depvar] + xvars] = reg_df[[depvar] + xvars].fillna(0)
    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)

avg_val_reg = summary_col(result_list, stars=True,
                            float_format='%0.2f',
                            model_names=['1581 Percentile', '1674 Percentile', '1840 Percentile'],
                            info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                         },
                            regressor_order=var_names
                            ).as_latex()
with open(f'{TABLES}/avg_val_reg_interaction.tex', 'w', encoding='utf-8') as f:
    f.write(avg_val_reg)

print(avg_val_reg)

#%% Total Wealth Reg With Interaction
result_list = []

depvar_list = ['tot_val_pctile_1524',
    'tot_val_pctile_1581',
    'tot_val_pctile_1674',
    'tot_val_pctile_1840']
always_xvars = ['recipient_interaction', 'control_interaction']
var_names = always_xvars + depvar_list[:-1]
var_names = [pretty_dict[x] for x in var_names]
for i, depvar in enumerate(depvar_list):
    if depvar == 'tot_val_pctile_1524':
        continue
    xvars = [depvar_list[i-1]] + always_xvars
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)
    # reg_df[[depvar] + xvars] = reg_df[[depvar] + xvars].fillna(0)
    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
total_wealth_reg = summary_col(result_list, stars=True,
                               float_format='%0.2f',
                               model_names=['1581 Percentile', '1674 Percentile', '1840 Percentile'],
                               info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                          },
                          regressor_order=var_names
                               ).as_latex()
with open(f'{TABLES}/total_wealth_reg_interaction.tex', 'w', encoding='utf-8') as f:
    f.write(total_wealth_reg)
print(total_wealth_reg)

#%% Max Val Reg With Interaction
result_list = []
depvar_list = ['max_val_pctile_1524',
    'max_val_pctile_1581',
    'max_val_pctile_1674',
    'max_val_pctile_1840']
always_xvars = ['recipient_interaction', 'control_interaction']
var_names = always_xvars + depvar_list[:-1]
var_names = [pretty_dict[x] for x in var_names]
for i, depvar in enumerate(depvar_list):
    if depvar == 'max_val_pctile_1524':
        continue
    xvars = [depvar_list[i-1]] + always_xvars
    reg_df = df.copy()
    reg_df = reg_df.dropna(subset=[depvar] + xvars)
    # reg_df[[depvar] + xvars] = reg_df[[depvar] + xvars].fillna(0)
    y = reg_df[depvar]
    y.rename(pretty_dict, inplace=True)
    x = reg_df[xvars]
    x = sm.add_constant(x)
    x.rename(columns=pretty_dict, inplace=True)
    # Fit the model with heteroskedasticity robust standard errors
    model = sm.OLS(y, x).fit(cov_type='HC3')
    result_list.append(model)
max_val_reg = summary_col(result_list, stars=True,
                          float_format='%0.2f',
                          model_names=['1581 Percentile', '1674 Percentile', '1840 Percentile'],
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     },
                          regressor_order=var_names
                          ).as_latex()
with open(f'{TABLES}/max_val_reg_interaction.tex', 'w', encoding='utf-8') as f:
    f.write(max_val_reg)
print(max_val_reg)
