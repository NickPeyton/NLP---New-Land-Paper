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
with open(f'{PROCESSED}/pretty_dict.json', 'r', encoding='utf-8') as f:
    pretty_dict = json.load(f)
# %% Load data
df = pd.read_csv(f'{PROCESSED}/hundred_dataset.csv', encoding='utf-8')
df['lland'] = np.log(df['landOwned'] + 1)
df['monast_land_share'] = (df['landOwned']/240)/(df['hundred_val_1524'] + df['landOwned']/240)

# Set np.nan if ind_1831 == agr_1831 == 0
df.loc[(df['ind_1831'] == 0) & (df['agr_1831'] == 0), 'ind_1831'] = np.nan
df.loc[(df['ind_1831'] == 0) & (df['agr_1831'] == 0), 'agr_1831'] = np.nan

# %% Regression time!
for year in [1524, 1543, 1581, 1674, 1840]:
    for measure in ['count', 'tot_val']:
        result_list = []
        # Regression on changes in average value rank
        for depvar in ['avg_val_rank_1524_1581_corr',
                       'avg_val_rank_1581_1674_corr',
                       'avg_val_rank_1674_1840_corr',
                       'avg_val_rank_1524_1840_corr',
                       'avg_val_rank_1581_1840_corr']:
            xvars = ['monast_land_share', f'recipient_{measure}_share_{year}', f'control_{measure}_share_{year}', 'mean_slope', 'wheatsuit', 'area', 'lspc1525', 'distmkt']
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
                                  info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))},
                                  regressor_order=[pretty_dict[x] for x in xvars]
                                  ).as_latex()
        result_list = []
        for depvar in ['tot_val_rank_1524_1581_corr',
                        'tot_val_rank_1581_1674_corr',
                        'tot_val_rank_1674_1840_corr',
                       'tot_val_rank_1524_1840_corr',
                        'tot_val_rank_1581_1840_corr'
                       ]:
            xvars = ['monast_land_share', f'recipient_{measure}_share_{year}', f'control_{measure}_share_{year}', 'mean_slope', 'wheatsuit', 'area', 'lspc1525', 'distmkt']
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
                                  info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))},
                                  regressor_order=[pretty_dict[x] for x in xvars]
                                  ).as_latex()
        result_list = []
        for depvar in ['max_val_rank_1524_1581_corr',
                        'max_val_rank_1581_1674_corr',
                        'max_val_rank_1674_1840_corr',
                       'max_val_rank_1524_1840_corr',
                       'max_val_rank_1581_1840_corr']:
            xvars = ['monast_land_share', f'recipient_{measure}_share_{year}', f'control_{measure}_share_{year}', 'mean_slope', 'wheatsuit', 'area', 'lspc1525', 'distmkt']
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
                                    info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))},
                                    regressor_order=[pretty_dict[x] for x in xvars]
                                    ).as_latex()



        for table, filename in zip([avg_val_reg, total_wealth_reg, max_val_reg], ['avg_val_reg', 'total_wealth_reg', 'max_val_reg']):
            table = table.replace('<', '$<$').replace('>', '$>$').replace('\\label{}', '\\label{hundred_' + measure + '_' + str(year) + '}')
            table = table.replace('\\begin{center}', '\\begin{center}\n\\resizebox{\\textwidth}{!}{')
            table = table.replace('\\end{tabular}', '\\end{tabular}\n}')
            table = table.replace('\\begin{table}', '\\begin{table}[H]')
            table = table.replace('{llllll}', '{llll|l|l}')
            with open(f'{TABLES}/{filename}_hundred_{measure}_{year}.tex', 'w', encoding='utf-8') as f:
                f.write(table)
            print(table)
        
        #%% Regression on economic indicators
        result_list = []
        for depvar in ['industrial_share', 'agriculture_share', 'other_share']:
            xvars = ['monast_land_share', f'recipient_{measure}_share_{year}', f'control_{measure}_share_{year}', 'mean_slope', 'wheatsuit', 'area', 'lspc1525',
                     'distmkt']
            reg_df = df.copy()
            reg_df = reg_df.dropna(subset=['industrial_share', 'agriculture_share', 'other_share'] + xvars)
        
            y = reg_df[depvar]
            y.rename(pretty_dict, inplace=True)
            x = reg_df[xvars]
            x = sm.add_constant(x)
            x.rename(columns=pretty_dict, inplace=True)
            # Fit the model with heteroskedasticity robust standard errors
            model = sm.OLS(y, x).fit(cov_type='HC3')
            result_list.append(model)
        economic_reg = summary_col(result_list, stars=True,
                                   float_format='%0.2f',
                                   model_names=['Industry Share', 'Agriculture Share', 'Other Share'],
                                   info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))},
                                   regressor_order=[pretty_dict[x] for x in xvars]
                                   ).as_latex()
        economic_reg = economic_reg.replace(
            '\\end{center}\n\\end{table}\n\\bigskip\nStandard errors in parentheses. \\newline \n* p<.1, ** p<.05, ***p<.01',
            '\\end{center}\n* p<.1, ** p<.05, ***p<.01\n\\end{table}')
        economic_reg = economic_reg.replace('<', '$<$').replace('>', '$>$').replace('\\label{}', '\\label{hundred_' + measure + '_' + str(year) + '}')
        economic_reg = economic_reg.replace('\\begin{center}', '\\begin{center}\n\\resizebox{\\textwidth}{!}{')
        economic_reg = economic_reg.replace('\\end{tabular}', '\\end{tabular}\n}')
        economic_reg = economic_reg.replace('\\begin{table}', '\\begin{table}[H]')
        for table, filename in zip([economic_reg], ['economic_reg']):
            with open(f'{TABLES}/{filename}_hundred_{measure}_{year}.tex', 'w', encoding='utf-8') as f:
                f.write(table)
            print(table)