#%%

import pandas as pd
import numpy as np
import os
import json
import platform

if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

RAW = 'Data/Raw'
PROCESSED = 'Data/Processed'

for subsidy in [
    # 1524,
    # 1543,
    # 1581,
    # 1642,
    # 1647,
    # 1660,
    1674
    ]:
    print(f'Using subsidy: {subsidy}')
    SUBSIDY_FOLDER = f'{PROCESSED}/subsidy{subsidy}'
    TEXT_FOLDER = f'{SUBSIDY_FOLDER}/text_pages'

    filename = os.path.join(SUBSIDY_FOLDER, f'subsidy{subsidy}_labels.json')
    with open(filename, 'r') as f:
        data = json.load(f)

    image_id_dict = {}
    for mini_dict in data['images']:
        image_id_dict[mini_dict['id']] = mini_dict['file_name']
    category_id_dict = {}
    for mini_dict in data['categories']:
        category_id_dict[mini_dict['id']] = mini_dict['name']
    cat_nums = {v: k for k, v in category_id_dict.items()}
    cat_nums['parish'] = 0
    cat_list = list(cat_nums.keys())
    tdf = pd.DataFrame(columns=['image_name', 'x1', 'x2', 'y1', 'y2', 'category', 'text', 'page']).astype({
        'image_name': 'object',
        'category': 'object',
        'text': 'object',
        'x1': 'float',
        'x2': 'float',
        'y1': 'float',
        'y2': 'float',
        'page': 'int'
    })
    last_page = 0
    for mini_dict in data['annotations']:
        image_id = mini_dict['image_id']
        image_name = image_id_dict[image_id]
        page_string = image_name[-7:-4]
        page = int(page_string)
        last_page = page
        x1, y1, width, height = mini_dict['bbox']
        x2 = x1 + width
        y2 = y1 + height
        category = category_id_dict[mini_dict['category_id']]
        tdf.loc[len(tdf)] = {'image_name': image_name,
                             'x1': x1,
                             'x2': x2,
                             'y1': y1,
                             'y2': y2,
                             'category': category,
                             'text': '',
                             'page': page}

    if subsidy == 1524:
        tdf = tdf[tdf['page'] != 23]

    tdf.sort_values(by=['page', 'y1'], inplace=True)
    tdf = tdf[tdf['category'] != 'add_remove']
    tdf = tdf[tdf['category'] != 'total_hearths']

#%%

    new_df = pd.DataFrame(columns=tdf.columns).astype(tdf.dtypes)
    for page in tdf.page.unique():
        indices_to_sort = [0]
        page_df = tdf[tdf['page'] == page].copy()
        page_df.reset_index(inplace=True, drop=True)
        for i in range(len(page_df)):
            if page_df.loc[i, 'category'] == 'parish':
                continue
            elif i ==0:
                indices_to_sort = [i]
            elif abs(page_df.loc[i-1, 'y1'] - page_df.loc[i, 'y1']) > 5:
                # Sort previous indices_to_sort and recombine
                top_df = page_df[:min(indices_to_sort)].copy()
                mid_df = page_df.iloc[indices_to_sort].copy()
                mid_df.sort_values(by='x1', inplace=True)
                bot_df = page_df[max(indices_to_sort)+1:].copy()
                page_df = pd.concat([top_df, mid_df, bot_df])
                page_df.reset_index(inplace=True, drop=True)
                indices_to_sort = [i]
            else:
                indices_to_sort += [i]
                if i == len(page_df) - 1:
                    top_df = page_df[:min(indices_to_sort)].copy()
                    mid_df = page_df.iloc[indices_to_sort].copy()
                    mid_df.sort_values(by='x1', inplace=True)
                    bot_df = page_df[max(indices_to_sort) + 1:].copy()
                    page_df = pd.concat([top_df, mid_df, bot_df])
                    page_df.reset_index(inplace=True, drop=True)

        new_df = pd.concat([new_df, page_df])
    new_df.reset_index(inplace=True, drop=True)

    #%% Number each of the notes on each page
    note = 0
    page = 0
    for i, row in new_df.iterrows():
        new_page = row['page']
        if new_page != page:
            note = 0
        note += 1
        new_df.loc[i, 'note'] = note
        page = new_page

    new_df['note'] = new_df['note'].astype(int)
    new_df['subisdy'] = subsidy


    #%%
    if os.path.isdir(TEXT_FOLDER):
        for file in os.listdir(TEXT_FOLDER):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(TEXT_FOLDER, file), 'r', encoding='utf-8') as f:
                text = f.read()
            text_list = text.split('\n\n')
            page = int(file[-7:-4])
            new_df.loc[new_df['page'] == page, 'text'] = text_list
    new_df = new_df[['subisdy', 'image_name', 'x1', 'x2', 'y1', 'y2', 'category', 'page', 'note', 'text']]
    new_df.to_csv(os.path.join(SUBSIDY_FOLDER, f'subsidy{subsidy}_sections.csv'), index=False)
    print(f'{subsidy} csv saved!')


