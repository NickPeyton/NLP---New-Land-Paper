#%%
import os

from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import cv2
tqdm.pandas()
import re
import platform
import pandas as pd
import matplotlib.pyplot as plt
import time
import pytesseract as pt
import shutil
import scipy.signal as signal

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
    #%%
    print(f'Using subsidy: {subsidy}')
    SUBSIDY_FOLDER = f'{PROCESSED}/subsidy{subsidy}'
    TEXT_FOLDER = f'{SUBSIDY_FOLDER}/text_pages'
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_sections.csv')

    #%% Splitting sub-images
    new_df = pd.DataFrame(columns=['image_name', 'x1', 'x2', 'y1', 'y2', 'category', 'text', 'page', 'note', 'line']).astype({
        'image_name': 'object',
        'category': 'object',
        'text': 'object',
        'x1': 'float',
        'x2': 'float',
        'y1': 'float',
        'y2': 'float',
        'page': 'int',
        'note': 'int',
        'line': 'int'
    })

    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        file = row['image_name'].replace('little_', 'processed_')
        category = row['category']
        page = row['page']

        if 'parish' in category:
            new_row = row
            new_row[['x1', 'x2', 'y1', 'y2']] = row[['x1', 'x2', 'y1', 'y2']] * 10
            new_row[['x1', 'x2', 'y1', 'y2']] = new_row[['x1', 'x2', 'y1', 'y2']].astype(int)
            new_row['line'] = 1
            new_df.loc[len(new_df)] = new_row
            continue

        image = cv2.imread(f'{SUBSIDY_FOLDER}/processed_pages/{file}', cv2.IMREAD_GRAYSCALE)
        x1 = int(row['x1']*10)
        x2 = int(row['x2']*10)
        y1 = int(row['y1']*10)
        y2 = int(row['y2']*10)
        image_array = image[y1:y2, x1:x2]
        cropped_image_array = image_array[:, :-500]
        if row['text'] != '' and not pd.isna(row['text']):
            text_list = row['text'].split('\n')
        else:
            text_list = []
        horiz_proj = np.sum(cropped_image_array, axis=1)
        projection_series = pd.Series(horiz_proj)
        projection_rolling = projection_series.rolling(window=30, center=True)
        projection_smoothed = projection_rolling.mean()
        peaks, properties = signal.find_peaks(x=projection_smoothed, distance=50, prominence=2000)
        if len(peaks) > len(text_list) - 1 and len(text_list) > 1:
            peaks = np.delete(peaks, np.argmin(properties['prominences']))
            print(f'Double-check page {row['page']}, file {file}')

        split_list = [0] + list(peaks) + [cropped_image_array.shape[0]]
        if len(text_list) == 0:
            text_list = [''] * (len(split_list) - 1)
        while len(split_list) < len(text_list) + 1:
            diff_list = [split_list[x+1] - split_list[x] for x in range(len(split_list) - 1)]
            biggest_diff = max(diff_list[:-1])
            new_split = int(split_list[np.argmax(diff_list)] + biggest_diff/2)
            split_list.insert(np.argmax(diff_list) + 1, new_split)
        assert len(split_list) == len(text_list) + 1
        for j in range(len(split_list) - 1):
            top = split_list[j]
            bottom = split_list[j + 1]
            line_height = bottom - top
            if line_height < 80:
                gap = 80 - line_height
                top = int(max(0, top - (3 * gap // 4)))
                bottom = int(min(cropped_image_array.shape[0], bottom + gap // 4))
            new_row = row
            new_row['y1'] = (y1 + top)
            new_row['y2'] = (y1 + bottom)
            new_row['x1'] = x1
            new_row['x2'] = x2
            new_row['line'] = j + 1
            new_row['text'] = text_list[j]
            new_row['subsidy'] = subsidy
            new_df.loc[len(new_df)] = new_row

    new_df.to_csv(os.path.join(SUBSIDY_FOLDER, f'subsidy{subsidy}_lines.csv'), index=False)
