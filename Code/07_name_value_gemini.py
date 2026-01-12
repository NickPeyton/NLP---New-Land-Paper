import os
import shutil
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()
import json
import google.generativeai as genai
import time
import nltk
import typing_extensions as typing
import platform
import matplotlib.pyplot as plt
import cv2
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

#%% Configurating my little Gemini
genai_model = genai.GenerativeModel('gemini-2.0-flash')
genai.configure(api_key='AIzaSyBfl8mrxUCsSHpOV8ToJWL9Po8fH0fshJY')

class NameList(typing.TypedDict):
    surnames: list[str]

class ValueList(typing.TypedDict):
    values: list[float]

name_config = genai.GenerationConfig(response_mime_type='application/json',
                                        response_schema=NameList)
value_config = genai.GenerationConfig(response_mime_type='application/json',
                                        response_schema=ValueList)

#%% Filling in all forenames, surnames, types, and values with agreement between readers
for subsidy in [
    1524,
    1543,
    1581,
    1642,
    1647,
    1660,
    # 1661,
    1674,
     ]:
    SUBSIDY_FOLDER = f'Data/Processed/subsidy{subsidy}'
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv')

    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        # All the different things we need to get agreement on
        for thing in ['forename', 'surname', 'type', 'title', 'value']:
            if f'gemini_{thing}' not in tdf.columns:
                if thing == 'value':
                    tdf[f'gemini_{thing}'] = np.nan
                else:
                    tdf[f'gemini_{thing}'] = ''
            if row[f'gemini_{thing}'] != '' and not pd.isna(row[f'gemini_{thing}']):
                continue
            # The non-trocr readers actually can't fucking read
            if thing == 'value' and row['tr_value'] % 1 != 0:
                tdf.at[i, f'gemini_{thing}'] = row['tr_value']
                continue
            thing_list = []
            for reader in ['tr', 'eo', 'pt']:
                thing_list.append(row[f'{reader}_{thing}'])
            thing_list = [x for x in thing_list if x != '']
            thing_list = [x for x in thing_list if not pd.isna(x)]
            thing_set = set(thing_list)
            # Unanimity
            if len(thing_set) == 1:
                tdf.at[i, f'gemini_{thing}'] = thing_list[0]
            # Majority vote
            elif len(thing_set) == 2 and len(thing_list) == 3:
                if thing_list[0] == thing_list[1]:
                    tdf.at[i, f'gemini_{thing}'] = thing_list[0]
                else:
                    tdf.at[i, f'gemini_{thing}'] = thing_list[2]
            # No agreement, leaving blank for little gemini
            else:
                if thing == 'value':
                    tdf.at[i, f'gemini_{thing}'] = np.nan
                elif thing == 'type':
                    tdf.at[i, f'gemini_{thing}'] = 'G'
                else:
                    tdf.at[i, f'gemini_{thing}'] = ''
    tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)

#%% Getting surnames and values from Gemini
for subsidy in [
    # 1524,
    1543,
    1581,
    1642,
    1647,
    1660,
    # 1661,
    1674,
     ]:
    SUBSIDY_FOLDER = f'Data/Processed/subsidy{subsidy}'
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv')

    surname_options_list = []
    surname_images_list = []
    surname_indices_list = []

    value_options_list = []
    value_images_list = []
    value_indices_list = []

    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        if pd.isna(row['gemini_surname']) or row['gemini_surname'] == '':
            surname_options_list.append([row['tr_surname'], row['eo_surname'], row['pt_surname']])
            page_image = cv2.imread(f'{SUBSIDY_FOLDER}/processed_pages/{row['image_name'].replace('little_', 'processed_')}')
            surname_image = page_image[row['y1']:row['y2'], row['x1']:row['x2']]
            surname_images_list.append(surname_image)
            surname_indices_list.append(i)
        if pd.isna(row['gemini_value']) or row['gemini_value'] == '':
            value_options_list.append([row['tr_value'], row['eo_value'], row['pt_value']])
            page_image = cv2.imread(f'{SUBSIDY_FOLDER}/processed_pages/{row['image_name'].replace('little_', 'processed_')}')
            value_image = page_image[row['y1']:row['y2'], row['x1']:row['x2']]
            value_images_list.append(value_image)
            value_indices_list.append(i)
        for j, sub_list in enumerate([surname_options_list, value_options_list]):
            if len(sub_list) < 20 and i != len(tdf) - 1:
                continue
            if j == 0:
                image_list = surname_images_list
                index_list = surname_indices_list
                thing = 'surname'
                config = name_config
                print('Getting surnames...')
            else:
                image_list = value_images_list
                index_list = value_indices_list
                thing = 'value'
                config = value_config
                print('Getting values...')

            LINE_FOLDER = f'{SUBSIDY_FOLDER}/{thing}_lines'
            os.makedirs(LINE_FOLDER, exist_ok=True)
            prompt_list = []
            for k, image in enumerate(image_list):
                image_path = f'{LINE_FOLDER}/{thing}_{k}.png'
                cv2.imwrite(image_path, image)
                uploaded_image = genai.upload_file(image_path)
                prompt_list.append(uploaded_image)
            gemini_prompt = f'''
            Hello Gemini, I've uploaded some lines from a typed-up tax document. 
            Here are three possible {thing}s from each line image.
            I need you to SELECT the correct {thing} based on the choices and the image provided.
            If there is no correct {thing}, please output \'NO_SURNAME\' for a surname and \'NO_VALUE\' for a value.
            If there are multiple {thing}s in the image, please output only one {thing}, the bottom one if possible.
            Please output a JSON file with the selected {thing} for each image. 
            Make sure the list is exactly {len(image_list)} {thing}s long. Thanks!
            '''
            prompt_list.append(gemini_prompt)
            sleepytime = 4
            no_response = True
            while no_response:
                time.sleep(sleepytime)
                try:
                    print('Asking Gemini...')
                    response = genai_model.generate_content(prompt_list,
                                                            generation_config=config)
                    print('Got a response!')
                    response = json.loads(response.text)
                    correct_thing_list = response[f'{thing}s']
                    for k, correct_thing in enumerate(correct_thing_list):
                        tdf.at[index_list[k], f'gemini_{thing}'] = correct_thing
                    print(f'Correct {thing}s added to the dataframe!')
                    if j == 0:
                        surname_options_list = []
                        surname_images_list = []
                        surname_indices_list = []
                    else:
                        value_options_list = []
                        value_images_list = []
                        value_indices_list = []
                    no_response = False
                except Exception as e:
                    sleepytime *= 2
                    if sleepytime >= 1024:
                        print('Gemini is being shitty, I give up')
                        tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)
                        raise

    tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)

#%%

tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)