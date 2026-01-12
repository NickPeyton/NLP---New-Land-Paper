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
class ParishList(typing.TypedDict):
    parishes: list[str]
parish_config = genai.GenerationConfig(response_mime_type='application/json',
                                       response_schema=ParishList)

#%% Getting parish list
phdf = pd.read_csv('Data/Raw/parishes_hundreds.csv', encoding='utf-8')

sub_parish_list = ['tithing', 'hamlet',]
parish_name_correction = {
    'Sherwill': 'Shirwell',
    'South Sydenham': 'Sydenham Damerel',
    'Tidcombe': 'Tiverton',
    'Tiverton All Fours': 'Tiverton',
    'Tiverton Clare': 'Tiverton',
    'Tiverton Priors': 'Tiverton',
    'Uplowman': 'Uplowman (T)',
    'Virginstowe': 'Virginstow',
    'Weare Gifford': 'Weare Giffard',
    'Widecombe': 'Widecombe In The Moor',
    'Townstall': 'Dartmouth Townstall',
    'Loxbear': 'Loxbeare',
    'St Marychurch': 'Torquay St Marychurch',
    'Marystow': 'Marystowe',
    'Milton Damarel': 'Milton Damerel',
    'Morley': 'Moreleigh',
    'Northlew': 'North Lew',
    'Hawkchurch': 'Axminster',
    'Pancraswike': 'Pancrasweek',
    'Plymouth Charles': 'Charles The Martyr',
    'St. Budeaux (Plymouth)': 'St Budeaux',
    'Down Saint Mary': 'Down St Mary',
    'Teignmouth East': 'East Teignmouth',
    'Teignmouth West': 'West Teignmouth',
    'Exeter Holy Trinity': 'Holy Trinity',
    'Exeter Saint David': 'St David',
    'Exeter St David': 'St David',
    'Exeter Saint Edmund': 'St Edmund',
    'Exeter Saint George': 'St George',
    'Exeter Saint John': 'St John',
    'Exeter Saint Kerrian': 'St Kerrian',
    'Exeter Saint Lawrence': 'St Lawrence',
    'Exeter Saint Leonard': 'St Leonard',
    'Exeter St Leonard': 'St Leonard',
    'Exeter Saint Martin': 'St Martin',
    'Exeter Saint Mary Arches': 'St Mary Arches',
    'Exeter Saint Mary Major': 'St Mary Major',
    'Exeter Saint Mary Steps': 'St Mary Steps',
    'Exeter St Mary Steps': 'St Mary Steps',
    'Exeter Saint Olave': 'St Olave',
    'Exeter Saint Pancras': 'St Pancras',
    'Exeter Saint Paul': 'St Paul',
    'Exeter Saint Petrock': 'St Petrock',
    'Exeter Saint Sidwell': 'St Sidwell',
    'Exeter St Sidwell': 'St Sidwell',
    'Exeter Saint Stephen': 'St Stephen',
    'Exeter Saint Thomas the Apostle': 'St Thomas The Apostle',
    'Exeter St Thomas': 'St Thomas The Apostle',
    'Idford': 'Ideford',
    'Lewtrenchard': 'Lew Trenchard',
    'Littleham (Bideford)': 'Littleham',
    'Compton Gifford': 'Charles The Martyr',
    'Weston Peverell': 'Pennycross',
    'Plymouth St. Andrew': 'St Andrew',
    'Salcombe Regis Chilston': 'Salcombe Regis',
    'Shillingford': 'Shillingford St George',
    'Allington, East': 'East Allington',
    'Bickleigh': 'Bickleigh (Near Tiverton)',
    'Bow otherwise Nymet Tracey': 'Bow',
    'Bridgerule East': 'Bridgerule',
    'Bridgerule West': 'Bridgerule',
    'Broadnymet': 'Broad Nymet',
    'Burlescombe': 'Burlescombe (Bampton)',
    'Chardstock': 'Yarcombe',
    'Churchshtow': 'Churchstow',
    'Bystock': 'Withycombe Raleigh',
    'Seaton': 'Seaton And Beer',
    'Bridestow': 'Bridestowe',
    ' ' : '',
    'Broad Clyst': 'Broadclyst',
    'Edbury Tithing': 'Cruwys Morchard',
    'Woolfardisworthy': 'Woolfardisworthy (West)',
    'Witheridge Hundred': 'Bishops Nympton',
    'Petrockstow': 'Petrockstowe',
    'Willeworthy Hamlet': 'Peter Tavy',
    'Ford': 'Winkleigh',
    'Losebeer Tithing': 'Winkleigh',
    'Corneworthy': 'Cornworthy',
    'Buckland Toutsaints': 'Buckland Tout Saints',
    'Milton Damerell': 'Milton Damerel',
    'Asheton': 'Ashton',
    'Colbrooke': 'Colebrooke',
    'Exeter St Leonards': 'St Leonard',
    'Brydford': 'Bridford',
    'Alphyngton': 'Alphington',
    'Colompton': 'Cullompton',
    'Sydbury': 'Sidbury',
    'Widdicombe': 'Widecombe In The Moor',
    'Paynton': 'Paignton',
    'Kigbeare': 'Okehampton',
    'Monckeokehampton': 'Monkokehampton',
    'Crecombe': 'Creacombe',
    'Twychen': 'Twitchen',
    'Huyshe': 'Huish',
    'Weare Gifforde': 'Weare Giffard',
    'Bideforde': 'Bideford',
    'Shepewashe': 'Sheepwash',
    'Frithelstocke': 'Frithelstock',
    'Torrington Magna': 'Great Torrington',
    'Newton Tracye': 'Newton Tracey',
    'Tawstocke': 'Tawstock',
    'Eastdowne': 'East Down',
    'Ashforde': 'Ashford',
    'Combmartine': 'Combe Martin',
    'Sampford Spyny': 'Sampford Spiney',
    'Plympton Morrice': 'Plympton St Maurice',
    'Plympton Mary': 'Plympton St Mary',
    'Plympton Marye': 'Plympton St Mary',
    'Maristowe': 'Marystowe',
    'Milton Abbott': 'Milton Abbot',
    'Townstal': 'Dartmouth Townstall',
    'Stekylpth Hamlet': 'Sampford Courtenay',
    'Thornebury': 'Thornbury',
    'Kakebeare Hamlet': 'Okehampton',
    'Newton Tracy': 'Newton Tracey',
    'Morthoe': 'Mortehoe',
    'Oxbeare': 'Loxbeare',
    'Doddiscombleigh': 'Doddiscombsleigh',
    'Galmpton In Churston Ferrers': 'Churston Ferrers',
    'Townstal In Dartmouth': 'Dartmouth Townstall',
    'Botterford Tithing': 'North Huish',
    'Dunstone In Yealmpton': 'Yealmpton',
    'Shaugh': 'Shaugh Prior',
    'Parcel Of Sheave': 'Shaugh Prior',
    'Stonehouse': 'East Stonehouse',
    'Weston Peverel': 'Pennycross',
    'Stoke Damarel': 'Stoke Damerel',
    'Willisworthy Tithing': 'Peter Tavy',
    'Monk Okehampton': 'Monkokehampton',
    'The Hamlet Of Cakebar (Kigbeare In Okehampton)': 'Okehampton',
    'The Hamlytt Of Ye Same (Northcott)': 'Boyton',
    'Bow Or Nymet Tracy': 'Bow',
    'St Thomas the Apostle': 'St Thomas The Apostle',
    'Littleham (near Exmouth)': 'Littleham (Near Exmouth)',
    'Loosebeer Tithing': 'Winkleigh',
}
#%% Regexes

parish_pattern = re.compile(r'([a-z\s]*)\sparish\s?[\[(]?([a-z\s+]+)?(\d+)?[\])]?', re.IGNORECASE)
sub_parish_pattern = re.compile(r'[(\[]in ([A-Z][a-z]+\s?[A-Z]?[a-z]*)[A-Za-z\s]*[)\]]')

#%% Picking Parishes
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
    doc = subsidy
    if subsidy in [1642, 1647, 1660]:
        doc = 1581
    # Creating the page-parish dictionary
    page_parish_dict = {}
    for i, row in phdf.iterrows():
        page_list = ast.literal_eval(row[f'page_list_{doc}'])
        if len(page_list) == 0:
            continue
        for num in page_list:
            if num not in page_parish_dict.keys():
                page_parish_dict[num] = [row['parish'].title()]
            else:
                page_parish_dict[num] += [row['parish'].title()]
    page_parish_dict = dict(sorted(page_parish_dict.items()))

    tdf = pd.read_csv(f'Data/Processed/subsidy{subsidy}/subsidy{subsidy}_lines.csv')
    if 'gemini_parish' not in tdf.columns:
        tdf['gemini_parish'] = ''

    # Correcting for the lack of parish readings in parish rows
    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        if row['category'] == 'parish':
            tdf.at[i, 'tr_parish'] = tdf.iloc[i+1]['tr_parish']
            tdf.at[i, 'eo_parish'] = tdf.iloc[i+1]['eo_parish']
            tdf.at[i, 'pt_parish'] = tdf.iloc[i+1]['pt_parish']

    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        # Interruptible thank god
        if (row['gemini_parish'] != '' and not pd.isna(row['gemini_parish'])) or row['pt_parish'] == tdf.iloc[i-1]['pt_parish']:
            continue
        # Getting potential parishes for the page
        page = row['page']
        if page in page_parish_dict.keys():
            possible_parishes = page_parish_dict[page]
        else:
            # This should never happen
            possible_parishes = []
            print('the FUCK')
            print(f'{subsidy}, page {page}')
        # If there's only one possible parish, it's that one
        if len(possible_parishes) == 1:
            tdf.loc[tdf['pt_parish'] == row['pt_parish'], 'gemini_parish'] = possible_parishes[0]
            continue
        # Grabbing parish names from parish texts
        parish_text_list = [row['tr_parish'], row['eo_parish'], row['pt_parish']]
        # Replacing all nans in parish_text_list
        parish_text_list = [x if pd.notna(x) else '' for x in parish_text_list]
        parish_list = []
        for parish_text in parish_text_list:
            search_result = parish_pattern.search(parish_text)
            # If no search result, we try for sub-parish level
            if not search_result:
                if any([x in parish_text.lower() for x in sub_parish_list]):
                    sub_parish_search_result = re.search(sub_parish_pattern, parish_text)
                    if sub_parish_search_result is None:
                        parish_list.append(parish_text)
                    else:
                        parish_list.append(sub_parish_search_result.group(1).title())
                else:
                    parish_list.append(parish_text)
            elif type(search_result.group(1)) is str and search_result.group(2) is None:
                parish_list.append(search_result.group(1).title())
            elif type(search_result.group(1)) is str and type(search_result.group(2)) is str:
                parish_list.append(search_result.group(2).title())
            else:
                print('There shouldn\'t be an "else"!!!')
        parish_set = set(parish_list)
        # Checking for overlap in parishes generated by OCR and regex
        if len(parish_set) == 1:
            if list(parish_set)[0] in possible_parishes:
                parish_candidate = list(parish_set)[0]
                distance_dict = {}
                for parish in possible_parishes:
                    distance_dict[parish] = nltk.edit_distance(parish, parish_candidate)
                parish = min(distance_dict, key=distance_dict.get)
                tdf.loc[tdf['pt_parish'] == row['pt_parish'], 'gemini_parish'] = parish
                continue
        # Choosing the closest parish to the most popular option from the readers
        elif len(parish_set) == 2:
            if parish_list[0] == parish_list[1]:
                parish_candidate = parish_list[0]
                distance_dict = {}
                for parish in possible_parishes:
                    distance_dict[parish] = nltk.edit_distance(parish, parish_candidate)
                parish = min(distance_dict, key=distance_dict.get)
                tdf.loc[tdf['pt_parish'] == row['pt_parish'], 'gemini_parish'] = parish
                continue
            else:
                parish_candidate = parish_list[2]
                distance_dict = {}
                for parish in possible_parishes:
                    distance_dict[parish] = nltk.edit_distance(parish, parish_candidate)
                parish = min(distance_dict, key=distance_dict.get)
                tdf.loc[tdf['pt_parish'] == row['pt_parish'], 'gemini_parish'] = parish
                continue
        else:
            assert len(parish_set) == 3
            edit_distance_dict = {}
            for parish in possible_parishes:
                edit_distance_dict[parish] = 0
                for par in parish_set:
                    edit_distance_dict[parish] += nltk.edit_distance(parish, par)

            parish_candidate = min(edit_distance_dict, key=edit_distance_dict.get)
            if edit_distance_dict[parish_candidate] < 10:
                tdf.loc[tdf['pt_parish'] == row['pt_parish'], 'gemini_parish'] = parish_candidate
                continue

    tdf.to_csv(f'Data/Processed/subsidy{subsidy}/subsidy{subsidy}_lines.csv', index=False)

#%% Going over the blanks, asking Gemini for help
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
    doc = subsidy
    if subsidy in [1642, 1647, 1660]:
        doc = 1581
    # Creating the page-parish dictionary
    page_parish_dict = {}
    for i, row in phdf.iterrows():
        page_list = ast.literal_eval(row[f'page_list_{doc}'])
        if len(page_list) == 0:
            continue
        for num in page_list:
            if num not in page_parish_dict.keys():
                page_parish_dict[num] = [row['parish'].title()]
            else:
                page_parish_dict[num] += [row['parish'].title()]
    page_parish_dict = dict(sorted(page_parish_dict.items()))
    all_parish_list = phdf['parish'].tolist()
    SUBSIDY_FOLDER = f'Data/Processed/subsidy{subsidy}'
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_lines.csv')
    tricky_line_files = []
    reader_choices_list = []
    index_choices_list = []
    parishes_to_replace = []
    last_parish = False
    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        if row['pt_parish'] in parishes_to_replace and row['pt_parish'] != tdf.iloc[(len(tdf) - 1)]['pt_parish']:
            continue
        if row['pt_parish'] == tdf.iloc[(len(tdf) - 1)]['pt_parish']:
            last_parish = True
        # Interruptible thank god
        if row['gemini_parish'] != '' and not pd.isna(row['gemini_parish']):
            continue
        # Getting potential parishes for the page
        page = row['page']
        if page in page_parish_dict.keys():
            possible_parishes = page_parish_dict[page]
        else:
            # This should never happen
            possible_parishes = []
            print('the FUCK')
            print(page)

        reader_parish_list = [row['tr_parish'], row['eo_parish'], row['pt_parish']]
        page_string = str(page)
        while len(page_string) < 3:
            page_string = '0' + page_string
        page_image = cv2.imread(f'{SUBSIDY_FOLDER}/processed_pages/processed_subsidy{subsidy}_page_{page_string}.png', cv2.IMREAD_GRAYSCALE)
        line_image = page_image[row['y1']:row['y2'], row['x1']:row['x2']]
        os.makedirs(f'{SUBSIDY_FOLDER}/tricky_lines', exist_ok=True)
        cv2.imwrite(f'{SUBSIDY_FOLDER}/tricky_lines/line_{i}.png', line_image)
        tricky_line_files.append(f'{SUBSIDY_FOLDER}/tricky_lines/line_{i}.png')
        reader_choices_list.append(reader_parish_list)
        index_choices_list.append(possible_parishes)
        parishes_to_replace.append(row['pt_parish'])


        if len(tricky_line_files) < 20 and not last_parish:
            continue

        no_response = True
        sleepytime = 8
        prompt_list = []
        print('Uploading files to Gemini:')
        for line_file in tqdm(tricky_line_files):
            uploaded_file = genai.upload_file(line_file)
            prompt_list.append(uploaded_file)
        gemini_prompt = f'''Hello Gemini, I've uploaded some images of parish headings from a typed-up tax document. 
        Here are three possible parish names for each image, in the same order as the images:
        {reader_choices_list}
        Here are the lists of possible parish names for each image taken from the document's index, again in the same order as the images:
        {index_choices_list}
        Please SELECT the correct parish name for each image from its list of possible parish names.
        If the text in the image truly does not match any of the possible parish names, please select it from this list of all parishes in Devon: {all_parish_list}.
        Do not output any other text besides the SELECTED parish name.
        Please output a .json file of the selected parish names in the same order as the images.
        Capitalize each word in each parish name and do not include the word "parish" in the name.
        '''

        prompt_list.append(gemini_prompt)
        while no_response:
            try:
                print('Beseeching Gemini')
                gemini_response = genai_model.generate_content(prompt_list,
                                                               generation_config=parish_config
                                                               )
                print('GEMINI HATH RESPONDED, REJOICE')
                correct_parish_list = json.loads(gemini_response.text)['parishes']

                for j, correct_parish in enumerate(correct_parish_list):
                    tdf.loc[tdf['pt_parish'] == parishes_to_replace[j], 'gemini_parish'] = correct_parish
                no_response = False
                tricky_line_files = []
                reader_choices_list = []
                index_choices_list = []
                parishes_to_replace = []
            except:
                print(f'Waiting {sleepytime} seconds')
                time.sleep(sleepytime)
                sleepytime *= 2
                if sleepytime > 1024:
                    tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_lines.csv', index=False)
                    print('Gemini is unavailable, woe unto thee')
                    break
    tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_lines.csv', index=False)
    shutil.rmtree(f'{SUBSIDY_FOLDER}/tricky_lines')