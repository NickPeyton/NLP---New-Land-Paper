import os
import re
import platform
import numpy as np
import pandas as pd
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

#%% REGEX MANIA

forename_pattern = re.compile(r'^([A-Z][a-z]+)')
surname_pattern = re.compile(r'^\b(?:at |atte |de |de la )?([A-Z][a-z]+)\b')

fraction = re.compile(r'\s*(\d)⁄(\d)$')
number_at_end = re.compile(r'(\d+)$')

missing_last_name = re.compile(r'([a-zA-Z0-9]+)([._\-\s]+)(\d+)')
missing_first_name = re.compile(r'([._\-\s]+)([A-Z][a-z]+)\s+(\d+)')
missing_both = re.compile(r'([._\-]+)\s*(\d+)')

shit_at_end = re.compile(r'([\s\w._\-]+\d+)([._\-]+)$')
number_at_start = re.compile(r'^(\d+)')

other_assessment = re.compile(r'(\(\s?[WwGCcLl]\s?\d+\s?\\?\.?-?\d*d?\\?\.?\s?\)?)')

missing_space = re.compile(r'([a-z])([A-Z0-9])')

misidentified_number_dict = {
    re.compile(r'(I)$'): '1',
    re.compile(r'(l)$'): '1',
    re.compile(r'(L)$'): '1',
    re.compile(r'(G)$'): '6',
    re.compile(r'(B)$'): '8',
    re.compile(r'(S)$'): '5',
    re.compile(r'(O)$'): '0',
    re.compile(r'(Z)$'): '2',
    re.compile(r'(A)$'): '4',
}
type_dict = {re.compile('W$'): 'W',
             re.compile('w$'): 'W',
             re.compile('G$'): 'G',
             re.compile('C$'): 'G',
             re.compile('c$'): 'G',
             re.compile('L$'): 'L',
             re.compile('l$'): 'L'}

damage_words = ['mutilat', ' damage', ' lost', '']

title_list = [re.compile(r'([Ss]en)$'),
              re.compile(r'([Jj]un)$'),
              re.compile(r'([Ww]id)$'),
              re.compile(r'([Aa]rmiger)$'),
              re.compile(r'([Aa]r)$'),
              re.compile(r'([Kk]t)$'),
              re.compile(r'([Gg]t)$'),
              re.compile(r'([Gg]ent)$')]

alien_list = [' frenchman ', ' german ', ' norman ', ' alien ']

name_abbrev_dict = {'Thos': 'Thomas',
                    'Tho': 'Thomas',
                    'Rd': 'Richard',
                    'Jon': 'John',
                    'Jn': 'John',
                    'Wm': 'William',
                    'Willm': 'William',
                    'Robt': 'Robert',
                    'Nich': 'Nicholas',
                    'Edw': 'Edward',
                    'Geo': 'George',
                    'Xtopher': 'Christopher',
                    'Xpher': 'Christopher',
                    'Xtofer': 'Christopher',
                    'Xian': 'Christian',
                    'Xtian': 'Christian',
                    'Math': 'Matthew',
                    'Hen': 'Henry',
                    'Richd': 'Richard',
                    }

fully_parenthetical = re.compile(r'^[\[({].*[\])}]$')

damaged_assessment = re.compile(r'[A-Za-z\.\-\_] [\.\-\_]$')
nil_assessment = re.compile(r'nil$')
#%% Need to combine all lines where one doesn't end in a number but the next one does.
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
    tdf = pd.read_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_lines.csv')
    new_df = pd.DataFrame()
    prev_text = ''
    tr_prev_text = ''
    eo_prev_text = ''
    pt_prev_text = ''
    prev_y1 = 0
    prev_note = 0
    # Replace all nans with empty strings
    tdf[['text', 'tr_text', 'eo_text', 'pt_text']] = tdf[['text', 'tr_text', 'eo_text', 'pt_text']].fillna('')
    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        if row['category'] == 'parish':
            prev_text = ''
            tr_prev_text = ''
            eo_prev_text = ''
            pt_prev_text = ''
            prev_y1 = 0
            prev_note = 0
            continue
        if fully_parenthetical.search(row['tr_text']) or damaged_assessment.search(row['tr_text']) or nil_assessment.search(row['tr_text']):
            if damaged_assessment.search(row['tr_text']) or nil_assessment.search(row['tr_text']):
                print(row['tr_text'])
            prev_text = ''
            tr_prev_text = ''
            eo_prev_text = ''
            pt_prev_text = ''
            prev_y1 = 0
            prev_note = 0
            continue
        new_row = row.copy()
        for reader in ['tr', 'eo', 'pt']:
            if row[f'{reader}_text'] == '' or pd.isna(row[f'{reader}_text']):
                continue
            text = row[f'{reader}_text'].strip()
            while '  ' in text:
                text = text.replace('  ', ' ')
            if number_at_start.search(text):
                text = number_at_start.sub('', text)
                text = text.strip()
            for missed_number in misidentified_number_dict:
                text = missed_number.sub(misidentified_number_dict[missed_number], text)
            if other_assessment.search(text):
                re.sub(other_assessment, '', text)
                while ' ' in text:
                    text = text.replace(' ', '')
            if missing_space.search(text):
                text = missing_space.sub(r'\1 \2', text)
            new_row[f'{reader}_text'] = text
        if not (number_at_end.search(new_row['tr_text']) or number_at_end.search(new_row['eo_text']) or number_at_end.search(new_row['pt_text'])):
            prev_text = prev_text + ' ' + new_row['text']
            prev_text = prev_text.strip()
            tr_prev_text = tr_prev_text + ' ' + new_row['tr_text']
            tr_prev_text = tr_prev_text.strip()
            eo_prev_text = eo_prev_text + ' ' + new_row['eo_text']
            eo_prev_text = eo_prev_text.strip()
            pt_prev_text = pt_prev_text + ' ' + new_row['pt_text']
            pt_prev_text = pt_prev_text.strip()
            prev_y1 = new_row['y1']
            prev_note = new_row['note']
            continue
        if prev_y1 != 0 and prev_note == row['note']:
            new_row['text'] = prev_text + ' ' + new_row['text']
            new_row['tr_text'] = tr_prev_text + ' ' + new_row['tr_text']
            new_row['eo_text'] = eo_prev_text + ' ' + new_row['eo_text']
            new_row['pt_text'] = pt_prev_text + ' ' + new_row['pt_text']
            new_row['y1'] = prev_y1

        new_df = pd.concat([new_df, pd.DataFrame(new_row).T], ignore_index=True)
        prev_text = ''
        tr_prev_text = ''
        eo_prev_text = ''
        pt_prev_text = ''
        prev_y1 = 0
        prev_note = 0

    new_df.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)

#%% Sorting out the individual names and surnames

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
    for reader in ['tr', 'eo', 'pt']:
        if f'{reader}_forename' not in tdf.columns:
            tdf[f'{reader}_forename'] = ''
            tdf[f'{reader}_surname'] = ''
            tdf[f'{reader}_title'] = ''
            tdf[f'{reader}_type'] = ''
            tdf[f'{reader}_value'] = 0

    for i, row in tqdm(tdf.iterrows(), total = len(tdf)):
        if row['category'] == 'parish':
            continue
        for reader in ['tr', 'eo', 'pt']:
            if row[f'{reader}_text'] == '' or pd.isna(row[f'{reader}_text']):
                continue
            text = row[f'{reader}_text'].strip()
            while '  ' in text:
                text = text.replace('  ', ' ')
            if number_at_start.search(text):
                text = number_at_start.sub('', text)
                text = text.strip()
            for missed_number in misidentified_number_dict:
                text = missed_number.sub(misidentified_number_dict[missed_number], text)
            if other_assessment.search(text):
                re.sub(other_assessment, '', text)
                while ' ' in text:
                    text = text.replace(' ', '')
            if missing_space.search(text):
                text = missing_space.sub(r'\1 \2', text)
            # Preparing to grab forename, surname, title, type, value
            reader_forename = ''
            reader_surname = ''
            reader_title = ''
            reader_type = 'G'
            reader_value = np.nan
            frac_value = 0
            # Deploying my little regexes
            if forename_pattern.search(text):
                reader_forename = forename_pattern.search(text).group(1)
                text = forename_pattern.sub('', text)
                text = text.strip()
            if surname_pattern.search(text):
                reader_surname = surname_pattern.search(text).group(1)
                text = surname_pattern.sub('', text)
                text = text.strip()
            if fraction.search(text):
                frac_value = int(fraction.search(text).group(1))/int(fraction.search(text).group(2))
                text = fraction.sub('', text)
                text = text.strip()
            if number_at_end.search(text):
                reader_value = int(number_at_end.search(text).group(1))
                reader_value += frac_value
                text = number_at_end.sub('', text)
                text = text.strip()
            for type in type_dict:
                if type.search(text):
                    reader_type = type_dict[type]
                    text = type.sub('', text)
                    text = text.strip()
            for title in title_list:
                if title.search(text):
                    reader_title = title.search(text).group(1)
                    text = title.sub('', text)
                    text = text.strip()
            tdf.at[i, f'{reader}_forename'] = reader_forename
            tdf.at[i, f'{reader}_surname'] = reader_surname
            tdf.at[i, f'{reader}_title'] = reader_title
            tdf.at[i, f'{reader}_type'] = reader_type
            tdf.at[i, f'{reader}_value'] = reader_value


    tdf.to_csv(f'{SUBSIDY_FOLDER}/subsidy{subsidy}_taxpayers.csv', index=False)