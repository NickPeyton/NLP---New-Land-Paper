#%%
import json
import pandas as pd
import numpy as np
import phonetics as ph
import re
from tqdm.auto import tqdm
tqdm.pandas()
import os
import unidecode
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz import fuzz
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#%% Globals
PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'
MODELS = f'Code/ml_models/'

with open(f'{PROCESSED}/combined_surnames.json') as f:
    surname_lists = json.load(f)
surname_id_df = pd.DataFrame()
for i, surname_list in tqdm(enumerate(surname_lists), total=len(surname_lists)):
    for surname in surname_list:
        surname_alpha = ''.join(char for char in surname if char.isalpha()).title()
        surname_soundex = ph.soundex(surname_alpha)
        surname_metaphone = ph.metaphone(surname_alpha)
        surname_id_df = pd.concat([surname_id_df, pd.DataFrame({'id': [i+1],
                                                                'surname': [surname],
                                                                'soundex': [surname_soundex],
                                                                'metaphone': [surname_metaphone],
                                                                's_m': [surname_soundex + ' ' + surname_metaphone]})], ignore_index=True)
surname_id_df = surname_id_df.reset_index(drop=True)


#%% Regex for some of the weirder formats

alt_curly_brackets = re.compile(r'([A-Za-z]+)\s+\{[A-Za-z]+\}')
single_curly_brackets = re.compile(r'\{([A-Za-z]+)\}')
alias_pattern = re.compile(r'([A-Za-z]+) ali?a?s ([A-Za-z]+)')
or_pattern = re.compile(r'([A-Za-z]+) or ([A-Za-z]+)')
double_name = re.compile(r'([A-Za-z]+)\s+([A-Za-z]+)')
and_sons = re.compile(r'([A-Za-z]+) & sons+', re.IGNORECASE)
and_co = re.compile(r'([A-Za-z]+) & co', re.IGNORECASE)
saint_pattern = re.compile(r'Sa?i?n?t\s+([A-Za-z]+)')

#%% Loading our little model
def encode_surname(surname, max_len=24):
    CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0'
    CHARSET_DICT = {char: i + 1 for i, char in enumerate(CHARSET)}
    PAD = 0
    surname = surname.upper()
    surname = ''.join([char for char in surname if char in CHARSET])
    metaphone = ph.metaphone(surname)

    encoded = [CHARSET_DICT[char] for char in surname]
    if len(encoded) < max_len:
        encoded += [PAD] * (max_len - len(encoded))
    encoded = torch.tensor(encoded).long()

    encoded_metaphone = [CHARSET_DICT[char] for char in metaphone]
    if len(encoded_metaphone) < max_len:
        encoded_metaphone += [PAD] * (max_len - len(encoded_metaphone))
    encoded_metaphone = torch.tensor(encoded_metaphone).long()
    return encoded, encoded_metaphone

class CrossEncoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, fc_dim=32):
        super(CrossEncoder, self).__init__()

        # Embedding layer (shared between names and metaphones)
        self.name_embedding = nn.Embedding(28, embed_dim)  # Assuming 27 letters + 1 padding
        self.metaphone_embedding = nn.Embedding(28, embed_dim)
        # BiLSTM for sequence encoding
        self.name_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.metaphone_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(4*2 * hidden_dim, fc_dim)  # Combining all four encodings
        self.fc2 = nn.Linear(fc_dim, 1)

    def name_encode(self, x):
        name_embedded = self.name_embedding(x)
        _, (name_hidden, _) = self.name_lstm(name_embedded)
        name_hidden = torch.cat((name_hidden[0], name_hidden[1]), dim=1)  # Concatenate forward & backward LSTM outputs
        return name_hidden
    def metaphone_encode(self, x):
        metaphone_embedded = self.metaphone_embedding(x)
        _, (metaphone_hidden, _) = self.metaphone_lstm(metaphone_embedded)
        metaphone_hidden = torch.cat((metaphone_hidden[0], metaphone_hidden[1]), dim=1)  # Concatenate forward & backward LSTM outputs
        return metaphone_hidden

    def forward(self, name1, metaphone1, name2, metaphone2):
        # Encode each input separately
        name1_encoded = self.name_encode(name1)
        metaphone1_encoded = self.metaphone_encode(metaphone1)
        name2_encoded = self.name_encode(name2)
        metaphone2_encoded = self.metaphone_encode(metaphone2)
        # Concatenate all representations
        combined = torch.cat((name1_encoded, metaphone1_encoded, name2_encoded, metaphone2_encoded), dim=1)

        # Fully connected layers
        fc1_out = self.fc1(combined)
        fc1_relud = F.relu(fc1_out)
        output = torch.sigmoid(self.fc2(fc1_relud))  # Binary classification

        return output

# Load the model
model = CrossEncoder()
model.load_state_dict(torch.load(f'{MODELS}/name_matcher/cross_encoder_1'))
model.eval()
model.to(device)


#%%

master_subsidy_list = []

for doc_file in [
    # f'{PROCESSED}/master_subsidy_data.csv',
    # f'{PROCESSED}/tithe_landowners.csv',
    # f'{RAW}/freeholders_list_1713_1780.csv',
    # f'{RAW}/bank_returns_1845_1880.csv',
    # f'{RAW}/bankrupts_list_1800_1820.csv',
    # f'{RAW}/bankrupts_list_1820_1843.csv',
    # f'{RAW}/indictable_offenses_1745_1782.csv',
    # f'{RAW}/monumental_brasses.csv',
    # f'{RAW}/victuallers_list_1651_1828.csv',
    # f'{RAW}/workhouse_list_1861.csv',
    f'{PROCESSED}/ukda_pcc_wills.csv'
     ]:
    tdf = pd.read_csv(doc_file, encoding='utf-8')
    if 'subsidy' in doc_file:
        surname_col = 'gemini_surname'
    elif 'tithe' in doc_file:
        surname_col = 'occupier_surname'
    else:
        surname_col = 'surname'
        tdf = tdf[tdf['surname'] != '[No entries]']
        tdf = tdf[tdf['surname'] != 'None Qualified']
    tdf.rename(columns={'surname_metaphone': 'metaphone',
                        'surname_soundex': 'soundex'}, inplace=True)
    tdf[surname_col] = tdf[surname_col].replace('NO_SURNAME', np.nan)
    tdf[surname_col] = tdf[surname_col].str.replace('1', 'l')
    tdf[surname_col] = tdf[surname_col].str.replace('2', 'z')
    tdf[surname_col] = tdf[surname_col].str.replace('3', 'e')
    tdf[surname_col] = tdf[surname_col].str.replace('4', 'a')
    tdf[surname_col] = tdf[surname_col].str.replace('5', 's')
    tdf[surname_col] = tdf[surname_col].str.replace('6', 'b')
    tdf[surname_col] = tdf[surname_col].str.replace('7', 't')
    tdf[surname_col] = tdf[surname_col].str.replace('8', 'b')
    tdf[surname_col] = tdf[surname_col].str.replace('9', 'g')
    tdf[surname_col] = tdf[surname_col].str.replace('0', 'o')
    tdf[surname_col] = tdf[surname_col].str.replace(',', '')
    tdf[surname_col] = tdf[surname_col].str.replace('.', '')
    tdf[surname_col] = tdf[surname_col].str.replace('\'', '')
    tdf[surname_col] = tdf[surname_col].str.replace('(' , '')
    tdf[surname_col] = tdf[surname_col].str.replace(')', '')
    tdf[surname_col] = tdf[surname_col].str.replace('-', '')
    tdf[surname_col] = tdf[surname_col].str.replace('/', '')
    tdf[surname_col] = tdf[surname_col].str.replace('?', '')
    tdf[surname_col] = tdf[surname_col].str.replace('­', '')
    tdf[surname_col] = tdf[surname_col].str.replace(';', '')
    tdf[surname_col] = tdf[surname_col].str.replace(':', '')
    tdf[surname_col] = tdf[surname_col].str.replace('‘', '')
    tdf[surname_col] = tdf[surname_col].str.replace('’', '')
    tdf[surname_col] = tdf[surname_col].str.replace('“', '')
    tdf[surname_col] = tdf[surname_col].str.replace('”', '')
    tdf[surname_col] = tdf[surname_col].str.replace('•', '')
    tdf[surname_col] = tdf[surname_col].str.replace('!', '')
    tdf[surname_col] = tdf[surname_col].str.replace('ĩ', 'i')
    tdf[surname_col] = tdf[surname_col].str.replace('"', '')
    tdf[surname_col] = tdf[surname_col].replace('#NAME', np.nan)
    tdf[surname_col] = tdf[surname_col].str.replace('ó', 'o')

    tdf[surname_col] = tdf[surname_col].str.replace(alt_curly_brackets, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(single_curly_brackets, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(alias_pattern, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(or_pattern, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(and_sons, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(and_co, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.replace(saint_pattern, r'\1', regex=True)

    tdf[surname_col] = tdf[surname_col].str.replace('& ', '')
    tdf[surname_col] = tdf[surname_col].str.replace('{', '')
    tdf[surname_col] = tdf[surname_col].str.replace('}', '')
    tdf[surname_col] = tdf[surname_col].str.replace('[', '')
    tdf[surname_col] = tdf[surname_col].str.replace(']', '')
    tdf[surname_col] = tdf[surname_col].str.strip()
    tdf[surname_col] = tdf[surname_col].str.replace(double_name, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.strip()
    tdf[surname_col] = tdf[surname_col].str.replace(double_name, r'\1', regex=True)
    tdf[surname_col] = tdf[surname_col].str.strip()

    tdf = tdf[tdf[surname_col] != '']
    tdf = tdf[tdf[surname_col] != 'None Qualified']
    tdf = tdf[tdf[surname_col] != 'NO_SURNAME']
    tdf = tdf[tdf[surname_col] != 'No Entries']

    # Attach a surname id to each surname, generate new id if not found
    tdf = tdf.merge(surname_id_df[['surname', 'id']], how='left', left_on=surname_col, right_on='surname')
    if 'tithe' in doc_file or 'subsidy' in doc_file:
        tdf = tdf.drop(columns=['surname'])
    # Create id dict for surname soundex_metaphone column
    with open(f'{PROCESSED}/soundex_metaphone_id_dict.json', 'r', encoding='utf-8') as f:
        s_m_id_dict = json.load(f)
    if 'subsidy' in doc_file or 'soundex' not in tdf.columns or 'metaphone' not in tdf.columns:
        tdf['soundex'] = tdf[surname_col].progress_apply(lambda x: ph.soundex(''.join(unidecode.unidecode(char) for char in x if char.isalpha()).title()) if not pd.isna(x) and x != '' else np.nan)
        tdf['metaphone'] = tdf[surname_col].progress_apply(lambda x: ph.metaphone(''.join(unidecode.unidecode(char) for char in x if char.isalpha()).title()) if not pd.isna(x) and x != '' else np.nan)

    tdf['s_m'] = tdf['soundex'] + ' ' + tdf['metaphone']
    if 'id' not in tdf.columns:
        tdf['id'] = np.nan
    unique_surname_id_df = surname_id_df[['surname', 'metaphone', 'id']].drop_duplicates(subset='id')
    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        if not pd.isna(row['id']):
            continue
        if pd.isna(row[surname_col]):
            tdf.at[i, 'id'] = 0
            continue
        if row['s_m'] in s_m_id_dict:
            tdf.at[i, 'id'] = s_m_id_dict[row['s_m']]
        surname_tensor, metaphone_tensor = encode_surname(row[surname_col])
        surname_tensor = surname_tensor.to(device)
        metaphone_tensor = metaphone_tensor.to(device)
        pred_list = []
        if 'mp_edit_dist' in unique_surname_id_df.columns:
            unique_surname_id_df.drop(columns=['mp_edit_dist'], inplace=True)
        unique_surname_id_df['mp_edit_dist'] = unique_surname_id_df['metaphone'].apply(lambda x: fuzz.ratio(x, row['metaphone']))
        unique_surname_id_df.sort_values('mp_edit_dist', ascending=False, inplace=True)
        for j, surname_row in unique_surname_id_df.iterrows():
            surname2_tensor, metaphone2_tensor = encode_surname(surname_row['surname'])
            surname2_tensor = surname2_tensor.to(device)
            metaphone2_tensor = metaphone2_tensor.to(device)
            with torch.no_grad():
                pred = model(surname_tensor.unsqueeze(0), metaphone_tensor.unsqueeze(0),
                             surname2_tensor.unsqueeze(0), metaphone2_tensor.unsqueeze(0))
            pred_list.append(pred.item())
            if pred.item() > 0.90:
                tdf.at[i, 'id'] = surname_row['id']
                s_m_id_dict[row['s_m']] = surname_row['id']
                found = True
                break

        else:
            print('oar noar')
            tdf.at[i, 'id'] = len(unique_surname_id_df) + 1
            unique_surname_id_df = pd.concat([unique_surname_id_df, pd.DataFrame({'surname': [row[surname_col]],
                                                                                 'id': [len(unique_surname_id_df) + 1]})], ignore_index=True)
            s_m_id_dict[row['s_m']] = len(unique_surname_id_df)

    tdf.to_csv(doc_file.replace('.csv', '_final.csv'), index=False, encoding='utf-8')
    with open(f'{PROCESSED}/soundex_metaphone_id_dict.json', 'w', encoding='utf-8') as f:
        json.dump(s_m_id_dict, f, ensure_ascii=False, indent=4)



