import os
import re

import numpy as np
import pandas as pd

from PIL import Image
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel, \
    default_data_collator, AutoTokenizer
import torch
import pytesseract as pt
import easyocr
import shutil
from tqdm.auto import tqdm
import shutil
import google.generativeai as genai
import json
import time
import platform

# %%
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
eo_model = easyocr.Reader(['en'])

model_size = 'small'

if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

MODEL_FOLDER = f'Code/ml_models/trocr_model_{model_size}'
PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using device: {device}')


# %% Functions

def fix_trailing_characters(text):
    '''Replaces mis-read digits with correct digits'''
    replacements = {
        ' B': ' 8',
        ' b': ' 6',
        ' S': ' 5',
        ' G': ' 6',
        ' I': ' 1',
        ' l': ' 1'
    }
    if text[-2:] in replacements.keys():
        text = text[:-2] + replacements[text[-2:]]
    return text


def fix_weird_shit(text):
    '''Removing stupid nonsense characters'''

    weird_shit = re.compile(r'[?!+*\n®­;:‘’“”•]')
    text = re.sub(weird_shit, '', text)

    return text


# %% Getting Latest Model


try:
    checkpoint_dict = {}
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    for folder in os.listdir(MODEL_FOLDER):
        modified_time = os.path.getmtime(f'{MODEL_FOLDER}/{folder}')
        checkpoint_dict[modified_time] = folder
    latest_checkpoint = checkpoint_dict[max(checkpoint_dict.keys())]
    latest_model_location = os.path.join(MODEL_FOLDER, latest_checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(latest_model_location,
                                                      cache_dir=latest_model_location,
                                                      local_files_only=True)
    print('Latest Model Loaded!')
    for folder in os.listdir(MODEL_FOLDER):
        if folder != latest_checkpoint:
            shutil.rmtree(f'{MODEL_FOLDER}/{folder}')
    print('Older models vaporized!')
except:
    model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/trocr-{model_size}-printed')
    print('Model Downloaded')

# Processor and tokenizer

processor = TrOCRProcessor.from_pretrained(f'microsoft/trocr-{model_size}-printed', use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(f'microsoft/trocr-{model_size}-printed', use_fast=False)
print('Processor and tokenizer downloaded')

model.to(device)
parish_model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/trocr-{model_size}-printed')
parish_model.to(device)

# %%

parish_pattern = re.compile(r'([a-z\s*]+)+\sparish\s?(\([a-z\s+]+\))?')

# %%

text_df = pd.DataFrame(columns=['page', 'note', 'category', 'line', 'eo_text', 'pt_text'])

tr_parish = ''
eo_parish = ''
pt_parish = ''

for subsidy in [
    1524,
    # 1543,
    # 1581,
    # 1642,
    # 1647,
    # 1660,
    # 1674
]:
    tdf = pd.read_csv(f'{PROCESSED}/subsidy{subsidy}/subsidy{subsidy}_lines.csv')
    prev_page = 0
    for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
        page = row['page']
        category = row['category']
        note = row['note']
        line = row['line']
        image_name = row['image_name'].replace('little_', 'processed_')

        if page != prev_page:
            page_image = Image.open(f'{PROCESSED}/subsidy{subsidy}/processed_pages/{image_name}')
            prev_page = page
        x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']
        line_image = page_image.crop((x1, y1, x2, y2))
        line_image_array = np.array(line_image)
        line_image_color = line_image.convert('RGB')
        if row['category'] == 'parish':
            pixel_values = processor(line_image_color, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = parish_model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            tr_parish = generated_text.strip()
            eo_parish = eo_model.readtext(line_image_array)
            eo_parish = ' '.join([x[1] for x in eo_parish])
            pt_parish = pt.image_to_string(line_image).strip()
            tdf.at[i, 'tr_parish'] = tr_parish
            tdf.at[i, 'eo_parish'] = eo_parish
            tdf.at[i, 'pt_parish'] = pt_parish
            continue

        pixel_values = processor(line_image_color, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        tr_text = generated_text.strip()

        eo_text = eo_model.readtext(line_image_array)
        eo_text = ' '.join([x[1] for x in eo_text])

        pt_text = pt.image_to_string(line_image).strip()

        tr_text = fix_trailing_characters(fix_weird_shit(tr_text))
        eo_text = fix_trailing_characters(fix_weird_shit(eo_text))
        pt_text = fix_trailing_characters(fix_weird_shit(pt_text))

        if tr_text == '' and eo_text == '' and pt_text == '':
            print('NOTHING HERE')
            continue

        tdf.at[i, 'tr_text'] = tr_text
        tdf.at[i, 'eo_text'] = eo_text
        tdf.at[i, 'pt_text'] = pt_text

        tdf.at[i, 'tr_parish'] = tr_parish
        tdf.at[i, 'eo_parish'] = eo_parish
        tdf.at[i, 'pt_parish'] = pt_parish

    tdf.to_csv(f'{PROCESSED}/subsidy{subsidy}/subsidy{subsidy}_lines.csv', index=False)
