import os
import re
from tqdm.auto import tqdm
import platform
import time

import numpy as np
import pandas as pd

from PIL import Image
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch

import pytesseract as pt
import easyocr
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator, AutoTokenizer


if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    print('oh shit oh fuck')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


#%% Globals
PROCESSED_DATA = 'Data/Processed'
MODEL_FOLDER = f'Code/ML Models/trocr_model'
#%% Load our models

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
eo_model = easyocr.Reader(['en'])
model_size = 'small'
#%% Getting Latest Model


try:
    checkpoint_dict = {}
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    for folder in os.listdir(MODEL_FOLDER):
        modified_time = os.path.getmtime(f'{MODEL_FOLDER}/{folder}')
        checkpoint_dict[modified_time] = folder
    latest_checkpoint = checkpoint_dict[max(checkpoint_dict.keys())]
    latest_model_location = os.path.join(MODEL_FOLDER, latest_checkpoint)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(latest_model_location,
                                                      cache_dir=latest_model_location,
                                                      local_files_only=True)
    print('Latest Model Loaded!')
    for folder in os.listdir(MODEL_FOLDER):
        if folder != latest_checkpoint:
            shutil.rmtree(f'{MODEL_FOLDER}/{folder}')
    print('Older models vaporized!')
except:

    trocr_model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/trocr-{model_size}-printed')
    print('Model Downloaded')

processor = TrOCRProcessor.from_pretrained(f'microsoft/trocr-{model_size}-printed',
                                           use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(f'microsoft/trocr-{model_size}-printed',
                                          use_fast=False)
print('Processor and tokenizer downloaded')

trocr_model.to(device)

#%% Creating the Two Lists

page_pattern = re.compile(r'(\d+,*)')
see_also_pattern = re.compile(r' ?see ?also ?([A-Z][a-z]+)')
see_pattern = re.compile(r' ?see ?([A-Z][a-z]+)')
replacement = r', \1'
symbol_pattern = re.compile(r'[-_+=/\\<>?;:\'\"!@#$%^&*|]')
alt_spelling_pattern = re.compile(r'([A-Z][a-z]+)\((\w)\)')
replacement_two = r'\1, \1\2'
cont_pattern = re.compile(r'([A-Z][a-z]+ \(cont\.\))')

#%%
for subsidy in [
    1524,
    1543,
    1581,
    # 1642,
    # 1647,
    # 1660,
    # # 1661,
    1674,
     ]:
    INDEX_FOLDER = f'{PROCESSED_DATA}/subsidy{subsidy}/processed_pages/surname_index'
    surname_index_df = pd.DataFrame()
    for image_file in tqdm(os.listdir(INDEX_FOLDER), total=len(os.listdir(INDEX_FOLDER))):
        if not image_file.endswith('.png'):
            continue
        page = int(image_file[-7:-4])
        start = time.time()
        img = Image.open(f'{INDEX_FOLDER}/{image_file}')
        img = img.crop((0, 400, img.size[0], img.size[1]))
        img_array = np.array(img)
        vert_proj = np.sum(img_array, axis=0)
        projection_series = pd.Series(vert_proj)
        projection_rolling = projection_series.rolling(window=100, center=True)
        projection_smoothed = projection_rolling.mean()
        projection_smoothed = projection_smoothed * -1 + 1_350_000
        peaks, properties = signal.find_peaks(x=projection_smoothed,
                                              height = 50_000,
                                              distance=1000,
                                              prominence = 10000)
        peaks = [x - 100 for x in peaks]
        peaks = peaks[1:]
        # plot the peaks on the image
        if page % 10 == 0:
            plt.imshow(img_array, cmap='gray')
            for peak in peaks:
                plt.axvline(x=peak, color='r', linestyle='--')
            plt.show()

        cut_list = [0] + list(peaks) + [img_array.shape[1]]
        cuts_done = time.time()
        print(f'Cuts done in {cuts_done - start} seconds')
        for i in range(len(cut_list) - 1):
            start_cut = time.time()
            if i > 0:
                break
            column_array = img_array[:, cut_list[i]:cut_list[i+1]]
            horiz_proj = np.sum(column_array, axis=1)
            projection_series = pd.Series(horiz_proj)
            projection_rolling = projection_series.rolling(window=50, center=True)
            projection_smoothed = projection_rolling.mean()
            peaks, properties = signal.find_peaks(x=projection_smoothed, distance=50, prominence=2000)
            slice_list = [0] + list(peaks) + [column_array.shape[0]]
            if page % 10 == 0:
                plt.imshow(column_array, cmap='gray')
                for peak in peaks:
                    plt.axhline(y=peak, color='r', linestyle='--')
                plt.show()
            end_cut = time.time()
            print(f'Cut {i} done in {end_cut - start_cut} seconds')
            for j in tqdm(range(len(slice_list) - 1)):
                if page == 276:
                    if i == 0 and j < 3:
                        continue
                    if i > 0 and j < 4:
                        continue
                slice_start = time.time()
                top = slice_list[j]
                bottom = slice_list[j+1]
                line_height = bottom - top

                if line_height < 100:
                    gap = 100 - line_height
                    top = max(0, top - gap//2)
                    bottom = min(column_array.shape[0], bottom + gap//2)

                # Getting image in right format for OCRs
                line_array = column_array[top:bottom, :]
                line_img = Image.fromarray(line_array)
                line_img_color = line_img.convert('RGB')

                # Pytesseract
                pt_text = pt.image_to_string(line_img, config='-c tessedit_char_whitelist=0123456789QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm,()').strip()
                pt_text = pt_text.replace('Ww', 'W').replace('wW', 'W')
                pt_text = re.sub(cont_pattern, '', pt_text)
                pt_text = re.sub(symbol_pattern, '', pt_text)
                pt_text = re.sub(see_also_pattern, replacement, pt_text)
                pt_text = re.sub(see_pattern, replacement, pt_text)
                pt_text = re.sub(alt_spelling_pattern, replacement_two, pt_text)
                pt_text = pt_text.replace('(', '').replace(')', '')
                # Getting pages
                pt_pages = re.findall(page_pattern, pt_text)
                pt_pages = [int(x.replace(',', '')) for x in pt_pages]
                # Getting names
                pt_names = re.sub(page_pattern, '', pt_text).strip().split(',')
                pt_names = [x.replace(' ', '').capitalize() for x in pt_names]
                while '' in pt_names:
                    pt_names.remove('')
                pt_done = time.time()
                print(f'PT done in {pt_done - slice_start} seconds')

                # EasyOCR
                eo_text = eo_model.readtext(line_array, detail=0)
                eo_text = ''.join(eo_text)
                eo_text = eo_text.replace('Ww', 'W').replace('wW', 'W')
                eo_text = re.sub(cont_pattern, '', eo_text)
                eo_text = re.sub(symbol_pattern, '', eo_text)
                eo_text = re.sub(see_also_pattern, replacement, eo_text)
                eo_text = re.sub(see_pattern, replacement, eo_text)
                eo_text = re.sub(alt_spelling_pattern, replacement_two, eo_text)
                eo_text = eo_text.replace('(', '').replace(')', '')
                # Getting pages
                eo_pages = re.findall(page_pattern, eo_text)
                eo_pages = [int(x.replace(',', '')) for x in eo_pages]
                # Getting names
                eo_names = re.sub(page_pattern, '', eo_text).strip().split(',')
                eo_names = [x.replace(' ', '').capitalize() for x in eo_names]
                while '' in eo_names:
                    eo_names.remove('')
                eo_done = time.time()
                print(f'EO done in {eo_done - pt_done} seconds')

                # TROCR
                pixel_values = processor(line_img_color, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
                generated_ids = trocr_model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                tr_text = generated_text.strip()
                tr_text = tr_text.replace('Ww', 'W').replace('wW', 'W')
                tr_text = re.sub(cont_pattern, '', tr_text)
                tr_text = re.sub(symbol_pattern, '', tr_text)
                tr_text = re.sub(see_also_pattern, replacement, tr_text)
                tr_text = re.sub(see_pattern, replacement, tr_text)
                tr_text = re.sub(alt_spelling_pattern, replacement_two, tr_text)
                tr_text = tr_text.replace('(', '').replace(')', '')
                # Getting pages
                tr_pages = re.findall(page_pattern, tr_text)
                tr_pages = [int(x.replace(',', '')) for x in tr_pages]
                # Getting names
                tr_names = re.sub(page_pattern, '', tr_text).strip().split(',')
                tr_names = [x.replace(' ', '').capitalize() for x in tr_names]
                while '' in tr_names:
                    tr_names.remove('')
                tr_done = time.time()
                print(f'TR done in {tr_done - eo_done} seconds')
                # Harmonizing the three predictions on page numbers
                page_tuple_list = [tuple(pt_pages), tuple(eo_pages), tuple(tr_pages)]
                if len(set(page_tuple_list)) == 1:
                    pages = pt_pages
                elif len(set(page_tuple_list)) == 2:
                    if page_tuple_list.count(tuple(pt_pages)) > page_tuple_list.count(tuple(eo_pages)):
                        pages = pt_pages
                    else:
                        pages = eo_pages
                else:
                    assert len(set(page_tuple_list)) == 3
                    pages = tr_pages

                # Harmonizing the three predictions on names
                name_tuple_list = [tuple(pt_names), tuple(eo_names), tuple(tr_names)]
                if len(set(name_tuple_list)) == 1:
                    names = pt_names
                elif len(set(name_tuple_list)) == 2:
                    if name_tuple_list.count(tuple(pt_names)) > name_tuple_list.count(tuple(eo_names)):
                        names = pt_names
                    else:
                        names = eo_names
                else:
                    assert len(set(name_tuple_list)) == 3
                    names = tr_names
                if len(names) == 0:

                    new_pages = surname_index_df.at[surname_index_df.index[-1], 'pages'] + pages
                    surname_index_df.at[surname_index_df.index[-1], 'pages'] = new_pages

                    new_pt_pages = surname_index_df.at[surname_index_df.index[-1], 'pt_pages'] + pt_pages
                    surname_index_df.at[surname_index_df.index[-1], 'pt_pages'] = new_pt_pages

                    new_eo_pages = surname_index_df.at[surname_index_df.index[-1], 'eo_pages'] + eo_pages
                    surname_index_df.at[surname_index_df.index[-1], 'eo_pages'] = new_eo_pages

                    new_tr_pages = surname_index_df.at[surname_index_df.index[-1], 'tr_pages'] + tr_pages
                    surname_index_df.at[surname_index_df.index[-1], 'tr_pages'] = new_tr_pages
                    continue

                new_line = pd.DataFrame({'pt_text': [pt_text], 'eo_text': [eo_text], 'tr_text': [tr_text],
                                         'pt_pages': [pt_pages], 'eo_pages': [eo_pages], 'tr_pages': [tr_pages],
                                         'pages': [pages], 'names': [names], 'year': [subsidy]})
                surname_index_df = pd.concat([surname_index_df, new_line], ignore_index=True)
                slice_done = time.time()
                print(f'Harmonization done in {slice_done - tr_done} seconds')



print(surname_index_df[['names', 'pages']])
print(surname_index_df.iloc[0]['pages'])
surname_index_df.to_csv(f'{PROCESSED_DATA}/surname_index.csv', index=False, encoding='utf-8')