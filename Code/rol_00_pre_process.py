#%%
import os
import cv2
import ast
import json
import time
import shutil
import platform
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
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
IMAGES = 'Output/Images'
TABLES = 'Output/Tables'

#%% load pdf page as image
for file in os.listdir(f'{RAW}/landowners_1873'):
    if not file.endswith('.pdf'):
        continue
    filename = file.replace('.pdf', '')
    if filename + '.png' in os.listdir(f'{RAW}/landowners_1873'):
        continue
    img = convert_from_path(os.path.join(RAW, 'landowners_1873', file),
                            dpi=500,
                            fmt='png',
                            output_folder=f'{RAW}/landowners_1873',
                            output_file = filename,
                            grayscale=True)

#%%
for file in os.listdir(f'{RAW}/landowners_1873'):
    if not file.endswith('.png') or '0001-1' not in file:
        continue
    os.rename(f'{RAW}/landowners_1873/{file}',
               f'{RAW}/landowners_1873/{file}'.replace('0001-1', ''))

#%% Pre-process images

def pre_process(image):
    # Denoise
    image_denoised = cv2.fastNlMeansDenoising(src=image,
                                              dst=None,
                                              h=10,
                                              templateWindowSize=7,
                                              searchWindowSize=21)
    cv2.imwrite('denoised.png', image_denoised)
    clahe = cv2.createCLAHE(clipLimit=1,
                            tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_denoised)
    cv2.imwrite('clahe.png', image_clahe)
    # Threshold
    image_clahe = cv2.medianBlur(image_clahe, 3)
    _, image_thresh = cv2.threshold(image_clahe, 127, 255, cv2.THRESH_BINARY)

    return image_thresh

for file in os.listdir(f'{RAW}/landowners_1873'):
    if not file.endswith('.png'):
        continue
    image = cv2.imread(os.path.join(RAW, 'landowners_1873', file), 0)
    image = pre_process(image)
    cv2.imwrite(os.path.join(RAW, 'landowners_1873', file), image)

#%% Bolding images
horiz_kernel = np.ones((1, 3), np.uint8)
vert_kernel = np.ones((3, 1), np.uint8)
for file in os.listdir(f'{RAW}/landowners_1873'):
    if not file.endswith('.png'):
        continue
    if 'bold_' in file or 'TEMPLATE' in file:
        continue
    image = cv2.imread(os.path.join(RAW, 'landowners_1873', file), 0)
    image = cv2.erode(image, horiz_kernel, iterations=1)
    image = cv2.erode(image, vert_kernel, iterations=1)
    image = cv2.erode(image, horiz_kernel, iterations=1)
    image = cv2.erode(image, vert_kernel, iterations=1)
    cv2.imwrite(os.path.join(RAW, 'landowners_1873', 'bold_' + file), image)

#%%
