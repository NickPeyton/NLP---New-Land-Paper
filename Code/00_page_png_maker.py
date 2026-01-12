import pdf2image
import os
import json
import platform
from tqdm.auto import tqdm
from torchvision import transforms
import torchvision
import torch
from tqdm.auto import tqdm
import cv2
import numpy as np
import time


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
PDF_FOLDER = f'{RAW}/subsidies'

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

#%%
with open(os.path.join(RAW, 'doc_description.json'), 'r') as f:
    doc_description = json.load(f)

for doc in doc_description:

    doc_subsidy = int(doc['name'].replace('subsidy', '').replace('.pdf', ''))
    if doc_subsidy != 1674:
        continue
    doc_sections = doc['sections']
    doc_sections_dict = {section['section_name']: [x for x in range(section['start'], section['end']+1)] for section in doc_sections}
    doc_page_dict = {}
    for key in doc_sections_dict:
        for page in doc_sections_dict[key]:
            doc_page_dict[page] = key


    IMAGE_FOLDER = f'{RAW}/staging_pages/'
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    pdf_file = doc['name']
    last_page = doc['pages']
    if f'subsidy{doc_subsidy}_page_{last_page}.png' not in os.listdir(IMAGE_FOLDER):
        pdf2image.convert_from_path(f'{PDF_FOLDER}/{pdf_file}',
                                    dpi=900,
                                    fmt='png',
                                    thread_count=12,
                                    output_folder=IMAGE_FOLDER,
                                    output_file=f'subsidy{doc_subsidy}_page_',
                                    grayscale=True)
        print('PDF converted to images!')
    else:
        print('PDF already converted to images, skipping this nonsense')
    #%% Renaming stupid files
    for file in os.listdir(IMAGE_FOLDER):
        if len(file) < 25:
            continue
        filename = file[:-12] + file[-7:-4] + '.png'
        os.rename(os.path.join(IMAGE_FOLDER, file), os.path.join(IMAGE_FOLDER, filename))
    print('Files renamed!')
    time.sleep(1)
    #%% Resizing images
    # Transforming all images to 5760 x 4480 aspect ratio - losing some efficiency but I'd rather do this all at once here
    resize = transforms.Resize([5760, 4480])
    little_resize = transforms.Resize([576, 448])
    prev_section = 'ooey gooey'

    for file in tqdm(os.listdir(IMAGE_FOLDER)):
        if 'page' not in file:
            print('AAAAAAAAAAAAAAAAAAAAAAAA THIS SHOULD NEVER HAPPEN')
            continue
        image_path = os.path.join(IMAGE_FOLDER, file)
        page = int(file[-7:-4])
        if page not in doc_page_dict:
            os.remove(image_path)
            continue
        section = doc_page_dict[page]
        if 'subsidy' not in section and 'index' not in section:
            os.remove(image_path)
            continue
        if section != prev_section:
            if 'subsidy' in section:
                subsidy = int(section.replace('subsidy', ''))
                destination_folder = f'{PROCESSED}/{section}/processed_pages'
                os.makedirs(destination_folder, exist_ok=True)
            elif 'index' in section:
                destination_folder = f'{PROCESSED}/subsidy{subsidy}/{section}'
                os.makedirs(destination_folder, exist_ok=True)

        LITTLE_PAGES_FOLDER = f'{PROCESSED}/subsidy{subsidy}/little_pages'
        os.makedirs(LITTLE_PAGES_FOLDER, exist_ok=True)
        processed_filename = 'processed_' + file.replace(str(doc_subsidy), str(subsidy))
        little_filename = 'little_' + file.replace(str(doc_subsidy), str(subsidy))
        if processed_filename in os.listdir(destination_folder):
            os.remove(image_path)
            continue
        image_path = os.path.join(IMAGE_FOLDER, file)
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
        image = image.to(device=device,
                         dtype=torch.float)
        image = resize(image).squeeze(0)
        image_array = image.cpu().numpy().astype(np.uint8)
        processed_image = pre_process(image_array)
        cv2.imwrite(os.path.join(destination_folder, processed_filename), processed_image)
        # Convert processed_image directly to torch to resize for tiny uwu page
        processed_image_torch = torch.from_numpy(processed_image).to(device=device,
                                                                     dtype=torch.float)
        little_image_torch = little_resize(processed_image_torch.unsqueeze(0)).squeeze(0)
        torchvision.utils.save_image(little_image_torch, os.path.join(f'{PROCESSED}/subsidy{subsidy}/little_pages', little_filename))
        # Delete the original
        os.remove(image_path)
        prev_section = section
    print('Images processed!')


#%%
little_resize = transforms.Resize([576, 448])
for subsidy in [
    # 1524,
    # 1543,
    # 1581,
    # 1642,
    # 1647,
    # 1660,
    # 1661,
    1674]:
    PAGES_FOLDER = f'{PROCESSED}/subsidy{subsidy}/processed_pages'
    LITTLE_PAGES_FOLDER = f'{PROCESSED}/subsidy{subsidy}/little_pages'
    os.makedirs(LITTLE_PAGES_FOLDER, exist_ok=True)
    for file in tqdm(os.listdir(PAGES_FOLDER)):
        little_filename = file.replace('processed_', 'little_')
        image_path = os.path.join(PAGES_FOLDER, file)
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
        image = image.to(device=device,
                         dtype=torch.float)
        little_image = little_resize(image)
        little_image_numpy = little_image.squeeze(0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(LITTLE_PAGES_FOLDER, little_filename), little_image_numpy)
