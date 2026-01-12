import os
import cv2
import ast
import json
import time
import shutil
import platform
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pytesseract as pt
from scipy import ndimage
from PIL import Image
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseract_config = r'--oem 3 --psm 11'

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

#%% Test image
points_dict = {}
for file in tqdm(os.listdir(f'{RAW}/landowners_1873/images')):
    page = file.replace('page_', '').replace('.png', '')
    if 'bold_' in file or '_lines' in file or '.png' not in file:
        continue
    if 'page_303' in file:
        cutoff = 3000
    else:
        cutoff = 15000
    img = cv2.imread(f'{RAW}/landowners_1873/images/{file}', cv2.IMREAD_GRAYSCALE)


    x_min = 0
    x_max = img.shape[1]
    y_min = 0
    y_max = img.shape[0]

    # Use Hough Lines to find near-horizontal lines and rotate the image
    edge = cv2.Canny(img, 100, 200, apertureSize=3)
    lines = cv2.HoughLines(edge[:cutoff, :], 1, np.pi / 180, threshold=500)
    horiz_lines = []
    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        edge_points = []
        if x2-x1 == 0:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        for x in [x_min, x_max]:
            y = slope * (x - x0) + y0
            if y_min <= y <= y_max:
                edge_points.append((x, int(y)))
        for y in [y_min, y_max]:
            x = (y - y0) / slope + x0
            if x_min <= x <= x_max:
                edge_points.append((int(x), y))
        if (edge_points[0][1] > img.shape[0] - 500 and edge_points[1][1] > img.shape[0] - 500) or \
                (edge_points[0][1] < 500 and edge_points[1][1] < 500):
            continue
        if (edge_points[0][0] > img.shape[1] - 500 and edge_points[1][0] > img.shape[1] - 500) or \
                (edge_points[0][0] < 500 and edge_points[1][0] < 500):
            continue
        if abs(slope) < .5:
            horiz_lines.append(edge_points)

    # Create list of angles
    angle_list = []
    for line in horiz_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        if x2-x1 == 0:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        angle = np.arctan(slope) * 180 / np.pi
        angle_list.append(angle)
    angles = np.array(angle_list)
    mean_angle = float(np.mean(angles))
    # Rotate the image
    img_rotated = ndimage.rotate(img, mean_angle)
    y_crop = (img_rotated.shape[0] - img.shape[0])
    x_crop = (img_rotated.shape[1] - img.shape[1])
    img_rotated = img_rotated[int(y_crop):int(img_rotated.shape[0]-y_crop),
                                int(x_crop):int(img_rotated.shape[1]-x_crop)]

    plt.imshow(img_rotated, cmap='gray')
    img_rotated = img_rotated[200:img_rotated.shape[0]-500, 200:img_rotated.shape[1]-200]
    plt.imshow(img_rotated, cmap='gray')
    cv2.imwrite(f'{RAW}/landowners_1873/rotated_images/rotated_{file}', img_rotated)