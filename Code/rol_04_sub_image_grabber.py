#%%
import os
import json
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
IMAGES = f'{RAW}/landowners_1873/rotated_images'

#%%

with open(f'{RAW}/landowners_1873/intersections_registered.json', 'r') as f:
    intersection_dict = json.load(f)
new_intersection_dict = {}
count = 0
for file in tqdm(os.listdir(f'{IMAGES}')):
    page = file.replace('rotated_page_', '').replace('.png', '')
    intersections = intersection_dict[page]
    new_intersections = [tuple(x) for x in intersections]
    count += 1

    if not file.endswith('.png'):
        continue
    image = cv2.imread(f'{IMAGES}/{file}')
    vert_line_list = [
    [intersections[0], intersections[7]],
    [intersections[1], intersections[8]],
    [intersections[2], intersections[9]],
    [intersections[3], intersections[10]],
    [intersections[4], intersections[11]],
    [intersections[5], intersections[12]],
    [intersections[6], intersections[13]]
        ]
    horiz_line_list = [
    [intersections[0], intersections[6]],
    [intersections[7], intersections[13]]
        ]
    new_vert_lines = []
    for line in vert_line_list:
        x1, y1 = line[0]
        x2, y2 = line[1]
        if x1 == x2:
            new_vert_lines.append([(x1, 0), (x1, image.shape[0])])
            new_intersections.append((x1, 0))
            new_intersections.append((x1, image.shape[0]))
        else:
            slope = (y2 - y1) / (x2 - x1)
            # Calculate x value for y = 0 and y = image.shape[0]
            x_at_0 = (0 - y1) / slope + x1
            x_at_max = (image.shape[0] - y1) / slope + x1
            new_vert_lines.append([(x_at_0, 0), (x_at_max, image.shape[0])])
            new_intersections.append((x_at_0, 0))
            new_intersections.append((x_at_max, image.shape[0]))

    new_horiz_lines = []
    for line in horiz_line_list:
        x1, y1 = line[0]
        x2, y2 = line[1]
        if y1 == y2:
            new_horiz_lines.append([(0, y1), (image.shape[1], y1)])
            new_intersections.append((0, y1))
            new_intersections.append((image.shape[1], y1))
        else:
            slope = (x2 - x1) / (y2 - y1)
            y_at_0 = (0 - x1) / slope + y1
            y_at_max = (image.shape[1] - x1) / slope + y1
            new_horiz_lines.append([(0, y_at_0), (image.shape[1], y_at_max)])
            new_intersections.append((0, y_at_0))
            new_intersections.append((image.shape[1], y_at_max))


    # sort new intersections by y, then by x
    new_intersections = list(set(new_intersections))
    new_intersections.sort(key=lambda x: (x[1], x[0]))
    new_intersection_dict[page] = new_intersections
    for line in new_vert_lines:
        cv2.line(image, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 255, 0), 8)
    for line in new_horiz_lines:
        cv2.line(image, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 0, 0), 8)
    for i, intersection in enumerate(new_intersections):
        cv2.circle(image, tuple(map(int, intersection)), 8, (0, 0, 255), -1)
        # number the intersections
        x = int(intersection[0])
        if x == image.shape[1]:
            x = int(intersection[0]) - 100
        y = int(intersection[1])
        if y == 0:
            y = int(intersection[1]) + 100
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)
    cv2.imwrite(f'{RAW}/landowners_1873/lines/lines_{file}', image)
    assert len(new_intersections) == 32
with open(f'{RAW}/landowners_1873/all_intersections.json', 'w') as f:
    json.dump(new_intersection_dict, f, indent=4)



