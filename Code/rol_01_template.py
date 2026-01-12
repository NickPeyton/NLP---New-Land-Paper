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

with open(f'{RAW}/landowners_1873/page_307_lines.txt', 'r') as f:
    text = f.read()
text = text.replace('\n', ',')
lines = ast.literal_eval(text)

#%%
x_max = 5112.00
y_max = 7223.00
# Find intersections of lines
intersections = []
for i, line in enumerate(lines):
    for line2 in lines[i+1:]:
        x1, y1 = line[0]
        x2, y2 = line[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        if 0 <= x <= x_max and 0 <= y <= y_max:
            intersections.append((x, y))

# Add endpoints of lines to intersections
for line in lines:
    x1, y1 = line[0]
    x2, y2 = line[1]
    if 0 <= x1 <= x_max and 0 <= y1 <= y_max:
        intersections.append((x1, y1))
    if 0 <= x2 <= x_max and 0 <= y2 <= y_max:
        intersections.append((x2, y2))
# Remove duplicates
intersections = list(set(intersections))
# Sort intersections by y, then by x
intersections.sort(key=lambda x: (x[1], x[0]))

# Plot intersections and lines on image
img = cv2.imread(f'{RAW}/landowners_1873/page_307.png', cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for line in lines:
    cv2.line(img, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 255, 0), 8)

for i, intersection in enumerate(intersections):
    cv2.circle(img, tuple(map(int, intersection)), 8, (0, 0, 255), -1)
    # number the intersections
    x = int(intersection[0])
    if x == x_max:
        x = int(intersection[0]) - 100
    y = int(intersection[1])
    if y == 0:
        y = int(intersection[1]) + 100
    cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)

plt.imshow(img)
plt.show()
cv2.imwrite(f'{RAW}/landowners_1873/TEMPLATE.png', img)

#%% Save intersections to .json file
with open(f'{RAW}/landowners_1873/intersections.json', 'w') as f:
    json.dump(intersections, f, indent=4)

#%% Grab the almost-rectangles and transform them into rectangles
img = cv2.imread(f'{RAW}/landowners_1873/page_307.png', cv2.IMREAD_GRAYSCALE)
name_heading_rect_1 = np.array([intersections[8], intersections[9], intersections[18], intersections[19]], dtype='float32')
address_heading_rect_1 = np.array([intersections[9], intersections[10], intersections[19], intersections[20]], dtype='float32')
extent_heading_rect_1 = np.array([intersections[10], intersections[11], intersections[20], intersections[21]], dtype='float32')
rental_heading_rect_1 = np.array([intersections[11], intersections[12], intersections[21], intersections[22]], dtype='float32')

name_entry_rect_1 = np.array([intersections[18], intersections[19], (0, img.shape[0]), intersections[28]], dtype='float32')
address_entry_rect_1 = np.array([intersections[19], intersections[20], intersections[28], intersections[29]], dtype='float32')
extent_entry_rect_1 = np.array([intersections[20], intersections[21], intersections[29], intersections[30]], dtype='float32')
rental_entry_rect_1 = np.array([intersections[21], intersections[22], intersections[30], intersections[31]], dtype='float32')

name_heading_rect_2 = np.array([intersections[13], intersections[14], intersections[23], intersections[24]], dtype='float32')
address_heading_rect_2 = np.array([intersections[14], intersections[15], intersections[24], intersections[25]], dtype='float32')
extent_heading_rect_2 = np.array([intersections[15], intersections[16], intersections[25], intersections[26]], dtype='float32')
rental_heading_rect_2 = np.array([intersections[16], intersections[17], intersections[26], intersections[27]], dtype='float32')

name_entry_rect_2 = np.array([intersections[23], intersections[24], intersections[32], intersections[33]], dtype='float32')
address_entry_rect_2 = np.array([intersections[24], intersections[25], intersections[33], intersections[34]], dtype='float32')
extent_entry_rect_2 = np.array([intersections[25], intersections[26], intersections[34], intersections[35]], dtype='float32')
rental_entry_rect_2 = np.array([intersections[26], intersections[27], intersections[35], (img.shape[1], img.shape[0])], dtype='float32')

rect_list = [
    name_heading_rect_1,
    name_entry_rect_1,
    address_heading_rect_1,
    address_entry_rect_1,
    extent_heading_rect_1,
    extent_entry_rect_1,
    rental_heading_rect_1,
    rental_entry_rect_1,
    name_heading_rect_2,
    name_entry_rect_2,
    address_heading_rect_2,
    address_entry_rect_2,
    extent_heading_rect_2,
    extent_entry_rect_2,
    rental_heading_rect_2,
    rental_entry_rect_2
]
for rect in rect_list:
    width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[3] - rect[2]))
    height = max(np.linalg.norm(rect[2] - rect[0]), np.linalg.norm(rect[3] - rect[1]))

    dest_rect = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dest_rect)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    plt.imshow(warped, cmap='gray')
    plt.show()