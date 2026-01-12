import os
import cv2
import ast
import json
import time
import shutil
import platform
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pycpd import DeformableRegistration, RigidRegistration, AffineRegistration
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pytesseract as pt
from scipy import ndimage
from PIL import Image
import easyocr
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseract_config = r'--oem 3 --psm 11'

reader = easyocr.Reader(['en'])

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

#%% Test image
page_count = 0
points_dict = {}
count = 0
for file in tqdm(os.listdir(f'{RAW}/landowners_1873/rotated_images')):
    count += 1
    if count > 1:
        break
    page = file.replace('page_', '').replace('.png', '')
    if 'bold_' in file or '_lines' in file or '.png' not in file:
        continue
    if page_count > 10:
        break
    page_count += 1
    img = cv2.imread(f'{RAW}/landowners_1873/rotated_images/{file}', cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if 'page_303' in file:
        cutoff = [900, 1100]
    else:
        cutoff = [500, 1000]


    x_min = 0
    x_max = img.shape[1]
    y_min = 0
    y_max = img.shape[0]
    full_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    full_image2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Detect text, then turn bounding box to white
    pil_img = Image.fromarray(img)
    boxes = pt.image_to_data(pil_img, output_type=pt.Output.DATAFRAME, config=tesseract_config)
    boxes = boxes[boxes.conf > 25]
    boxes = boxes[boxes.level == 5]
    boxes = boxes[boxes.text != '']
    boxes = boxes[boxes.width < 1000]
    boxes = boxes[boxes.height < 1000]
    for i, row in boxes.iterrows():
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        img[y:y + h, min(x+2, img.shape[1]):min(x + w - 2, img.shape[1])] = 255
    plt.imshow(img, cmap='gray')
    plt.show()
    new_file = file.replace('.png', '_text_removed.png')
    cv2.imwrite(f'{RAW}/landowners_1873/text_removed/{new_file}', img)
    edg = cv2.Canny(img, 100, 200, apertureSize=3)

    threshold = 500
    found_lines = False
    saved_vert = []
    saved_horiz = []

    while not found_lines:

        lines = cv2.HoughLines(edg[cutoff[0]:cutoff[1], :], 1, np.pi / 180, threshold=threshold)
        vert_lines = []
        horiz_lines = []
        for line in lines:
            rho, theta = line[0]
            # Eliminating lines that are nearly 45 degrees
            if (np.pi/4)*.7 < theta < (np.pi/4)*1.3 or (np.pi/4*3)*.7 < theta < (np.pi/4*3)*1.3:
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if x2-x1 == 0:
                slope = np.inf
            else:
                slope = (y2 - y1) / (x2 - x1)
            edge_points = []
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

            # Sort mostly-vertical vs mostly-horizontal lines
            if abs(np.tan(theta)) < 1:
                vert_lines.append(edge_points)
            else:
                horiz_lines.append(edge_points)

        # SORT lines by x coord for vertical lines and y coord for horizontal lines
        vert_lines = sorted(vert_lines, key=lambda x: x[0][0])
        horiz_lines = sorted(horiz_lines, key=lambda x: x[0][1])
        # Eliminate lines that start or end within 100 pixels of each other
        thinned_vert = []
        thinned_horiz = []

        for i, line in enumerate(vert_lines):
            if line[0][0] < 900:
                continue
            if i == 0:
                thinned_vert.append(line)
                continue
            if abs(line[0][0] - vert_lines[i-1][0][0]) > 100 and abs(line[1][0] - vert_lines[i-1][1][0]) > 100:
                thinned_vert.append(line)
        for i, line in enumerate(horiz_lines):
            if i == 0:
                thinned_horiz.append(line)
                continue
            if abs(line[0][1] - horiz_lines[i-1][0][1]) > 100 and abs(line[1][1] - horiz_lines[i-1][1][1]) > 100:
                thinned_horiz.append(line)

        vert_lines = thinned_vert
        horiz_lines = thinned_horiz

        if len(vert_lines) >= 7 and len(saved_vert) == 0:
            saved_vert = vert_lines
        if len(horiz_lines) >= 2 and len(saved_horiz) == 0:
            saved_horiz = horiz_lines
        if len(saved_horiz) > 0 and len(saved_vert) > 0:
            found_lines = True
        else:
            threshold = int(threshold * 0.9)

    lines = saved_vert + saved_horiz

    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.line(full_image, line[0], line[1], (0, 0, 255), 8)

    intersections = []
    for line in lines:
        intersections.append(line[0])
        intersections.append(line[1])
    # Find all intersections
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
            try:
                slope_1 = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                slope_1 = np.inf
            try:
                slope_2 = (y4 - y3) / (x4 - x3)
            except ZeroDivisionError:
                slope_2 = np.inf

            if slope_1 == slope_2:
                # lines are parallel
                continue

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                # still need to find intersections, even when there's one vertical or horizontal line
                if slope_1 == np.inf:
                    # line 1 is vertical
                    px = x1
                    py = slope_2 * (px - x3) + y3
                elif slope_2 == np.inf:
                    # line 2 is vertical
                    px = x3
                    py = slope_1 * (px - x1) + y1
                elif slope_1 == 0:
                    # line 1 is horizontal
                    py = y1
                    px = (py - y3) / slope_2 + x3
                elif slope_2 == 0:
                    # line 2 is horizontal
                    py = y3
                    px = (py - y1) / slope_1 + x1

            else:
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            if 0 <= px <= img.shape[1] and 0 <= py <= img.shape[0]:
                intersections.append((int(px), int(py)))

    cv2.line(full_image, (0,0), (0, img.shape[0]), (0, 0, 255), 8)
    cv2.line(full_image, (img.shape[1], 0), (img.shape[1], img.shape[0]) , (0, 0, 255), 8)
    cv2.line(full_image, (0, img.shape[0]), (img.shape[1], img.shape[0]), (0, 0, 255), 8)
    cv2.line(full_image, (0, 0), (img.shape[1], 0), (0, 0, 255), 8)
    # Draw intersections
    for point in intersections:
        cv2.circle(full_image, point, 20, (0, 255, 0), -1)



    plt.imshow(full_image)
    plt.show()

#%% Point cloud deformation babey
    with open(f'{RAW}/landowners_1873/intersections.json', 'r') as f:
        template = json.load(f)
    template_ymax = 7223.0
    template_xmax = 5112.0
    template = [x for x in template if x[1] > 0 and x[1] < template_ymax]
    template = [x for x in template if x[0] > 0 and x[0] < template_xmax]

    intersections = [x for x in intersections if x[1] > 0 and x[1] < img.shape[0]]
    intersections = [x for x in intersections if x[0] > 0 and x[0] < img.shape[1]]
    target = np.array(intersections, dtype='float32')
    # Transform template to same dimensions as target
    template = np.array(template, dtype='float32')
    template[:, 0] = template[:, 0] / template_xmax * img.shape[1]
    template[:, 1] = template[:, 1] / template_ymax * img.shape[0]


    reg = RigidRegistration(X=target, Y=template, alpha=3.0, beta=1.5)
    # run the registration & collect the results
    rigid, _ = reg.register()
    reg = DeformableRegistration(X=target, Y=rigid, alpha=3.0, beta=1.5)
    a, _ = reg.register()
    # convert to list of tuples
    a = [tuple(map(int, point)) for point in a]
    points_dict[page] = a
    cv2.line(full_image2, a[0], a[7], (0, 0, 255), 8)
    cv2.line(full_image2, a[1], a[8], (0, 0, 255), 8)
    cv2.line(full_image2, a[2], a[9], (0, 0, 255), 8)
    cv2.line(full_image2, a[3], a[10], (0, 0, 255), 8)
    cv2.line(full_image2, a[4], a[11], (0, 0, 255), 8)
    cv2.line(full_image2, a[5], a[12], (0, 0, 255), 8)
    cv2.line(full_image2, a[6], a[13], (0, 0, 255), 8)
    cv2.line(full_image2, a[0], a[6], (0, 0, 255), 8)
    cv2.line(full_image2, a[7], a[13], (0, 0, 255), 8)
    for point in a:
        cv2.circle(full_image2, point, 20, (0, 255, 0), -1)
    plt.imshow(full_image2)
    plt.show()
    registered_filename = file.replace('.png', '_lines_registered.png')
    cv2.imwrite(f'{RAW}/landowners_1873/lines/{registered_filename}', full_image2)

with open(f'{RAW}/landowners_1873/intersections_registered.json', 'w') as f:
    json.dump(points_dict, f, indent=4)