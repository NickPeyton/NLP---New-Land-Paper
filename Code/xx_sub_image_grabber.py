import os
import platform

import torch
from fsspec.utils import other_paths
from torch.amp import autocast
from torch.nn import Module as NNModule
import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import cv2

import pandas as pd

subsidy = 1642
print(f'Using subsidy: {subsidy}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    print('ooey gooey')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')
LITTLE_IMAGE_FOLDER = f'Data/Processed/subsidy{subsidy}/little_pages'
PROCESSED_IMAGE_FOLDER = f'Data/Processed/subsidy{subsidy}/processed_pages'
SUB_IMAGE_FOLDER = f'Data/Processed/subsidy{subsidy}/sub_images'
MODEL_FOLDER = f'Code/ML Models/doc_ufcn_model'



#%% Defining class colors and numbers


class_names = ["background",
               "left_column",
               "center_column",
               "right_column",
               "parish",
               "add_remove"]
class_colors = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 128, 0],
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0]
    ]
class_numbers = list(range(len(class_names)))

name_dict = {class_numbers[i]: class_names[i] for i in class_numbers}
color_dict = {class_numbers[i]: class_colors[i] for i in class_numbers}
print('Class colors and numbers defined!')

#%% Defining functions


class docUFCN(NNModule):

    def __init__(self, no_of_classes, use_amp=False):
        super().__init__()
        self.amp = use_amp
        self.dilated_block1 = self.dilated_block(3, 32)
        self.dilated_block2 = self.dilated_block(32, 64)
        self.dilated_block3 = self.dilated_block(64, 128)
        self.dilated_block4 = self.dilated_block(128, 256)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv_block1 = self.conv_block(256, 128)
        self.conv_block2 = self.conv_block(256, 64)
        self.conv_block3 = self.conv_block(128, 32)
        self.last_conv = torch.nn.Conv2d(64, no_of_classes, 3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def dilated_block(input_size, output_size):
        '''
        This is a dilated block, with dilation rates of 1, 2, 4, 8, 16
        '''
        modules = []
        modules.append(
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, padding=1, dilation=1, bias=False
            )
        )
        modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.Dropout(p=0.4))
        for i in [2,4,8,16]:
            modules.append(
                torch.nn.Conv2d(
                    output_size,
                    output_size,
                    3,
                    stride=1,
                    padding=i,
                    dilation=i,
                    bias=False
                )
            )
            modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Dropout(p=0.4))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def conv_block(input_size, output_size):
        '''
        This is a convolutional block w/ a convolution followed by an upsampling layer
        '''
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),

            torch.nn.ConvTranspose2d(output_size, output_size, 2, stride=2, bias=False),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
            )

    def forward(self, input_tensor):
        '''
        Define forward step of network.
        4 Successive dilated blocks followed by 3 convolutional blocks,
        one last convolution, and a softmax layer.
        '''
        with autocast(device_type=device, enabled=self.amp):
            tensor = self.dilated_block1(input_tensor)
            out_block1 = tensor
            tensor = self.dilated_block2(self.pool(tensor))
            out_block2 = tensor
            tensor = self.dilated_block3(self.pool(tensor))
            out_block3 = tensor
            tensor = self.dilated_block4(self.pool(tensor))
            tensor = self.conv_block1(tensor)
            tensor = torch.cat([tensor, out_block3], dim=1)
            tensor = self.conv_block2(tensor)
            tensor = torch.cat([tensor, out_block2], dim=1)
            tensor = self.conv_block3(tensor)
            tensor = torch.cat([tensor, out_block1], dim=1)
            output_tensor = self.last_conv(tensor)
            return self.softmax(output_tensor)


def rectangle_mask(mask_predicted, subsidy):

    mask_predicted = mask_predicted.squeeze(0)
    mask_predicted = mask_predicted.clone().to('cpu').numpy()
    final_mask = np.zeros_like(mask_predicted)
    if subsidy == 1524:
        y1 = 20
        y2 = 55
    else:
        y1 = 20
        y2 = 40
    # Manually adding the hundred row because the nn cannot figure it out
    box_df = pd.DataFrame({'x1': [105],
                           'x2': [345],
                           'y1': [y1],
                           'y2': [y2],
                           'category': ['hundred']})
    for i in range(mask_predicted.shape[0]):
        if i == 0:
            continue
        layer = mask_predicted[i]
        mask_layer = np.zeros_like(layer)
        layer = cv2.threshold(layer, 0.5, 1, cv2.THRESH_BINARY)[1]
        layer = layer.astype(np.uint8)
        contours, _ = cv2.findContours(layer, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x_list = []
            y_list = []
            contour = contour.reshape(contour.shape[0], contour.shape[2])
            for point in contour:
                x_list += [point[0]]
                y_list += [point[1]]
            x_max = max(x_list)
            x_min = min(x_list)
            y_max = max(y_list)
            y_min = min(y_list)
            if x_max - x_min < 50 or y_max - y_min < 10:
                continue
            mask_layer[y_min:y_max, x_min:x_max] = 1
            box_df = pd.concat([box_df, pd.DataFrame({'x1': [x_min],
                                                      'x2': [x_max],
                                                      'y1': [y_min],
                                                      'y2': [y_max],
                                                      'category': [name_dict[i]]})])
        final_mask[i] = mask_layer
    final_mask = torch.tensor(final_mask,
                             dtype=torch.float32,
                             device=device)
    return final_mask, box_df

def df_sorter(df):
    for i, row in df.iterrows():
        if 'hundred' in row['category']:
            continue
        if i < 2:
            continue
        # These three checks make sure we're looking at the center of a group of 3 rows representing
        # left, center, and right columns, the "try" makes sure we don't get weird shit at start/end
        above2_cat = df.iloc[i-2]['category']
        above_cat = df.iloc[i-1]['category']
        current_cat = df.iloc[i]['category']
        cat_list = [above2_cat, above_cat, current_cat]
        # Code from when I didn't just have the columns, may use later

        if 'column' not in above2_cat:
            continue
        if 'column' not in current_cat:
            continue
        # Just in case
        if 'column' not in above_cat:
            continue

        # Catching and ignoring triples with multiple of the same category
        if len(set(cat_list)) != 3:
            continue
        # Catching and ignoring triples that aren't within a certain y-value of each other
        if abs(df.iloc[i-2]['y1'] - df.iloc[i]['y1']) > 10:
            continue
        if abs(df.iloc[i-1]['y1'] - df.iloc[i]['y1']) > 10:
            continue
        # Splitting the df around the column triple
        topdf = df.iloc[:i - 2].copy()
        centerdf = df.iloc[i - 2:i + 1].copy()
        bottomdf = df.iloc[i + 1:].copy()
        # Creating a special var to sort the center df
        centerdf['special_category'] = pd.Categorical(centerdf['category'],
                                                      categories=['left_column',
                                                                  'center_column',
                                                                  'right_column'],
                                                      ordered=True)
        centerdf = centerdf.sort_values(by='special_category')
        centerdf = centerdf.drop(columns=['special_category'])
        # Putting the df back together
        df = pd.concat([topdf, centerdf, bottomdf])
        df.reset_index(drop=True, inplace=True)
    return df


#%% Load model and prepare for inference
try:
    model = docUFCN(len(class_numbers), use_amp=False)
    model.load_state_dict(torch.load(f'{MODEL_FOLDER}/subsidy{subsidy}_doc_ufcn.pth'))
    model.to(device)
    model.eval()
    print('Model loaded and ready for inference!')
except:
    if subsidy <= 1581:
        other_subsidy = 1543
    elif subsidy > 1581 and subsidy < 1674:
        other_subsidy = 1581
    else:
        other_subsidy = 1543
    model = docUFCN(len(class_numbers), use_amp=False)
    model.load_state_dict(torch.load(f'{MODEL_FOLDER}/subsidy{other_subsidy}_doc_ufcn.pth'))
    model.to(device)
    model.eval()
    print('OTHER model loaded and ready for inference!')



#%% Remove all sub-images
os.makedirs(SUB_IMAGE_FOLDER, exist_ok=True)

for file in os.listdir(SUB_IMAGE_FOLDER):
    print(file)
    os.remove(os.path.join(SUB_IMAGE_FOLDER, file))
#%% Inference
sub_image_df = pd.DataFrame()
for image_file in tqdm(os.listdir(LITTLE_IMAGE_FOLDER)):
    # Get last 3 digits of image filename (cutting off the '.png')
    page = image_file[-7:-4]
    # Loading the little image and running it through the model
    image_path = os.path.join(LITTLE_IMAGE_FOLDER, image_file)
    image = Image.open(image_path)
    torch_image = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = model(torch_image)

    # Getting the mask and the bounding boxes
    rect_mask, box_df = rectangle_mask(pred_mask, subsidy)

    # Sorting the box dataframe
    box_df = box_df.sort_values(by='y1')
    box_df.reset_index(drop=True, inplace=True)
    page_df = df_sorter(box_df)


    # Getting the processed image path
    processed_image_path = os.path.join(PROCESSED_IMAGE_FOLDER, image_file.replace('little_', 'processed_'))
    processed_image = torchvision.io.read_image(processed_image_path)
    processed_image = processed_image.to(device=device, dtype=torch.float).squeeze(0)

    page_df['page'] = page
    subimage_num = 0
    for i, row in page_df.iterrows():
        if (row['category'] == 'center_column' or row['category'] == 'right_column') and page_df.iloc[i-1]['category'] == 'hundred':
            page_df.drop(i, inplace=True)
            print(f'Dropped {i}')
            continue
        x1 = row['x1']
        x2 = row['x2']
        y1 = row['y1']
        y2 = row['y2']
        category = row['category']

        # Getting the transformed x and y values
        x1_new = int(x1 * processed_image.shape[1] / image.width)
        x2_new = int(x2 * processed_image.shape[1] / image.width)
        y1_new = int(y1 * processed_image.shape[0] / image.height)
        y2_new = int(y2 * processed_image.shape[0] / image.height)

        # Removing everything outside the resized bounding box
        subimage = processed_image[y1_new:y2_new, x1_new:x2_new]

        # Saving the subimage
        subsidy_name = image_file.replace('little_', '')[:11]

        subimage_name = f'{subsidy_name}_page_{page}_{subimage_num}_{category}.png'
        subimage_path = os.path.join(SUB_IMAGE_FOLDER, subimage_name)
        torchvision.utils.save_image(subimage, subimage_path, normalize=True)
        subimage_num += 1
    page_df = page_df[page_df['category']!='add_remove']
    page_df.reset_index(drop=True, inplace=True)
    page_df['annotation'] = page_df.index
    sub_image_df = pd.concat([sub_image_df, page_df])


#%% Optional Hundred Remover

for file in os.listdir(SUB_IMAGE_FOLDER):
    if 'hundred' in file:
        os.remove(os.path.join(SUB_IMAGE_FOLDER, file))
print('Hundred sub-images removed!')