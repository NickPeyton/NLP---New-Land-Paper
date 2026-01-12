import os

import torch
from torch.amp import autocast
from torch.nn import Module as NNModule
import torchvision

from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import cv2
import random
import matplotlib.pyplot as plt
import platform
if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
subsidy = 1524

print(f'Using subsidy: {subsidy}')

os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')
LABEL_FOLDER = f'Data/Processed/subsidy{subsidy}/label_pages'
IMAGE_FOLDER = f'Data/Processed/subsidy{subsidy}/little_pages'
MODELS = f'Code/ML Models/doc_ufcn_model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


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

#%% Defining shit

class doc_ufcn(NNModule):

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
        with autocast(device, enabled=self.amp):
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


def rectangle_mask(mask_predicted):
    mask_predicted = mask_predicted.squeeze(0)
    mask_predicted = mask_predicted.clone().to('cpu').numpy()
    final_mask = np.zeros_like(mask_predicted)
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
        final_mask[i] = mask_layer
    final_mask = torch.tensor(final_mask,
                             dtype=torch.float32,
                             device=device)
    return final_mask

#%% Creating image and label lists


label_list = []
image_list = []
for filename in os.listdir(LABEL_FOLDER):
    if not filename.endswith('.png'):
        continue
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = torchvision.io.read_image(image_path)
    image = image / 255.0
    image = image.float()
    image = image.unsqueeze(0)
    image = image.to(device=device)

    label_path = os.path.join(LABEL_FOLDER, filename)
    label = Image.open(label_path)
    label = label.convert("RGB")
    label_array = np.array(label)

    masks = []
    for i in class_numbers:
        mask = np.all(label_array == class_colors[i], axis=-1).astype(np.uint8)
        masks.append(mask)
    # Mask to tensor
    masks_array = np.stack(masks, axis=-1)
    # Convert to tensor
    label_tensor = torch.from_numpy(masks_array).float()
    label_tensor = label_tensor.permute(2, 0, 1).to(device)
    label_tensor = label_tensor.half()
    image_list.append(image)
    label_list.append(label_tensor)

train_image_list = image_list[:int(len(image_list) * 0.8)]
train_dict = {image_list[i]: label_list[i] for i in range(len(train_image_list))}
test_image_list = image_list[int(len(image_list) * 0.8):]
test_mask_list = label_list[int(len(image_list) * 0.8):]
test_dict = {image_list[i]: label_list[i] for i in range(len(test_image_list))}
print('Image and label lists created!')

#%% Setup for training
epochs = 30000
model = doc_ufcn(len(class_numbers), use_amp=False)
model.to(device=device)
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=.01)

weight = torch.tensor([1, 1, 1, 1, 5, 1]).to(dtype=torch.float, device=device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight).to(device=device)
try:
    model.load_state_dict(torch.load(os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcn.pth'), weights_only=True))
    optimizer.load_state_dict(torch.load(os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcnOptim.pth'), weights_only=True))
    print('Model loaded :D')
except:
    try:
        model.load_state_dict(torch.load(os.path.join(MODELS,'subsidy1543_doc_ufcn.pth'), weights_only=True))
        optimizer.load_state_dict(torch.load(os.path.join(MODELS,'subsidy1543_doc_ufcnOptim.pth'), weights_only=True))
        print('1543 Model loaded :D')
    except:
        try:
            model.load_state_dict(torch.load(os.path.join(MODELS,'subsidy1524_doc_ufcn.pth'), weights_only=True))
            optimizer.load_state_dict(torch.load(os.path.join(MODELS,'subsidy1524_doc_ufcnOptim.pth'), weights_only=True))
            print('1524 Model loaded :D')
        except:
            print('ALERT, NO MODEL LOADED')


#%% Training

for epoch in tqdm(range(epochs)):
    shuffled_image_list = train_image_list
    random.shuffle(shuffled_image_list)
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    model.train()
    for image in shuffled_image_list:
        mask = train_dict[image]
        # 1. Forward pass
        predicted_mask = model(image)
        # 2. Calculate loss
        loss = loss_fn(predicted_mask, mask.unsqueeze(0))
        train_loss += loss.item()
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()

    # Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for test_image, test_mask in zip(test_image_list, test_mask_list):
            # 1. Forward pass
            test_pred = model(test_image)
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, test_mask.unsqueeze(0)).item()

    # Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}\n")
    if epoch % 50 == 0:
        # Save the model
        torch.save(model.state_dict(), os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcn.pth'))
        old_state_dict = model.state_dict()
        torch.save(optimizer.state_dict(), os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcnOptim.pth'))

        # Display a predicted mask
        rectangle_pred = rectangle_mask(test_pred)
        test_pred = test_pred.squeeze(0)
        for i in range(test_pred.shape[0]):
            plt.imshow(test_pred[i].to('cpu').numpy(), cmap='gray')
            plt.show()

torch.save(model.state_dict(), os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcn.pth'))
old_state_dict = model.state_dict()
torch.save(optimizer.state_dict(), os.path.join(MODELS, f'subsidy{subsidy}_doc_ufcnOptim.pth'))
