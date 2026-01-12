import os
import re
import ast
import json
import torch
import shutil
import random
import platform
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import phonetics as ph
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
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
MODELS = f'Code/ml_models/'
MODEL_FOLDER = f'{MODELS}/name_matcher'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
#%%
class CrossEncoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, fc_dim=32):
        super(CrossEncoder, self).__init__()

        # Embedding layer (shared between names and metaphones)
        self.name_embedding = nn.Embedding(28, embed_dim)  # Assuming 27 letters + 1 padding
        self.metaphone_embedding = nn.Embedding(28, embed_dim)
        # BiLSTM for sequence encoding
        self.name_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.metaphone_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(4*2 * hidden_dim, fc_dim)  # Combining all four encodings
        self.fc2 = nn.Linear(fc_dim, 1)

    def name_encode(self, x):
        name_embedded = self.name_embedding(x)
        _, (name_hidden, _) = self.name_lstm(name_embedded)
        name_hidden = torch.cat((name_hidden[0], name_hidden[1]), dim=1)  # Concatenate forward & backward LSTM outputs
        return name_hidden
    def metaphone_encode(self, x):
        metaphone_embedded = self.metaphone_embedding(x)
        _, (metaphone_hidden, _) = self.metaphone_lstm(metaphone_embedded)
        metaphone_hidden = torch.cat((metaphone_hidden[0], metaphone_hidden[1]), dim=1)  # Concatenate forward & backward LSTM outputs
        return metaphone_hidden

    def forward(self, name1, metaphone1, name2, metaphone2):
        # Encode each input separately
        name1_encoded = self.name_encode(name1)
        metaphone1_encoded = self.metaphone_encode(metaphone1)
        name2_encoded = self.name_encode(name2)
        metaphone2_encoded = self.metaphone_encode(metaphone2)
        # Concatenate all representations
        combined = torch.cat((name1_encoded, metaphone1_encoded, name2_encoded, metaphone2_encoded), dim=1)

        # Fully connected layers
        fc1_out = self.fc1(combined)
        fc1_relud = F.relu(fc1_out)
        output = torch.sigmoid(self.fc2(fc1_relud))  # Binary classification

        return output

#%%
# Load training data
if (os.path.exists(f'{PROCESSED}/surname_training_pairs.csv')):
    training_data = pd.read_csv(f'{PROCESSED}/surname_training_pairs.csv')
    surname_pairs = [x for x in zip(training_data['name_1'], training_data['name_2'])]
    matches = training_data['match'].tolist()
    print('Training data loaded!')
else:
    with open(f'{PROCESSED}/non_combined_surnames.json') as f:
        surname_lists = json.load(f)
    surname_lists = [x for x in surname_lists if len(x) > 1]

    # Combine all the lists into one big pile
    surname_pile = []
    for surname_list in surname_lists:
        surname_pile += surname_list
    surname_pile = list(set(surname_pile))
    random.shuffle(surname_pile)

    # Create EVERY matched pair so the network can learn what a match looks like
    surname_lists_copy = surname_lists.copy()
    random.shuffle(surname_lists_copy)
    surname_pairs = []
    for list in surname_lists_copy:
        for pair in itertools.combinations(list, 2):
            surname_pairs.append(pair)

    # Grab 3x as many random pairs as there are matched pairs
    combo_list = []
    combos = itertools.combinations(surname_pile, 2)
    for i,v in tqdm(enumerate(combos), total = (len(surname_pile)*(len(surname_pile)-1))/2):
        combo_list.append(v)
    selected_combos = random.sample(combo_list, len(surname_pairs)*3)
    surname_pairs += selected_combos
    matches = []
    print('Matching pairs...')
    for pair in tqdm(surname_pairs):
        if any(pair[0] in x and pair[1] in x for x in surname_lists):
            matches.append(1)
        else:
            matches.append(0)

    names_1 = [x[0] for x in surname_pairs]
    names_2 = [x[1] for x in surname_pairs]
    training_data = pd.DataFrame({'name_1': names_1, 'name_2': names_2, 'match': matches})
    training_data.to_csv(f'{PROCESSED}/surname_training_pairs.csv', index=False)

#%%

def encode_surname(surname, max_len=24):
    CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0'
    CHARSET_DICT = {char: i + 1 for i, char in enumerate(CHARSET)}
    PAD = 0
    surname = surname.upper()
    surname = ''.join([char for char in surname if char in CHARSET])
    metaphone = ph.metaphone(surname)

    encoded = [CHARSET_DICT[char] for char in surname]
    if len(encoded) < max_len:
        encoded += [PAD] * (max_len - len(encoded))
    encoded = torch.tensor(encoded).long()

    encoded_metaphone = [CHARSET_DICT[char] for char in metaphone]
    if len(encoded_metaphone) < max_len:
        encoded_metaphone += [PAD] * (max_len - len(encoded_metaphone))
    encoded_metaphone = torch.tensor(encoded_metaphone).long()

    return encoded, encoded_metaphone

#%%
# Create the DataLoader for training
encoded_surnames_1 = []
encoded_metaphones_1 = []
encoded_surnames_2 = []
encoded_metaphones_2 = []

for pair in surname_pairs:
    surname_1 = pair[0]
    surname_2 = pair[1]

    encoded_1, encoded_metaphone_1 = encode_surname(surname_1)
    encoded_2, encoded_metaphone_2 = encode_surname(surname_2)

    encoded_surnames_1.append(encoded_1)
    encoded_metaphones_1.append(encoded_metaphone_1)
    encoded_surnames_2.append(encoded_2)
    encoded_metaphones_2.append(encoded_metaphone_2)

encoded_surnames_1 = torch.stack(encoded_surnames_1)
encoded_metaphones_1 = torch.stack(encoded_metaphones_1)
encoded_surnames_2 = torch.stack(encoded_surnames_2)
encoded_metaphones_2 = torch.stack(encoded_metaphones_2)
matches = torch.tensor(matches).float()

dataset = TensorDataset(encoded_surnames_1, encoded_metaphones_1, encoded_surnames_2, encoded_metaphones_2, matches)

#%% Train, validate, test split

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print(train_dataset[0])
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True, pin_memory=True)


#%% Setting up training

epochs = 1000
# Instantiate the model
model = CrossEncoder(embed_dim=128, hidden_dim=64, fc_dim=32).to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
loss_fn = nn.BCELoss().to(device)
os.makedirs(MODEL_FOLDER, exist_ok=True)
try:
    model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'cross_encoder_1')))
    optimizer.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'cross_encoder_1_optim.pth'), map_location=device))
    scheduler.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'cross_encoder_1_sched.pth')))
    print('Model loaded :D')
except:
    print('No model found :(')

#%% Training the model
epoch_array = []
val_loss_array = []

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_dataloader):
        name1, metaphone_1, name2, metaphone_2, match = [x.to(device) for x in data]
        optimizer.zero_grad()
        output = model(name1, metaphone_1, name2, metaphone_2)
        loss = loss_fn(output, match.unsqueeze(1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            name1, metaphone_1, name2, metaphone_2, match = [x.to(device) for x in data]
            output = model(name1, metaphone_1, name2, metaphone_2)
            loss = loss_fn(output, match.unsqueeze(1))
            val_loss += loss.item()
    print(f'Train Loss: {total_loss / len(train_dataloader)}\nValidation Loss: {val_loss / len(val_dataloader)}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
    scheduler.step(val_loss/len(val_dataloader))

    epoch_array.append(epoch)
    val_loss_array.append(val_loss/len(val_dataloader))
    plt.plot(epoch_array, val_loss_array)
    plt.show()

    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1'))
        torch.save(optimizer.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1_optim.pth'))
        torch.save(scheduler.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1_sched.pth'))

torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1'))
torch.save(optimizer.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1_optim.pth'))
torch.save(scheduler.state_dict(), os.path.join(MODEL_FOLDER, 'cross_encoder_1_sched.pth'))
#%% Testing the model

model.eval()
eval_loss = 0
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        name1, metaphone_1, name2, metaphone_2, match = [x.to(device) for x in data]
        output = model(name1, metaphone_1, name2, metaphone_2)
        loss = loss_fn(output, match.unsqueeze(1))
        eval_loss += loss.item()
print(f'Evaluation Loss: {eval_loss / len(test_dataloader)}')