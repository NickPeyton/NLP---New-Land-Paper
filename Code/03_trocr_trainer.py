# Imports
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
from torchmetrics.text import CharErrorRate as cer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator, AutoTokenizer
import platform
import random
import matplotlib.pyplot as plt

if platform.node() == 'Nick_Laptop':
    drive = 'C'
elif platform.node() == 'MSI':
    drive = 'D'
else:
    drive = 'uhhhhhh'
    print('Uhhhhhhhhhhhhh')
os.chdir(f'{drive}:/PhD/DissolutionProgramming/LND---Land-Paper')

model_size = 'base'
print(f'Using model size: {model_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

RAW = 'Data/Raw'
PROCESSED = 'Data/Processed'
MODELS = f'Code/ml_models/trocr_model_{model_size}'

random.seed(0)
torch.manual_seed(0)


tdf = pd.DataFrame()
for subsidy in [
    1524,
    1543,
    # 1581,
    # 1642,
    # 1647,
    # 1660,
    # 1674
    ]:
    TEXT_FOLDER = f'Data/Processed/subsidy{subsidy}/text_pages'
    sdf = pd.read_csv(f'{PROCESSED}/subsidy{subsidy}/subsidy{subsidy}_lines.csv', encoding='utf-8')
    sdf['subsidy'] = subsidy
    sdf = sdf[sdf['text'] != '']
    sdf = sdf[sdf['text'].notnull()]
    tdf = pd.concat([tdf, sdf])
#%%
tdf.reset_index(drop=True, inplace=True)
tdf['text'] = tdf['text'].apply(lambda x: str(x).strip())
max_text_length = int(tdf['text'].apply(len).max() + 10)
prev_page = 0
for i, row in tqdm(tdf.iterrows(), total=len(tdf)):
    page = row['page']
    subsidy = row['subsidy']
    if page != prev_page:
        page_str = str(page).zfill(3)
        page_image_path = f'{PROCESSED}/subsidy{subsidy}/processed_pages/processed_subsidy{subsidy}_page_{page_str}.png'
        page_image = Image.open(page_image_path)
        # To color for idiotic reasons
        page_image = page_image.convert('RGB')
        prev_page = page
        page_image_array = np.array(page_image)
    x1 = int(row['x1'])
    x2 = int(row['x2'])
    y1 = int(row['y1'])
    y2 = int(row['y2'])
    line_image = page_image.crop((x1, y1, x2, y2))
    tdf.at[i, 'line_image'] = line_image

train_df = tdf.sample(frac=0.8, random_state=0)
eval_df = tdf.drop(train_df.index)

#%% Defining Loader Class

class tsaLoader(Dataset):
    def __init__(self, df, feature_extractor, tokenizer, max_target_length=max_text_length):
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        image = self.df.iloc[idx]['line_image']

        # Use feature_extractor for images
        pixel_values = self.feature_extractor(images=image, return_tensors='pt').pixel_values

        text = str(text) if text is not None else ''

        # Use tokenizer for text
        labels = self.tokenizer(text, padding='max_length', max_length=self.max_target_length).input_ids
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        attention_mask = [1 if label != self.tokenizer.pad_token_id else 0 for label in labels]

        encoding = {'pixel_values': pixel_values.squeeze(),
                    'labels': torch.tensor(labels),
                    'attention_mask': torch.tensor(attention_mask)
                    }
        return encoding

#%%

try:
    checkpoint_dict = {}
    if not os.path.isdir(MODELS):
        os.mkdir(MODELS)

    for folder in os.listdir(MODELS):
        modified_time = os.path.getmtime(f'{MODELS}/{folder}')
        checkpoint_dict[modified_time] = folder
    latest_checkpoint = checkpoint_dict[max(checkpoint_dict.keys())]
    latest_model_location = os.path.join(MODELS, latest_checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(latest_model_location,
                                                      cache_dir=latest_model_location,
                                                      local_files_only=True)
    print('Latest Model Loaded!')
    for folder in os.listdir(MODELS):
        if folder != latest_checkpoint:
            shutil.rmtree(f'{MODELS}/{folder}')
    print('Older models vaporized!')
except:
    model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/trocr-{model_size}-printed')
    print('Model Downloaded')

# Processor and tokenizer
try:
    processor = TrOCRProcessor.from_pretrained(f'{MODELS}/{latest_checkpoint}', use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(f'{MODELS}/{latest_checkpoint}', use_fast=False)
    print('Processor and tokenizer loaded from local files')
except:
    processor = TrOCRProcessor.from_pretrained(f'microsoft/trocr-{model_size}-printed', use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(f'microsoft/trocr-{model_size}-printed', use_fast=False)
    print('Processor and tokenizer downloaded')

model.to(device)

#%% Creating Datasets
train_dataset = tsaLoader(df=train_df,
                          feature_extractor=processor,
                          tokenizer=tokenizer)
print(f'Length of training dataset: {train_dataset.__len__()}')
eval_dataset = tsaLoader(df=eval_df,
                        feature_extractor=processor,
                        tokenizer=tokenizer)
print(f'Length of evaluation dataset: {eval_dataset.__len__()}')

#%% Setting parameters for our tokenizer and model

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.remove_invalid_tokens = True
#model.config.vocab_size = len(tokenizer)

# set beam search parameters
model.config.max_length = max_text_length + 2
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

#%% Configuring the Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy='steps',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    output_dir=MODELS,
    logging_steps=100,
    save_steps=100,
    eval_steps=1000,
    learning_rate=5e-5,
    num_train_epochs=100,
)

#%% Setting up metrics

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer_metric = cer()
    char_error = cer_metric(pred_str, label_str)

    return {'cer': char_error}


#%% Setting up the trainer

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

#%% TRAINING

trainer.train()

#%% Removing garbage
checkpoint_dict = {}
for folder in os.listdir(MODELS):
    modified_time = os.path.getmtime(f'{MODELS}/{folder}')
    checkpoint_dict[modified_time] = folder
latest_checkpoint = checkpoint_dict[max(checkpoint_dict.keys())]
latest_model_location = os.path.join(MODELS, latest_checkpoint)

for folder in os.listdir(MODELS):
    if folder != latest_checkpoint:
        shutil.rmtree(f'{MODELS}/{folder}')
print('Older models vaporized!')