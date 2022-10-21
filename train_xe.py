import numpy as np
import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import json
from datetime import datetime
import random 
import datasets

model_name = 'google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name) #model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #selecting gpu
model.to(device)

with open('/qrecc/qrecc_train.json') as file: #open qrecc training set
    train_data = json.load(file)

random.seed(42) #seed for reproducibility

random_indexes = random.sample(range(len(train_data)),int(len(train_data)*0.2)) #selecting indexes to create validation set

val_data=[]
for rnd in random_indexes:
  val_data.append(train_data[rnd])
  train_data[rnd] = None

train_data = [i for i in train_data if i]

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, file):
        data = file   
        self.texts=[]
        self.labels=[]
        for sample in data:
            sample['Context'].reverse()
            self.texts.append(' '.join([sample['Question']]+sample['Context']))
            self.labels.append(sample['Rewrite'])
        

    def __getitem__(self, idx):
        item = { }
        aux = tokenizer(self.texts[idx], max_length=500, truncation=True, padding='max_length')
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        aux = tokenizer(self.labels[idx], max_length=30, truncation=True, padding='max_length')
        item['labels'] = torch.tensor(aux['input_ids'])
        item['decoder_attention_mask'] = torch.tensor(aux['attention_mask'])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

del train_data
del val_data

now = datetime.now()
time_string = now.strftime("%d_%b_%H:%M_")

training_args = TrainingArguments(
    output_dir='pegasus_large_qreccCE' + time_string + '/results', 
    num_train_epochs=10,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=50,                
    gradient_accumulation_steps=4,               
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    group_by_length=True,
    optim = 'adamw_torch'
    )

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    tokenizer=tokenizer,
)


trainer.train()
trainer.save_model('pegasus_large_qreccCE' + time_string + '/model')

