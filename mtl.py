import numpy as np
import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import copy
import datasets
import random

online_model_name='google/pegasus-large'
target_model_name='google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(online_model_name)

online_model = PegasusForConditionalGeneration.from_pretrained(online_model_name)
target_model = PegasusForConditionalGeneration.from_pretrained(target_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_model.to(device)
target_model.to(device)

with open('/qrecc/qrecc_train.json') as file:
    train_data = json.load(file)

random.seed(42)

random_indexes = random.sample(range(len(train_data)),int(len(train_data)*0.2))

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
        self.answers=[]
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

def MSE(t,o):
    t = t.detach()
    return (t - o).square().mean()

target_decay_rate = 0.999
distilation_weight = 0.01
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        with torch.no_grad():
            #Target Decoder update
            params_s = list(online_model.model.decoder.parameters())
            params_t = list(target_model.model.decoder.parameters())
            torch._foreach_mul_(params_t, target_decay_rate)
            w = torch._foreach_mul(params_s, 1 - target_decay_rate)
            torch._foreach_add_(params_t, w)
            #Target Encoder update
            params_s = list(online_model.model.encoder.parameters())
            params_t = list(target_model.model.encoder.parameters())
            torch._foreach_mul_(params_t, target_decay_rate)
            w = torch._foreach_mul(params_s, 1 - target_decay_rate)
            torch._foreach_add_(params_t, w)

        target_output = target_model.forward(**inputs)
        online_output = online_model.forward(**inputs)

        loss = online_output['loss'] + MSE(target_output.logits,online_output.logits) * distilation_weight

        return (loss, online_output) if return_outputs else loss


now = datetime.now()
time_string = now.strftime("%d_%b_%H:%M_")

training_args = TrainingArguments(
    output_dir='pegasus_large_MTL_online' + time_string + '/results',
    num_train_epochs=10,              
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,   
    warmup_steps=50, #500               
    gradient_accumulation_steps = 16,               
    load_best_model_at_end=True,      
    evaluation_strategy="epoch",
    learning_rate = 5e-5,
    save_strategy="epoch",
    group_by_length=True,
    optim = 'adamw_torch'
    )

trainer = CustomTrainer(
    model=online_model,                  
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)


trainer.train()
trainer.save_model('pegasus_large_MTL_online' + time_string + '/model')

trainer2 = Trainer(
    model=target_model,
)
trainer.save_model('pegasus_large_MTL_target' + time_string + '/model')

