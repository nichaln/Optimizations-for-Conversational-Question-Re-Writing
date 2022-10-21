import numpy as np
import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import copy
import random
import datasets


model_name='google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
            if(sample['Answer'] != ""):
                sample['Context'].reverse()
                self.texts.append(' '.join([sample['Question']]+sample['Context']))
                self.labels.append(sample['Rewrite'])
                self.answers.append(sample['Answer'])
        

    def __getitem__(self, idx):
        item = { }
        aux = tokenizer(self.texts[idx], max_length=500, truncation=True, padding='max_length')
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        aux = tokenizer(self.labels[idx], max_length=30, truncation=True, padding='max_length')
        item['labels'] = torch.tensor(aux['input_ids'])
        item['decoder_attention_mask'] = torch.tensor(aux['attention_mask'])
        aux = tokenizer(self.answers[idx], max_length =100, truncation=True, padding='max_length')
        item['answer'] = torch.tensor(aux['input_ids'])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

del train_data
del val_data

answer_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', device='cuda')

LogSoftmax_function = torch.nn.LogSoftmax(dim=2)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        forward_inputs = copy.deepcopy(inputs)
        del forward_inputs['answer']
        outputs = model.forward(**forward_inputs)

        with torch.no_grad():
            sampled_outputs = model.generate(input_ids = inputs['input_ids'], do_sample=True, attention_mask = inputs['attention_mask'], max_length=30)
            greedy_outputs = model.generate(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], num_beams=5, num_return_sequences=1, max_length=30)

        mask = sampled_outputs.ge(1).to(device)

        scores = torch.gather(LogSoftmax_function(outputs.logits),2,sampled_outputs[:,:,None]).squeeze(-1) * mask
        scores = scores.sum(1) / mask.int().sum(1)

        rewards = []

        for i in range(len(sampled_outputs)):

            sampled_query_emb = answer_model.encode(tokenizer.decode(sampled_outputs[i]))
            baseline_query_emb = answer_model.encode(tokenizer.decode(greedy_outputs[i]))

            answer_emb = answer_model.encode([tokenizer.decode(inputs.get("answer")[i])])

            sample_score = util.dot_score(sampled_query_emb, answer_emb)[0].cpu().tolist()[0]
            baseline_score = util.dot_score(baseline_query_emb, answer_emb)[0].cpu().tolist()[0]

            rewards.append(sample_score - baseline_score)

        rewards = torch.tensor(rewards).to(device)

        loss = - scores * rewards
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


now = datetime.now()
time_string = now.strftime("%d_%b_%H:%M_")

training_args = TrainingArguments(
    output_dir='pegasus_large_RLANSWER' + time_string + '/results', 
    num_train_epochs=6,              
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,   
    warmup_steps=50,                            
    gradient_accumulation_steps = 8,               
    load_best_model_at_end=True,      
    evaluation_strategy="epoch", 
    learning_rate = 2.5e-6,
    save_strategy="epoch",
    group_by_length=True,
    optim = 'adamw_torch'
    )

trainer = CustomTrainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)


trainer.train()
trainer.save_model('pegasus_large_RLANSWER' + time_string + '/model')

