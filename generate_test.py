import numpy as np
import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
import datasets


model_name='google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open('/qrecc/qrecc_test.json') as file:
    test_data = json.load(file)

predictions = []
for example in test_data:
  example['Context'].reverse()
  input = ' '.join([example['Question']]+example['Context'])

  input_ids = tokenizer(input, max_length=500, truncation=True, padding='max_length', return_tensors="pt").input_ids.cuda()

  outputs = model.generate(input_ids = input_ids, num_beams=5, num_return_sequences=1)

  predictions.append(tokenizer.decode(outputs[0]))


text_file = open('results.txt', 'a')
for pred in predictions:
  text_file.write(pred + "\n")
text_file.close()