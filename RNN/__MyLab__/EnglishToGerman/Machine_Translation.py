# -*- coding: utf-8 -*-
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import spacy
# Define a function to load captions from a .txt file

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.__version__)
# print(device)


data = pd.read_csv("wmt14_translate_de-en_validation.csv", encoding='utf-8')

train, test = train_test_split(data, test_size=0.2, random_state=42)

# tabular_dataset can take csv already i don't have to make the above code.

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


english = Field(sequential=True, use_vocab=True,
                tokenize=tokenize_eng, lower=True)

german = Field(sequential=True, use_vocab=True,
               tokenize=tokenize_ger, lower=True)

fields = {"en": ("eng", english), "de": ("ger", german)}

train_data, test_data = TabularDataset.splits(
    path="", train='train.csv', test="test.csv", format="csv", fields=fields)


english.build_vocab(train_data, max_size=10000, min_freq=1)
german.build_vocab(train_data, max_size=10000, min_freq=1)


train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=32, device='cuda')

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(test_data)}")
print(f"Number of unique English words: {len(english.vocab)}")

for batach in train_iterator:
    print(batach.eng)
    print(batach.ger)
    break
