# Bulding sequence2equence model  from arabic to englishes
import string
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim
import unicodedata
import spacy
import os
import torchtext as text
import re
# from torchtext.datasets import Multi30k

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, use_vocab=True,
                tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True,
               tokenize=tokenize_ger, lower=True)


train_data, valid_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english))
