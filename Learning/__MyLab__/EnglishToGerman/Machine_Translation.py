import random

# # -*- coding: utf-8 -*-
# import pandas as pd
# from torchtext.data import Field, BucketIterator, TabularDataset
# from sklearn.model_selection import train_test_split
# import torch
# import matplotlib.pyplot as plt
# import spacy
# # Define a function to load captions from a .txt file

# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # print(torch.__version__)
# # print(device)


# data = pd.read_csv("wmt14_translate_de-en_validation.csv", encoding='utf-8')

# train, test = train_test_split(data, test_size=0.2, random_state=42)

# # tabular_dataset can take csv already i don't have to make the above code.

# train.to_csv('train.csv', index=False)
# test.to_csv('test.csv', index=False)

# spacy_ger = spacy.load('de_core_news_sm')
# spacy_eng = spacy.load('en_core_web_sm')


# def tokenize_ger(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]


# def tokenize_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]


# english = Field(sequential=True, use_vocab=True,
#                 tokenize=tokenize_eng, lower=True)

# german = Field(sequential=True, use_vocab=True,
#                tokenize=tokenize_ger, lower=True)

# fields = {"en": ("eng", english), "de": ("ger", german)}

# train_data, test_data = TabularDataset.splits(
#     path="", train='train.csv', test="test.csv", format="csv", fields=fields)


# english.build_vocab(train_data, max_size=10000, min_freq=1)
# german.build_vocab(train_data, max_size=10000, min_freq=1)


# train_iterator, test_iterator = BucketIterator.splits(
#     (train_data, test_data), batch_size=32, device='cuda')

# print(f"Number of training examples: {len(train_data)}")
# print(f"Number of validation examples: {len(test_data)}")
# print(f"Number of unique English words: {len(english.vocab)}")

# for batach in train_iterator:
#     print(batach.eng)
#     print(batach.ger)
#     break

# Lst = [50, 70, 30, 20, 90, 10, 50]
# # Display list
# print(Lst[0:4])  # 4 is exclusive
# print(Lst[::])
# print(Lst[::2])
# print(Lst[::-1])

# text = 'Hello World Udacity Company '
# # splits with string in the form of list
# list_1 = text.split()

# print(list_1)

# for i in range(0, len(list_1), 2):
#     print(list_1[i]+" "+list_1[i+1])


# m = "mohamed"  # collection of characters

# print(m)
# print(m[2])

# dictionary = {"key": "value"}
# z = {"mo": 528, "ar": 589}

# print(z)
# print(z["mo"])


x = "Udacity"
# dj

# print(x[0:])

# y = "ahmed mohamed samy mai jana"

# for name in y.split(" "):  # by default it uses spaces
#     print(name, "welcome to the django")


# text = 'Hello, World, Udacity,mohamed'
# split_1 = text.split(',', 2)

# print(split_1)


s = [10, 20, 30, 40, 50, 70]

print(s[1])

s = "Mohamed Fathallah"

print(s[:])


# names = "ahmed mohamed samy aser"


# x = names.replace(" ", "--")  # []

# print(names)
# print(x)


# text = 'Hello, World, Udacity, English'
# split_1 = text.split(',', 2)

# print(split_1)
