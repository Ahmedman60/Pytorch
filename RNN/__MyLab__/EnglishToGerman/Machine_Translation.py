# -*- coding: utf-8 -*-
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import spacy
# Define a function to load captions from a .txt file


data = pd.read_csv("wmt14_translate_de-en_validation.csv", encoding='utf-8')

data.head(5)

print(data.head(5))

# # Print a sample
# print(dataset[:1])
# print(len(dataset))


# print(dataset[0])
# sample = dataset[0]

# image_path = os.path.join(image_dir, f"{sample['image_id']}")
# #Testing code for geting path
# print(os.getcwd())  # Prints the current working directory
# print(image_path)
