import string
import unicodedata
from io import open
import glob
import os
import torch
import random


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def load_data():
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def load_random_example(category_lines, all_categories):
    # random_language_path
    country = random.sample(category_lines.keys(), 1)[0]
    lines = category_lines[country]  # this will get all lines or words
    line = random.choice(lines)
    # this will return the random word from random categorie as tensor.
    # i return the categorie and the word also
    line_tensor = lineToTensor(line)
    category_tensor = torch.tensor(
        [all_categories.index(country)], dtype=torch.long)

    return line_tensor, category_tensor, country, line
    # categories line already loaded we don't need to load it again


if __name__ == "__main__":
    cl, c = load_data()

    oneexample = print(load_random_example(cl, c))
