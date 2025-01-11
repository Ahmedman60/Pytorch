import csv
import os

# # print(os.path.dirname())
# # print(__file__)  # this gives you the full path to the file

# with open(os.path.join(os.path.dirname(__file__), 'train.csv'), 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # skip the header
#     for row in reader:
#         print(row)


# with open(os.path.join(os.path.dirname(__file__), 'train.csv'), 'r') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         print(row)
#         print(row["score"])


# Counting number of training examples for each score
# with open(os.path.join(os.path.dirname(__file__), 'train.csv'), 'r') as f:
#     reader = csv.DictReader(f)
#     count = {}
#     for row in reader:
#         if row["score"] in count:
#             count[row["score"]] += 1
#         else:
#             count[row["score"]] = 1


# for score, count in sorted(count.items()):
#     print(f"{score}: {count}")


# Counting number of training examples for each score
# with open(os.path.join(os.path.dirname(__file__), 'train.csv'), 'r') as f:
#     reader = csv.DictReader(f)
#     count = {}
#     for row in reader:
#         if row["score"] in count:
#             count[row["score"]] += 1
#         else:
#             count[row["score"]] = 1


# for score, count in sorted(count.items, key=lambda x: x[1], reverse=True):
#     print(f"{score}: {count}")

# for score in sorted(count.items()):
#     print(score)
#     # print(f"{score}: {count}")


# Using Counter in collections to
from collections import Counter

with open(os.path.join(os.path.dirname(__file__), 'train.csv'), 'r') as f:
    reader = csv.DictReader(f)
    count = Counter()
    for row in reader:
        # it will intialize eveything to zero and then start counting in the dictionary like we did above.
        count[row["score"]] += 1

print(count.most_common())
