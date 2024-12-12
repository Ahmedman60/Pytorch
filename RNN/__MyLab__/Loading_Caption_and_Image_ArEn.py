# -*- coding: utf-8 -*-
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
# Define a function to load captions from a .txt file


def load_file(file_path):
    """
    Load file contents, stripping whitespace from each line.

    Args:
        file_path (str): Path to the file to be read

    Returns:
        list: Lines from the file with whitespace stripped
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except IOError as e:
        print(f"Error reading file: {e}")
        return []


ids = load_file("IDs.txt")
# english_captions = load_file("English.txt")
# arabic_captions = load_file("Arabic.txt")

print(ids)

# # Combine them into a single structure
# dataset = []
# for img_id, eng_caption, ar_caption in zip(ids, english_captions, arabic_captions):
#     dataset.append({
#         "image_id": img_id,
#         "english_caption": eng_caption,
#         "arabic_caption": ar_caption
#     })

# image_dir = os.getcwd()+"\Images"  # need to add the path here
# print(image_dir)
# # # Print a sample
# # print(dataset[:1])
# # print(len(dataset))


# print(dataset[0])
# sample = dataset[0]

# image_path = os.path.join(image_dir, f"{sample['image_id']}")
# #Testing code for geting path
# print(os.getcwd())  # Prints the current working directory
# print(image_path)

# class MultimodalDataset(Dataset):
#     def __init__(self, data, image_dir, transform=None):
#         self.data = data
#         self.image_dir = image_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         image_path = os.path.join(self.image_dir, f"{sample['image_id']}")
#         image = Image.open(image_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, sample["english_caption"], sample["arabic_caption"]


# # Define transformations (e.g., resizing, normalizing)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Initialize the dataset
# dataset = MultimodalDataset(
#     dataset, image_dir=image_dir, transform=transform)

# # Example: Access a sample
# image, eng_caption, ar_caption = dataset[0]
# print("English Caption:", eng_caption)
# print("Arabic Caption:", ar_caption)
# plt.imshow(image.permute(1, 2, 0))
# plt.show()
