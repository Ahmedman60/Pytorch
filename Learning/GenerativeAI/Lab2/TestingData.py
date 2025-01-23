from datasets import Dataset, DatasetDict

# Example data
train_data = {'input_ids': [[1, 2, 3], [4, 5, 6]], 'labels': [0, 1]}
validation_data = {'input_ids': [[7, 8, 9]], 'labels': [0]}
test_data = {'input_ids': [[10, 11, 12], [13, 14, 15]], 'labels': [1, 0]}

# Create Dataset objects
train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(validation_data)
test_dataset = Dataset.from_dict(test_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# Print the DatasetDict
print(dataset_dict)
