# Fashion-MNIST Classification with Fine-Tuned ConvNet

This project implements a Convolutional Neural Network (ConvNet) for classifying the Fashion-MNIST dataset, with fine-tuning capabilities. The model is trained in two phases: first with frozen feature extraction layers, and then with fine-tuning enabled.

## Project Structure

- `model.py`: Contains the ConvNet architecture with fine-tuning capabilities
- `data_utils.py`: Handles data loading and preprocessing for Fashion-MNIST
- `train.py`: Main training script with training and evaluation logic
- `requirements.txt`: Project dependencies

## Features

- Convolutional Neural Network architecture with batch normalization
- Two-phase training with fine-tuning
- Progress tracking with tqdm
- Training metrics visualization
- Model checkpointing

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

The script will:

1. Download the Fashion-MNIST dataset
2. Train the model for 20 epochs
3. Enable fine-tuning after 10 epochs
4. Save the trained model as 'fashion_mnist_model.pth'
5. Generate training metrics plots

## Model Architecture

The ConvNet consists of:

- 3 convolutional blocks with batch normalization and max pooling
- Dropout layers for regularization
- Fully connected layers for classification

## Fine-Tuning Strategy

The model is trained in two phases:

1. First 10 epochs: Only classifier layers are trained
2. Last 10 epochs: All layers are fine-tuned with a reduced learning rate

## Results

The training progress is displayed in real-time, showing:

- Training loss and accuracy
- Validation loss and accuracy
- Progress bars for each epoch

A plot of training metrics is saved as 'training_metrics.png' after training completes.
