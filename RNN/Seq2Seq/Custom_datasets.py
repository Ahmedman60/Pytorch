
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn

# Tokenization function


def tokenize(x): return x.split()


# Define Fields for processing
text = Field(sequential=True, tokenize=tokenize, use_vocab=True, lower=True)

# Assuming labels are integers
label = Field(sequential=False, use_vocab=False)

# Define dataset fields mapping
fields = {'text': ('t', text), 'label': ('l', label)}

# Load JSON dataset
train_data, test_data = TabularDataset.splits(
    path="mydata",
    train="train.csv",
    test="test.csv",
    format="csv",
    fields=fields,
)

# print(train_data[0].__dict__.keys())
# print(train_data[0].__dict__.values())


# # Build vocabulary for text field

text.build_vocab(train_data, max_size=10000, min_freq=1)

# # Create iterators

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    device="cuda",
)

# for batch in train_iterator:
#     print(batch.t)  # Padded text tensor, shape: (seq_len, batch_size)
#     print(batch.l.float())  # Labels, shape: (batch_size,)

# print(len(text.vocab))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text shape: (seq_len, batch_size)
        embedded = self.embedding(text)
        # embedded shape: (seq_len, batch_size, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (1, batch_size, hidden_dim) -> Last hidden state
        # Output shape: (batch_size, output_dim)
        return self.fc(hidden.squeeze(0))


# Define parameters
INPUT_DIM = len(text.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1  # Binary classification

# Initialize model, optimizer, and loss
model = LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to("cuda")
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(5):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in train_iterator:
        text = batch.t  # Padded text tensor, shape: (seq_len, batch_size)
        labels = batch.l.float()  # Labels, shape: (batch_size,)

        optimizer.zero_grad()
        # Predictions shape: (batch_size,)
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
