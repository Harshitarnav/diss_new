import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load the training dataset
# Assume the training data is in the format of list of texts and corresponding labels
df = pd.read_csv("dataset.csv")
train_texts = df["text"]
train_labels = df["label"]
# Split the training data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# Convert texts to input format suitable for BERT
train_inputs = convert_texts_to_inputs(train_texts, tokenizer)
val_inputs = convert_texts_to_inputs(val_texts, tokenizer)

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# Fine-tune the BERT model on the training set
model.train()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(num_epochs):
    # Calculate the loss and accuracy on the training set
    train_loss, train_acc = train_epoch(model, loss_fn, train_inputs, train_labels, optimizer)
    # Calculate the loss and accuracy on the validation set
    val_loss, val_acc = eval_epoch(model, loss_fn, val_inputs, val_labels)
    # Print the results
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Load the testing dataset
# Assume the testing data is in the format of list of texts and corresponding labels
test_texts, test_labels = load_testing_data()

# Convert texts to input format suitable for BERT
test_inputs = convert_texts_to_inputs(test_texts, tokenizer)

# Convert labels to tensors
test_labels = torch.tensor(test_labels)

# Evaluate the best model on the testing set
model.eval()
test_loss, test_acc = eval_epoch(model, loss_fn, test_inputs, test_labels)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f})
