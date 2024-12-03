import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from mlp import MLP

'''
To the future me: this is just me practicing setting up efficient data loaders taking reference from the PyTorch appendix (A.6) of the LLM book by Sebastian Raschka. I'm getting some feel for NN training before I start training the GPT-2 model I've constructed.
'''

print("=====================================================================================\nThis is where the results being printed for the current file begin\n=====================================================================================")

# creating a small dataset

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

# creating a custom Dataset class

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        indiv_x = self.features[idx]
        indiv_y = self.labels[idx]
        return indiv_x, indiv_y

train_ds = CustomDataset(X_train, y_train)
test_ds = CustomDataset(X_test, y_test)

print(f"Length of the training dataset: {len(train_ds)}\n")
print(f"X_train's shape: {X_train.shape}\ny_train's shape: {y_train.shape}\nX_test's shape: {X_test.shape}\ny_test's shape: {y_test.shape}")

print("---------------------------------------------------------------")

# instantiating data loaders

torch.manual_seed(123)

train_loader = DataLoader(
    train_ds,
    batch_size = 2,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)

test_loader = DataLoader(
    test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0,
    drop_last = False
)

# iterating over these

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:\nx: {x}\ny: {y}\n")


print("----------------------------------------------------------------------")

# a simple training loop for training

torch.manual_seed(123)

model = MLP(num_inputs = 2, num_outputs = 2)

print(f"Total parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.SGD( # stochastic gradient descent
    params = model.parameters(),
    lr = 0.5
)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        # Sets the gradients from the previous round to 0
        # to prevent unintended gradient accumulation
        optimizer.zero_grad()
        loss.backward()

        # optimizer uses the gradients to update the model parameters
        optimizer.step()

        # LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
    
    model.eval()
        # optional model eval code

# the model is trained now. doing the output:

model.eval()
with torch.no_grad():
    outputs = model(X_train)
print("Outputs:\n", outputs)

# softmax to obtain class membership probabilities

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(f"Class Membership Probabilities:\n{probas}")

# creating predictions based on probabilities
predictions = torch.argmax(probas, dim=1)
print(f"Predictions: {predictions}")

print(predictions == y_train)

print(torch.sum(predictions == y_train))

# a function to compute the prediction accuracy
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        # not doing softmax since this model doesn't need it (redundancy reasons)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (f"{(correct / total_examples).item() * 100}% accuracy")

print(compute_accuracy(model, train_loader))

print(compute_accuracy(model, test_loader))
