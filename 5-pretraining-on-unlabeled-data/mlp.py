import torch
import torch.nn as nn

'''
To the future me: this is just me practicing some classic NN using PyTorch to get a feel for things. No relation to the actual GPT building.
'''

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            # hidden layer 1
            nn.Linear(num_inputs, 30),
            nn.ReLU(),

            # hidden layer 2
            nn.Linear(30, 20),
            nn.ReLU(),

            # output layer
            nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

torch.manual_seed(123)
model = MLP(50, 3)
X = torch.rand(1, 50)
print(f"\nInput X:\n{X}\nShape of X: {X.shape}")
out = model(X)
print("\nOut:\n", out)

print("\nModel:\n", model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTotal trainable model parameters:\n", num_params)

print(f"\nFirst linear layer:\n{model.layers[0].weight}\nShape of this layer: {model.layers[0].weight.shape}")

print("-----------------------------------------------------------")

print('''
From Sebastian Raschka's LLM from Scratch Book:
In PyTorch, it’s common practice to code models such that they return the outputs of the last layer (logits) without passing them to a nonlinear activation function. That’s because PyTorch’s commonly used loss functions combine the softmax (or sigmoid for binary classification) operation with the negative log-likelihood loss in a single class. The reason for this is numerical efficiency and stability. So, if I want to compute class-membership probabilities for our predictions, I have to call the softmax function explicitly.
''')

with torch.no_grad():
    out1 = torch.softmax(model(X), dim=1)
print(f'''\nOut after softmax: {out1}

The values can now be interpreted as class-membership probabilities that sum up to 1. The values are roughly equal for this random input, which is expected for a randomly initialized model without training
''')
