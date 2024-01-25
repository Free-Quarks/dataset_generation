import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data_processing.tokenizer import tokens_to_indices
# let's first make the embedding layer, There are 8 different words,
# each of which can have n indexes mentioning it. Initially, perhaps a 
# dimension of 8, one basis vector for each direction, but I don't think
# we need that much information, so let's reduce it to 3. Let's overscope
# for this test model. In the future I want an embedding for each 8 words
# and then another one for every number from 0-500, just adding them together
# when relevant, for now let's assume 50 for each words, so vocab of 400

class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.embedding = torch.nn.Embedding(1500, 4)
        #self.lstm = torch.nn.LSTM(4, 8, 1)
        self.linear = torch.nn.Linear(4, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        #x, y = self.lstm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = x.sum(dim=1) # sum the feature vectors of the each sentence, one float per sentence
        x = x.sum(dim=1) # sum each sentence of a page, one float per page

        return x
    

# this is a sample run of the model    
output_tokens_directory = '../../data/function_nets_tokenized'
output_labels_directory = '../../data/labels_tokenized'

dataset = tokens_to_indices(output_tokens_directory, output_labels_directory)

simplemodel = SimpleModel()

num_epochs = 2
device = torch.device('cpu')
model = simplemodel.to(device)
x_values = dataset[0]
y_values = dataset[1]
tensor_x = torch.IntTensor(x_values)
print("tensor_x shape:", tensor_x.shape)
tensor_y = torch.Tensor(y_values)

training_dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(training_dataset)
optimizer = torch.optim.Adam(simplemodel.parameters(), lr=0.005, weight_decay=5e-4)

criterion = torch.nn.MSELoss(reduction='mean')

model.train()

for epoch in range(num_epochs):

    for batch in loader:
        # data in this is a list of data in the size of the batch
        optimizer.zero_grad()
        out = model(batch[0])
        loss = criterion(out, batch[1])
        loss.backward()
        optimizer.step()

    print("------Epoch updates-------")
    print("input shape: ", batch[0].shape)
    print('out shape: ', out.shape)
    print('label shape: ', batch[1].shape)
    print('out: ', out)
    print('label: ', batch[1])
    print('loss: ',loss)






