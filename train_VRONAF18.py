#!/Users/MikeJohn/anaconda3/bin/python
# -*- coding: utf-8 -*-
# */Formula_1 car and driver/image_classifier
#
# PROGRAMMER: Michael Hatchi
# DATE CREATED: 16/nov/2018
# PURPOSE: This is an image classifier python application paired of
#          Python scripts that run from the command line.
#          The principle is to let anyone submit a Formula_1 2018 championship
#          image in order to predict the driver(s) who have been pictured.
##

# Imports

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.autograd import Variable

# Mapping data (categories and pictures)
import json

# Replace classifier by new untrained feed-forward network hyperparameters
from collections import OrderedDict

# Import validation function
from validation import validation

# Data sets paths
data_dir = 'data/f1_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation and testing sets
# pylint: disable=no-member
train_transforms = transforms.Compose(
                               [transforms.RandomRotation(90),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
# pylint: disable=no-member
valid_transforms = transforms.Compose(
                            [transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose(
                            [transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,
                                          shuffle=True)
testloarder = torch.utils.data.DataLoader(test_data, batch_size=32)

# Categories to Labels
with open('Desktop/cat_to_nameF1.json',
          'r') as f:
    cat_label = json.load(f)

# Loading model
model = models.vgg13(pretrained=True)
print(model)
print("Pre-trained model Step done!")

# Set up the hyperparameters
input_size = 25088
hidden_size = [4096, 1000, 256]
output_size = len(cat_label)
lr = 0.001
epochs = 20

# # STEP CLASSIFIER Building the dict classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_size[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc3', nn.Linear(hidden_size[1], hidden_size[2])),
                          ('relu3', nn.ReLU()),
                          ('fc4', nn.Linear(hidden_size[2], output_size)),
                          ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

# Criterion and Optimizer for backpropagation
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# ----- Training & Validation & Test -----
# # STEP TRAINING then eval on, respectively, train and valid data sets
epochs
print_every = 20
steps = 0
running_loss = 0

# Change to cuda
model.to('cuda')

for e in range(epochs):
    model.train()
    for ii, (image, label) in enumerate(trainloader):
        steps += 1

        image = Variable(image.cuda())
        label = Variable(label.cuda())

        optimizer.zero_grad()

        # Forward and backward passes
        output = model.forward(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.to('cuda')
            model.eval()

            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training Loss:\
                  {:.3f}..".format(running_loss/print_every),
                  "Validation Loss:\
                  {:.3f}..".format(test_loss/len(validloader)),
                  "Validation Accuracy:\
                  {:.3f}..".format(accuracy/len(validloader)))

            running_loss = 0
            model.train()

# STEP TEST of network's accuracy on Test data sets
correct = 0
total = 0
with torch.no_grad():
    for data in testloarder:
        model.eval()
        image = Variable(image.cuda())
        label = Variable(label.cuda())

        model.to('cuda')

        optimizer.zero_grad()
        output = model(image)

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
print('Accuracy of the network on test_data: %d %%' % (100 * correct / total))


# STEP CHECKPOINT

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': input_size,
              'output_size': len(cat_label),
              'hidden_size': hidden_size,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')
print('checkpoint has been saved')
