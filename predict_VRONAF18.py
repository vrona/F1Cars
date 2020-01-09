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

# Imports python module
import argparse

# Imports

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
import json

# Creates parse and Define CLA
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='f1_data', help='Path to\
datasets')
parser.add_argument('--picture', type=str, help='data/f1_data/test/10/\
_906.jpg')
# parser.add_argument('--gpu', action='store_true', help='Use GPU')
parser.add_argument('--arch', type=str, help='Model of your choice:\
vgg11, 13, 16, 19')

args = parser.parse_args()

# Data sets paths
data_dir = 'Desktop/f1_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
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

# Label mapping
with open('Desktop/cat_to_nameF1.json',
          'r') as f:
    cat_label = json.load(f)


def main():
    # Importing the model that the user choosed --arch vgg11, 13, 16, 19.
    from train_TL_VRONAF18 import load_model

    # Loading of the checkpoint and rebuilds the model

    def load_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = load_model()
        model.classifier = checkpoint['classifier']
        model.load_state_dict = checkpoint['state_dict']
        # optimizer.load_state_dict(checkpoint['optimizer'])

        return model

    model = load_checkpoint('{}_checkpoint.pth'.format(args.arch))
    print("Checkpoint loaded!")

    # Processing the image
    def process_image(img_path, max_size=400, shape=None):

        # load image and convert PIL images to tensors
        image = Image.open(img_path)

        # no large images & transformation image as requested
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose(
                                [transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

        # instead of / 255
        image = in_transform(image).float()

        # move to an array with float between 0-1
        image = np.array(image)

        image = image.transpose((1, 2, 0))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image

    # Class Prediction

    model.class_to_idx = train_data.class_to_idx

    def predict(image_path, model, topk=3):

        # Prediction of the class from an image file

        global image
        image = process_image(image_path)
        image.unsqueeze_(0)
        image.requires_grad_(False)
        model = model

        image = Variable(image.cuda())

        model.eval()
        # Change to cuda
        model.cuda()
        output = model.forward(image)
        ps = torch.exp(output)

        probs, indices = ps.topk(topk)

        indices = indices.cpu().numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in indices]
        names = [cat_label[str(j)] for j in classes]

        return probs, classes, names

    # sanity checking with 3 F1 Drivers classes/names
    if args.picture:
        image_path = args.picture
    probs, classes, names = predict(image_path, model)
    print('picture path:', image_path)
    print('probability of classification:', probs)
    print('classes number associated:', classes)
    print('names of F1 Drivers associated:', names)


# Call to main function to run the program

if __name__ == "__main__":
    main()
