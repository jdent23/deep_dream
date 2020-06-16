# Deep Dream for PyTorch

[TOC]

## Classifier

Trains generic classification models

- Follows generic Xception submodule throughout each layer
- Tests for convergence to prevent overfitting at each epoch
- Start with pre-trained weights for faster convergence

## Dreamer

Enhances features that models identify within example images

- InceptionV3 model with pretrained weights
- Trains the image instead of the model through gradient ascent
- Creates surreal special effects

## Libutensor

Provides PyTorch helper functions for other sub-packages

## Reddit Scrapping

Downloads images about specific topics for training purposes

- Scrapes sub-Reddits for images
- Subreddits are ideal for retrieving large numbers of images about extremely specific topics

