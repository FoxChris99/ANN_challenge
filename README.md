# Artificial Neural Networks and Deep Learning challenges

## Image classification

For this competition we are required to classify images of plants which are divided into 8 categories
according to their species. Our goal is to predict the correct class label. We are given a dataset
made of 3542 images whose size is 96×96 pixels and the color space is RGB.
First we make an exploratory analysis of the dataset, then we move to the data preprocess and
the development of a model ”from scratch”. In the end we experiment transfer learning and fine
tuning with some famous pre-trained models, considering also ensemble learning techniques.
A key point of our training procedure concerns data augmentation; the possible random transformations we take into account are: rotations, zooms , random flips, shifts over height and width, brightness
changes and channel shifts.

#### CNN from scratch

| Layer   | filters/units |
|---------|--------------:|
| Conv1   |            24 |
| Conv2   |            32 |
|MaxPool2D|               |
|BatchNorm|               |
| Conv3   |            64 |
| Conv4   |            64 |
| Conv5   |            64 |
|MaxPool2D|               |
|BatchNorm|               |
| Conv6   |            96 |
|MaxPool2D|               |
|BatchNorm|               |
| Conv7   |           128 |
|GlobalMaxpool            |
|BatchNorm|               |
| Dropout |           0.2 |
| Dense   |           256 |
| Dropout |           0.2 |
| Dense   |           256 |
| Dropout |           0.2 |
| Output  |             8 |

With Transfer learning and fine tuning of Xception, VGG-16, EfficientNetB6 pre-trained on ImageNet we push the accuracy score up to 85%.




##  Time Series Classification

In this competition we have to work on a classification problem with multivariate time series data.
The dataset contains 2429 samples, where each sample is a time series of six different features having 36 time instances. For the classification task of this challenge there are 12 possible outcomes.
The two main typologies of models we consider are: reccurent neural networks (RNN) and one
dimension convolutional neural networks (CNN). We choose RNN because we have sequences of
data, in particular we focus on Long short-term memory (LSTM) networks.
When experimenting with LSTM models, we consider, as well as the
classical layers, also bidirectional LSTM layers (BiLSTM).
The last model we develop is composed both by convolution operations and LSTM
units.


#### CNN + BiLSTM

| \textbf{Layers}     | \textbf{Filters/Units} |
|---------------------|:----------------------:|
| Input               |                        |
| Conv1               |           32           |
| Conv2               |           64           |
| Conv3               |           128          |
| Bidirectional(LSTM) |           128          |
| Bidirectional(LSTM) |           128          |
| Dropout             |           0.2          |
| Dense               |           64           |
| Dropout             |           0.2          |
| Dense               |           64           |
| Output              |           12           |

Thanks to an ensemble model we could reduce the variance of the misclassification error, reaching 75% of overall accuracy.
