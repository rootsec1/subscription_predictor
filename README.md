# Kaggle Classical Music Meets Classical ML Fall 2023

Predicting whether a user is likely to buy a ticket to a musical event using machine learning.
Solution for [this kaggle competition](https://www.kaggle.com/competitions/classical-music-meets-classical-ml-fall-2023/data)

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict whether a user is likely to purchase a ticket to a musical event. It uses a machine learning model to make predictions based on user data. The project is divided into data preprocessing, model training, and inference.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies mentioned in the [Prerequisites](#prerequisites) section.
3. Follow the [Data Preprocessing](#data-preprocessing) section to prepare the dataset.
4. Train the model as described in [Training the Model](#training-the-model).
5. Use the model for predictions by following the [Inference](#inference) section.

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

- Python 3
- pandas
- scikit-learn
- numpy
- pandas

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn
```

## Data Preprocessing

Data preprocessing involves cleaning and preparing the dataset for model training. This step is crucial for model accuracy. You can use the provided PreProcess class for data preprocessing. Make sure you have your dataset in the specified directory.

```python
from preprocess import PreProcess
preprocess_instance = PreProcess(dataset_root_dir="../data/features")
training_df = preprocess_instance.clean_data()
```

## Feature Selection

Initially chose to work with only the `accounts.csv`, `subscrptions.csv` and `tickets.csv` and this in on itself seemed to contain significantly valuable information than the rest of them. `tickets.csv` didn't seem to contribute to much so I decided to not use it.

Initially decided to drop all the non-number datatype columns in order to avoid encoding each and every column which could've been very time consuming. I preserved all the integer/float datatype columns (alongwith the `account.id` - string) from `accounts.csv` and `subscriptions.csv` to start with. The `account.id` column which is a string is encoded into an integer using scikit-learn's `LabelEncoder`. In `subscriptions.csv`, I am grouping the subscriptions by `account.id` and creating a new column called `num_subscriptions` which seems to be the most important feature so far. This new dataframe with `num_subscriptions` in it is now merged with the `accounts` dataframe. The combination of this is then merged with the `train.csv` dataframe to get the labels associated with each `account.id`.

## Model selection and training

The model I've selected is `RandomForestClassifier`. Tried to bruteforce classifiers and hyperparameters to find the optimal accuracy and AUROC. Tried `KNeighborsClassifier` and other models to find the model with the best accuracy and AUROC. Through trial and error, I found out that the `RandomForestClassifier` with no hyperparameter turning was still significantly more efficient than the rest. Due to time constraints, I wasn't able to experiment more. If I had to continue working on this, I would use `GridSearch` and other techniques to find the most optimal hyperparameters and model.

The `Model` class is responsible for training the machine learning model. The following code snippet shows how to train the model:

```python
from model import Model
model_instance = Model(training_df=training_df)
model_instance.predict_and_return_accuracy_and_roc()
```

The model is trained using a RandomForestClassifier. After training, it saves the model using pickle.

## Inference

To make predictions using the trained model, use the following code:

```python
model_instance = Model(training_df=training_df)
user_data = [user_feature_1, user_feature_2, ...]  # Provide user account IDs as a list
prediction = model_instance.run_inference(user_data)
```

The `run_inference` method takes a list of user features as input and returns a prediction.

## Pipeline

#### Preprocessing (cleaning & selecting features)
`main.py` invokes the preprocessing module which cleans the `accounts.csv` and `subscriptions.csv` and puts them in a dataframe. The `PreProcessing` class also contains another function to not only clean the data but also add new features such as `num_subscriptions`.

#### Modelling (fitting the data into the model)
Using the resultant dataframe obtained from the `PreProcessing` module, we pass that on to the `Model` class which trains a `RandomForestClassifier` model. At the end of the training process, the accuracy and the AUROC are printed on screen. The trained model is then pickled and stored in `models/classifier.pkl` file

#### Inference (get predictions)
Once we have a model trained, any `account.id`s passed as input is then translated to a set of features which is then passed to the model for making predictions.

## Results

You can evaluate the model's performance using accuracy and the area under the ROC curve (ROC AUC). After training the model, these metrics are printed to the console:

```python
acc, roc = model_instance.predict_and_return_accuracy_and_roc()
print("Accuracy: ", acc)
print("Area under ROC curve: ", roc)
```
