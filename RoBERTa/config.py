import os

CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5"
# CUDA_VISIBLE_DEVICES = "0"

random_state = 42

data_file = "data/train-balanced-sarcasm.csv"

TRAIN_SIZE = 0.75

learning_rate = 1e-5
num_epochs = 10
batch_size = 128
num_workers = 80  # Number of workers for the dataloaders

threshold = (
    0.5  # Score threshold to convert model's output probabilities to binary predictions
)

EARLY_STOPPING = 3


use_sentiment_context = True
