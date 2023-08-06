# Sarcasm Detection with RoBERTa
This project is about sarcasm detection in reddit comments using a pre-trained model called RoBERTa.

This directory (`data/`) contains the dataset used for this project. The dataset should be downloaded from the following source:

## Dataset

The sarcasm detection model in this repository relies on the following dataset:

**Dataset Source:** [NLPrinceton/SARC](https://github.com/NLPrinceton/SARC)

Please read the README file in the dataset source for more information about setting up the dataset.

## Requirements

To run this project, you need to have the required packages installed by running the following command:
`pip install -r requirements.txt`

## Usage

To run this project, you need to follow these steps:

- Clone this repository to your local machine. 
- Download the SARC dataset and save it in the data folder (Refer to the README file in the data directory for more information).
- Run the main.py file with the following command:

`python main.py --mode <mode> --checkpoint <checkpoint>`

The mode argument specifies whether to run the project in train, test, or predict mode. The checkpoint argument specifies the path to a checkpoint file to load or save. If not given, the default values are “train” and None respectively.

You can also change some settings and hyperparameters in the config.py file, such as learning rate, batch size, number of epochs, threshold, and whether to use sentiment context or not.