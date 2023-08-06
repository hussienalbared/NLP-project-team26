import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from config import *
from preprocessing import preprocess
from textblob import TextBlob

# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import RobertaTokenizer

# Set a random seed for reproducibility
pl.seed_everything(random_state, workers=True)


class SARCDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        texts = None
        if use_sentiment_context:
            texts = X[0]
            sentiment_polarities = X[1]
            sentiment_subjectivities = X[2]

            texts = [
                preprocess(text, p, s)
                for text, p, s in tqdm(
                    zip(texts, sentiment_polarities, sentiment_subjectivities),
                    desc="Preprocessing",
                    total=len(texts),
                )
            ]
        else:
            texts = X
            texts = [preprocess(text) for text in tqdm(texts, desc="Preprocessing")]

        # texts = [preprocess(text) for text in tqdm(texts, desc="Preprocessing")]

        self._print_random_samples(texts)

        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=150,
                truncation=True,
                return_tensors="pt",
            )
            for text in tqdm(texts, desc="Tokenizing")
        ]

        self.labels = y

    def _print_random_samples(self, texts):
        print("Random samples after preprocessing:")
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, "labels"):
            label = self.labels[idx]

        return text, label


class SarcasmDetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_file, batch_size=8, num_workers=0, mode="train"):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sarcasm_df = None
        self.mode = mode
        self.dataset = None

        self.prepare_data()

    def prepare_data(self):
        print("Preparing data...")
        sarcasm_df = pd.read_csv(self.data_file)
        # sarcasm_df = pd.read_csv("data/sarcasm_preprocessed_sentiments.csv")
        # self.sarcasm_df = sarcasm_df
        # return

        # We just need comment & label columns
        # So, let's remove others.
        sarcasm_df.drop(
            [
                "author",
                "subreddit",
                "score",
                "ups",
                "downs",
                "date",
                "created_utc",
                "parent_comment",
            ],
            axis=1,
            inplace=True,
        )

        print("Removing empty rows...")
        # remove empty rows
        sarcasm_df.dropna(inplace=True)

        # Some comments are missing, so we drop the corresponding rows.
        sarcasm_df.dropna(subset=["comment"], inplace=True)

        # Calculate the lengths of comments
        comment_lengths = [len(comment.split()) for comment in sarcasm_df["comment"]]

        # Calculate the mean, maximum, and minimum lengths
        mean_length = sum(comment_lengths) / len(comment_lengths)
        max_length = max(comment_lengths)
        min_length = min(comment_lengths)

        # Print the results
        print("Mean length:", mean_length)
        print("Maximum length:", max_length)
        print("Minimum length:", min_length)

        print("Removing comments with length > 50...")
        # Filter the dataframe to keep only comments with length <= 50
        mask = [length <= 50 for length in comment_lengths]
        sarcasm_df = sarcasm_df[mask]

        # Reset the index of the dataframe
        sarcasm_df.reset_index(drop=True, inplace=True)

        if use_sentiment_context:
            print("Adding sentiment columns...")
            # Add sentiment polarity and subjectivity columns
            sarcasm_df["sentiment_polarity"] = sarcasm_df["comment"].apply(
                lambda x: round(TextBlob(x).sentiment.polarity, 1)
            )
            sarcasm_df["sentiment_subjectivity"] = sarcasm_df["comment"].apply(
                lambda x: round(TextBlob(x).sentiment.subjectivity, 1)
            )

        # Save the dataframe
        sarcasm_df.to_csv("data/sarcasm_preprocessed_sentiments.csv", index=False)

        self.sarcasm_df = sarcasm_df

    def setup(self, stage=None):
        print("Setting up data...")
        # print("Value counts:", self.sarcasm_df["label"].value_counts())

        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.sarcasm_df["comment"],
        #     self.sarcasm_df["label"],
        #     test_size=test_size,
        #     random_state=random_state,
        # )

        # train_dataset = SARCDataset(X_train, y_train, tokenizer)
        # test_dataset = SARCDataset(X_test, y_test, tokenizer)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        if not use_sentiment_context:
            self.dataset = SARCDataset(
                self.sarcasm_df["comment"], self.sarcasm_df["label"], tokenizer
            )

        else:
            self.dataset = SARCDataset(
                (
                    self.sarcasm_df["comment"],
                    self.sarcasm_df["sentiment_polarity"],
                    self.sarcasm_df["sentiment_subjectivity"],
                ),
                self.sarcasm_df["label"],
                tokenizer,
            )

        # save the dataset using torch
        # torch.save(self.dataset, "preprocessed_dataset_sentiments.pt")

        # load the dataset
        # self.dataset = torch.load("preprocessed_dataset_sentiments.pt")

        # Split the dataset into train and test set
        total_size = len(self.dataset)
        train_size = int(TRAIN_SIZE * total_size)
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
