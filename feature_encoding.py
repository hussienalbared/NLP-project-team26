import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import torch


def get_tfidf_encodings(
    x_train, x_test, ngram_range=(1, 2), max_features=50000, min_df=2
):
    # create tf-idf encoder
    tf_idf = TfidfVectorizer(
        ngram_range=ngram_range, max_features=max_features, min_df=min_df
    )
    # encode training data
    train_features = tf_idf.fit_transform(x_train)
    # encode test data
    test_features = tf_idf.transform(x_test)

    return train_features, test_features


# load the GloVe embeddings from a text file
def load_glove_embeddings(file):
    embeddings = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


# function that returns the mean of the word embeddings for a sentence
def glove_mean(sentence, embeddings, dim=100):
    words = sentence.split()
    vectors = np.zeros((len(words), dim))
    for i, word in enumerate(words):
        if word in embeddings:
            vectors[i] = embeddings[word]
    # check if vectors is empty or contains nan values
    if np.isnan(vectors).any() or len(vectors) == 0:
        # replace with zeros or random numbers
        return np.zeros(dim)
        # return np.random.rand(dim)
    else:
        return np.mean(vectors, axis=0)


glove_file = "glove.6B.100d.txt"  # path to the GloVe embeddings file


def get_glove_encodings(x_train, x_test, y_train, y_test, dim=100):
    """
    Function that returns the GloVe encodings for the training and test data.
    Dim should be the same as the dimension of the GloVe embeddings (glove_file).

    Returns:
        X_train_glove: PyTorch tensor of shape (num_train_examples, dim)
        X_test_glove: PyTorch tensor of shape (num_test_examples, dim)
    """
    # load the GloVe embeddings
    glove_embeddings = load_glove_embeddings(glove_file)
    # encode the training and test data using GloVe mean-pooling and convert them to PyTorch tensors
    X_train_glove = torch.tensor(
        [glove_mean(sent, glove_embeddings, dim=dim) for sent in x_train],
        dtype=torch.float32,
    )
    X_test_glove = torch.tensor(
        [glove_mean(sent, glove_embeddings, dim=dim) for sent in x_test],
        dtype=torch.float32,
    )

    # convert Y_train and Y_test to PyTorch tensors
    Y_train_glove = torch.tensor(y_train.values, dtype=torch.long)
    Y_test_glove = torch.tensor(y_test.values, dtype=torch.long)

    return X_train_glove, X_test_glove, Y_train_glove, Y_test_glove
