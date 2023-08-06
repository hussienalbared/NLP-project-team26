import numpy as np
from preprocessing import preprocess
from torch.utils.data import Dataset


class SARCDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        texts = X

        texts = [preprocess(text) for text in texts]

        self._print_random_samples(texts)

        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=150,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

        self.labels = y

    def _print_random_samples(self, texts):
        np.random.seed(42)
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
