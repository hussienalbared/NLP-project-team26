import numpy as np
import pytorch_lightning as pl
import torch
from config import *
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from transformers import RobertaModel


class SARCClassifier(nn.Module):
    def __init__(self, base_model):
        super(SARCClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][
            :, 0
        ]
        x = self.fc1(bert_out)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class SarcasmDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        base_model = RobertaModel.from_pretrained("roberta-base")
        self.model = SARCClassifier(base_model)

        self.learning_rate = learning_rate

        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        train_input, train_label = batch
        attention_mask = train_input["attention_mask"]
        input_ids = train_input["input_ids"].squeeze(1)

        output = self(input_ids, attention_mask)
        loss = self.criterion(output, train_label.float().unsqueeze(1))

        preds = (output >= threshold).int()
        acc = (preds == train_label.int().unsqueeze(1)).sum().item()
        f1 = f1_score(
            train_label.cpu().detach().numpy().flatten(),
            preds.cpu().detach().numpy().flatten(),
        )
        auc = roc_auc_score(
            train_label.cpu().detach().numpy().flatten(),
            output.cpu().detach().numpy().flatten(),
        )

        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)
        self.log(
            "train_acc", acc / len(train_label), sync_dist=True, batch_size=batch_size
        )
        self.log("train_f1", f1, sync_dist=True, batch_size=batch_size)
        self.log("train_auc", auc, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        val_input, val_label = batch
        attention_mask = val_input["attention_mask"]
        input_ids = val_input["input_ids"].squeeze(1)

        output = self(input_ids, attention_mask)
        loss = self.criterion(output, val_label.float().unsqueeze(1))

        preds = (output >= threshold).int()
        acc = (preds == val_label.int().unsqueeze(1)).sum().item()
        f1 = f1_score(
            val_label.cpu().detach().numpy().flatten(),
            preds.cpu().detach().numpy().flatten(),
        )
        auc = roc_auc_score(
            val_label.cpu().detach().numpy().flatten(),
            output.cpu().detach().numpy().flatten(),
        )

        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        self.log("val_acc", acc / len(val_label), sync_dist=True, batch_size=batch_size)
        self.log("val_f1", f1, sync_dist=True, batch_size=batch_size)
        self.log("val_auc", auc, sync_dist=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
