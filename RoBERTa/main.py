import os

from config import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import argparse

import pytorch_lightning as pl
import torch
import wandb
from data_module import SarcasmDetectionDataModule, SARCDataset
from model import SarcasmDetectionModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Set a random seed for reproducibility
pl.seed_everything(42, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["train", "test", "predict"],
    default="train",
    help="Mode of running: train or test",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint file to load or save",
)
args = parser.parse_args()

model = SarcasmDetectionModel()

data_module = SarcasmDetectionDataModule(
    data_file=data_file,
    batch_size=batch_size,
    num_workers=num_workers,
    mode=args.mode,
)

# logger = TensorBoardLogger(save_dir="lightning_logs")

wandb.login()
wandb_logger = WandbLogger()
wandb_logger = WandbLogger(
    name="RoBERTa-max_length-50-special-tokens",
    project="Sarcasm-Detection",
    log_model=True,
)

checkpoint_callback = ModelCheckpoint(
    dirpath="trained_weights",
    filename="model-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING, verbose=True, mode="min"
)


accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Parse the value of CUDA_VISIBLE_DEVICES to get a list of GPU ids
gpu_ids = [int(id) for id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

strategy = "ddp_find_unused_parameters_true" if len(gpu_ids) > 1 else "auto"

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=len(gpu_ids),
    strategy=strategy,
    max_epochs=num_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

if args.mode == "train":
    trainer.fit(model, data_module)
elif args.mode == "test":
    trainer.test(model, data_module, ckpt_path=args.checkpoint)
elif args.mode == "predict":
    trainer.predict(model, data_module, ckpt_path=args.checkpoint)
