# %%
# !pip install numpy requests nlpaug
# !pip install tensformers
# !pip install evaluate
# !pip install tensorboard
# !pip install accelerate -U
# !pip uninstall pillow
# !pip install pillow==9.4.0
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import evaluate
from sklearn.model_selection import train_test_split
import os
import shutil
from transformers import  DataCollatorWithPadding
from datasets import  load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertTokenizer,BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import time
import datetime
from transformers import get_linear_schedule_with_warmup
import random
import json
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

# %%
torch.__version__
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%

torch.cuda.empty_cache()


# %%
import gc
torch.cuda.empty_cache()
gc.collect()

# %%
MAX_SENTENCE_LENGTH=150
batch_size = 32
EPOCHS=3

# %%
df=pd.read_csv('../train-balanced-sarcasm.csv')


# %%

def getlen(x):
    return len(x)


df["lengths"]=df["comment"].astype(str).map(getlen)
filtered_rows = df[df["lengths"] < MAX_SENTENCE_LENGTH]
labels=filtered_rows["label"].to_list()
comments=filtered_rows["comment"].astype(str).to_list()


# %%
len(comments),len(labels)

# %%

# x_train, x_test_valid, y_train, y_test_valid = train_test_split(comments, labels, test_size=0.33, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(comments, labels, test_size=0.3, random_state=42)

# x_test, x_valid, y_test, y_valid = train_test_split(x_test_valid, y_test_valid, test_size=0.5, random_state=42)

# %%
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# %%
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False
)

model.to(device)
     
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8
                )

# %% [markdown]
# 

# %%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(pred_flat,labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# %%
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# %%
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
      self.encodings=encodings
      self.labels=labels   
    def __len__(self):
        return len(self.encodings)    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item                   



# %%
metric = load_metric("accuracy")
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(eval_pred)
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)


def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

# %%
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
# For every sentence...
# for sent in comments:

#     encoded_dict = tokenizer.encode_plus(
#                        sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]
#                         pad_to_max_length = True,
#                         max_length=512,
#                         truncation=True,
#                         return_attention_mask = True,   # Construct attn. masks.
#                         return_tensors = 'pt'     # Return pytorch tensors.
#                    )
    
#     # Add the encoded sentence to the list.    
#     input_ids.append(encoded_dict['input_ids'])
    
#     attention_masks.append(encoded_dict['attention_mask'])

# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# labels = torch.tensor(labels)

# %%

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)
# Calculate the number of samples to include in each set.
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# %%


train_dataloader = DataLoader(
            train_dataset,  
            batch_size = batch_size, # Trains with this batch size.
            shuffle=True
        )
validation_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size # Evaluate with this batch size.
        )

# %%
def save_model(model,path):
 torch.save(model.state_dict(), path)
def save_stats(training_stats,file_path):
    with open(file_path, 'w') as json_file:
        json.dump(training_stats, json_file) 

# %%

total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# %%
def train_epoch(model, train_loader, optimizer,scheduler, epoch):
        loss_list = []
        print("")
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_loader):
            # Progress update every 40 batches.
            if step % 1000 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            output = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss, logits=output.loss,output.logits
            loss_list.append(loss.item())
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)
        print(f"Epoch: {epoch+1} ")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))    
        mean_loss = np.mean(loss_list)
        return mean_loss, loss_list        
    

# %%
@torch.no_grad()
def eval_model(model, validation_dataloader):
        # correct = 0
        # total = 0
        loss_list = []
        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)            
            output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits=output.loss,output.logits  
            total_eval_loss += loss.item()
            loss_list.append(loss.item())
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
        # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
        # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)
            
        
        loss = np.mean(loss_list)
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_validation_precision=sum(val_precision)/len(val_precision) if len(val_precision)>0 else '\t - Validation Precision: NaN'
        avg_validation_recall=sum(val_recall)/len(val_recall) if len(val_recall)>0 else '\t - Validation Recall: NaN'
        avg_validation_specificity=sum(val_specificity)/len(val_specificity) if len(val_specificity)>0 else '\t - Validation Specificity: NaN'
        #
        
        #
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
#        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
        return avg_val_accuracy, loss,avg_validation_precision,avg_validation_recall,avg_validation_specificity


# %%
def train_model(model, optimizer, scheduler, train_loader, valid_loader, num_epochs):   
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []
    valid_spe = []
    valid_recall = []
    valid_prec= []
    
    for epoch in range(num_epochs): 
 
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer, 
                epoch=epoch,scheduler=scheduler
            ) 
        accuracy, loss,avg_validation_precision,avg_validation_recall,avg_validation_specificity = eval_model(
                    model=model, validation_dataloader=valid_loader
            )
        valid_acc.append(accuracy)
        valid_spe.append(avg_validation_specificity)
        valid_recall.append(avg_validation_recall) 
        valid_prec .append(avg_validation_precision)
        
        val_loss.append(loss)       
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print(f"    Accuracy: {accuracy}%")
        print("\n")   
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, valid_acc,valid_spe,valid_recall,valid_prec

# %%
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# %%
# train_loss, val_loss, loss_iters, valid_acc,valid_spe,valid_recall,valid_prec=train_model(model=model,optimizer=optimizer,scheduler=scheduler,train_loader=train_dataloader,
#             valid_loader=validation_dataloader,num_epochs=EPOCHS)

# %% [markdown]
# 

# %%
training_stats={"train_loss":train_loss, "val_loss":val_loss, "loss_iters":loss_iters,"valid_acc": valid_acc}

# %%
save_stats(file_path="stats_training.json",training_stats=training_stats)

# %%
save_model(model=model,path="model21.pth")

# %%


# %%
def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f




# %%
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1,3)
fig.set_size_inches(24,5)

smooth_loss = smooth(loss_iters, 31)
ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
ax[0].legend(loc="best")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("CE Loss")
ax[0].set_title("Training Progress")

epochs = np.arange(len(train_loss)) + 1
ax[1].plot(epochs, train_loss, c="red", label="Train Loss", linewidth=3)
ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
ax[1].legend(loc="best")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("CE Loss")
ax[1].set_title("Loss Curves")

epochs = np.arange(len(val_loss)) + 1
ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
ax[2].legend(loc="best")
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Accuracy (%)")
ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")

plt.show()



# %%
del model

# %% [markdown]
# ######## TEST

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback


# Read data
# data = pd.read_csv("train.csv")
df=pd.read_csv('../train-balanced-sarcasm.csv')
def getlen(x):
    return len(x)


df["lengths"]=df["comment"].astype(str).map(getlen)
filtered_rows = df[df["lengths"] < MAX_SENTENCE_LENGTH]
labels=filtered_rows["label"].to_list()
comments=filtered_rows["comment"].astype(str).to_list()

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(comments)
y = list(labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()


# %%



