# SARCASM DETECTION

# BERT Folder Strucuture
This folder contains code for a fine-tuned BERT model that can accurately detect sarcasm in text. By leveraging the powerful capabilities of BERT

## bert.ipynb 
Jupyter Notebook for Sarcasm Detection using Fine-Tuned BERT.
This notebook provides a step-by-step implementation of our  sarcasm detection model. It is designed to guide you through the process of loading the pre-trained BERT model, fine-tuning it on sarcasm-labeled data, and evaluating its performance. We have also included sections explaining the data preprocessing, model architecture, and result analysis. 
### Outputs folder
Inside the "output" folder, you will find two  files:

"config.json":
    This file contains the saved parameters  of the fine-tuned BERT model.

"trainer_state.json":
    This file contains the results and performance metrics of the fine-tuned BERT model. It includes evaluation metrics such as accuracy, precision, recall, F1 score.
### Data folder
In the "Data" folder, which was not committed due to its size, you would typically expect to find the dataset used to train and test the sarcasm detection model.
Since the data was not committed, users of the project will need to obtain the data used in the project.
## Parameters
- Dataset path
- Epoch number
- MAX_SENTENCE_LENGTH
- test split size
- Huggingface trainer parameters
- Evaluation metrics
- output_dir
- evaluation_strategy steps or epochs
- eval_steps 
- per_device_train_batch_size
- per_device_eval_batch_size=256


