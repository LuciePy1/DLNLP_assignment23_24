# DLNLP Assignment 2023 - 2024

The goal of this project is to explores the application of deep learning techniques in detecting and analysing emotional sentiment, focusing on two distinct tasks: A: hope speech detection in user-generated content and B: analysis of death row inmates last statements with transfer learn.

This project folder `DNLP_23-24_SN20099306` you will find:
- A folder `A`: this folder contains contains the model file codes for Task A:
  - `HopeEDI_Bert_model.py` which contains the fine-tuning BERT model Class and corresponding functions to code and execute model (including from scratch or from pre-saved as well as with and without set-up of class weights.
  - `HopeEDI_Roberta_model.py` which contains the fine-tuning ROBERTA model Class and corresponding functions to code and execute model (including from scratch or from pre-saved as well as with and without set-up of class weights.
- A folder `B`: this folder contains contains the model file codes for Task  B:
  - `Death_row_data_scraping.py` which contains functions to collect the dataset for Task B from the Texas Death Row information.
  - `SentimentAnalysis.py` This class contains the functions to run the model of sentiment analysis on the dataset.
  - `Transfer_learning_deathRow.py` This class contains the functions to apply, run and test the models from Task A through Transfer Learning to the death row data.
- `Datasets`: This folder contains the datasets for Task A: english_hope_dev.csv and english_hope_train.csv as well as the dataset for Task B: death_row_information.csv
- `main.py`: this file runs the MENU to select which tasks to run
- `README.md`: This file describing the assignment and guidance to run the code
  
# Packages required 
- `numpy`: For calculations and numerical functions
- `matplotlib`: For data visualisation
- `seaborn`: For advanced data visualisation (eg: confusion matrix)
- `nltk`: A Natural Language Toolkit, for text preprocessing, including tokenization, stopword removal...
- `torch`: the PyTorch library, needed for building and training deep learning models on tensors.
- `torch.nn`: to create custom neural network layers.
- `TensorDataset`, `DataLoader`, `RandomSampler`, `SequentialSampler`: Classes from PyTorch for efficient data handling and batching during model training.
- `AutoTokenizer`, `RobertaTokenizer`, BertTokenizer: From the Hugging Face `transformers` library, used for tokenizing text into format suitable for transformer models.
- `RobertaForSequenceClassification`, `BertForSequenceClassification`: Transformer models from Hugging Face for sequence classification tasks.
- `AdamW`: the optimizer from PyTorch, modified version of Adam, suited for training deep learning models.
- `get_linear_schedule_with_warmup`: A scheduler from Hugging Face to adjust learning rate dynamically during training.
- `compute_class_weight`: Utility from `sklearn` to handle class imbalance by computing weights for each class.
- `WordCloud`, `STOPWORDS`: From the `wordcloud` library, for generating visual representations of word frequencies.
- `pipeline`: From Hugging Face `transformers` for creating processing pipelines for transformer models.
- `requests`:   for sending HTTP requests, useful for data scraping
- `BeautifulSoup`: Library from `bs4` for parsing HTML and XML documents, used for web scraping to create the dataset of TaskB


# How to Run 
The entire assignment can be tested from main.py but make sure to do the following:
1. Check the packages required are installed
2. Check the dataset are all  in the folder `Datasets`
3. Run `main.py` to initiate the program. The following menu will appear:
     Please Select an option to run a Task:
      1. Task A - Hope Speech Detection with Transformer models
      2. Task B - Transfer learning & Analysis of Death Row Statements
      3. Exit

      IF you run TASK A, a subsequent menu appears:
      0. Visualise Dataset
      1. Train/Test Bert model from scratch
      2. Train/Test Roberta model from scratch
      3. Train/Test Bert model with balanced weights from scratch
      4. Run pre-saved Bert model(note please run 1. first)
      5. Run pre-saved Roberta model (please run 2. first)
      6. Run pre-saved Bert balanced model (please run 3. first)
  
      IF you run TASK B, a subsequent menu appears:
      0. Visualise Dataset
      1. Re-collect dataset from Texas Department Death Row info page
      2. Transfer Learning - Sentiment Analysis
      3. Transfer Learning - HopeSpeech Detection (please run TASK A - 2 first)

The options are quite straightforward and indicate which part of the project is ran. Input a number to run the corresponding Task and associated model.
Please note that the models were too heavy to load onto github as pre-trained, therefore to run TASK A with pre-saved models, you first need to train them from scratch. Again, to run TASK B Part 3, transfer learning of hope detection, please make sure to run TASK A, ROBERTA model first.
