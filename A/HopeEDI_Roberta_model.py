import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

class Hope_with_Roberta_class:

    #initialising class instance and calling the loading function
    def __init__(self, path_train, path_dev):

        # Setting seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        #calling the load, clean and split function
        self.load_clean_split(path_train, path_dev)

    def load_clean_split(self, path_train, path_dev):
        #loading from datapath
        eng_train = pd.read_csv(path_train, sep='\t', names=['text', 'labels', 'del']).drop(columns=['del'])
        eng_val = pd.read_csv(path_dev, sep='\t', names=['text', 'labels', 'del']).drop(columns=['del'])

        # Filter out sentences with the label "not english"
        eng_train = eng_train[eng_train['labels'] != 'not-English']
        eng_val = eng_val[eng_val['labels'] != 'not-English']

        df = pd.concat([eng_train, eng_val])  #combine both
        train_ratio = 0.80
        validation_ratio = 0.10
        test_ratio = 0.10

        #validation and test sizes from the split ratios
        val_size = validation_ratio / (test_ratio + validation_ratio)

        # First split to separate out the training set
        train_data, test_and_val_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
        # Second split to separate out the validation and test sets
        validation_data, test_data = train_test_split(test_and_val_data, test_size=val_size, random_state=42)

        # store these as instance variables
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        print(f"Training set size: {len(self.train_data)}")
        print(f"Validation set size: {len(self.validation_data)}")
        print(f"Test set size: {len(self.test_data)}")


    def preprocess(self):
        #preprocess by removing stopwords, characters special and lowcasing everywords
        nltk.download('stopwords')
        stop_words = stopwords.words('english')

        #training data
        self.train_data['text'] = self.train_data['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
        self.train_data['text'] = self.train_data['text'].str.replace(r'[^\w\s]', ' ', regex=True)

        #validation data
        self.validation_data['text'] = self.validation_data['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
        self.validation_data['text'] = self.validation_data['text'].str.replace(r'[^\w\s]', ' ', regex=True)

        #test data
        self.test_data['text'] = self.test_data['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
        self.test_data['text'] = self.test_data['text'].str.replace(r'[^\w\s]', ' ', regex=True)


    def tokenize(self): # tokenise + label encoding

        # Load the tokenizer for the pre-trained roberta model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Tokenize the text
        def tokenize_data(text):
            return tokenizer.encode_plus(
                text,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]' at the begining and end
                max_length=60,          # Maximum length is 60 and truncate/pad to this max
                padding='max_length',    # Pad to max_length
                truncation=True,         # Truncate to max_length
                return_attention_mask=True, #creates attention masks
                return_tensors='pt',     # Return PyTorch tensors
            )

        #Apply tokenization to our data
        self.train_data['encoded'] = self.train_data['text'].apply(lambda x: tokenize_data(x))
        self.validation_data['encoded'] = self.validation_data['text'].apply(lambda x: tokenize_data(x))
        self.test_data['encoded'] = self.test_data['text'].apply(lambda x: tokenize_data(x))

        # Extract components from encoded texts
        input_ids = torch.cat([item['input_ids'] for item in self.train_data['encoded']], dim=0)
        attention_masks = torch.cat([item['attention_mask'] for item in self.train_data['encoded']], dim=0)

        input_ids_val = torch.cat([item['input_ids'] for item in self.validation_data['encoded']], dim=0)
        attention_masks_val = torch.cat([item['attention_mask'] for item in self.validation_data['encoded']], dim=0)

        input_ids_test = torch.cat([item['input_ids'] for item in self.test_data['encoded']], dim=0)
        attention_masks_test = torch.cat([item['attention_mask'] for item in self.test_data['encoded']], dim=0)

        #label encoding
        labels={'Non_hope_speech': 0,'Hope_speech': 1,}

        self.train_data['labels']= self.train_data['labels'].map(labels)
        self.validation_data['labels'] = self.validation_data['labels'].map(labels)
        self.test_data['labels'] = self.test_data['labels'].map(labels)

        labels_data_train = torch.tensor(self.train_data.labels.values)
        labels_data_val = torch.tensor(self.validation_data.labels.values)
        labels_data_test = torch.tensor(self.test_data.labels.values)

        #creating tensor dataset from all components
        dataset_train = TensorDataset(input_ids, attention_masks, labels_data_train)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_data_val)
        dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_data_test)

        # Create DataLoader for training set
        self.train_dataloader = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train), # select random samples
            batch_size=32 #recommanded 16 to 32
        )

        # Create DataLoader for validation set
        self.validation_dataloader = DataLoader(
            dataset_val,
            sampler=RandomSampler(dataset_val), 
            batch_size=32 #recommanded 16 to 32
        )

        # Create DataLoader for training set
        self.test_dataloader = DataLoader(
            dataset_test,
            sampler=SequentialSampler(dataset_test), #Sequential Sampling recommanded when testing
            batch_size=32 #recommanded 16 to 32
        )

    def model_setup (self):
        # Load roberta with a classification
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base',
            num_labels = 2,  #  number of output labels--2 for binary classification.
            output_attentions = False,  # Whether the model returns attentions weights.
            output_hidden_states = False,  # Whether the model returns all hidden-states.
        )

        #  Set up device and move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(self.device)

        # Training configuration
        self.optimizer = AdamW(self.model.parameters(),
                  lr = 2e-5,  # learning_rate
                  eps = 1e-8  # adam_epsilon
                )

        # Schedule the learning rate
        self.num_epochs = 3 #recommanded 2 to 4
        total_steps = len(self.train_dataloader) * self.num_epochs # Total number of training steps

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, # Default value
                                            num_training_steps = total_steps)

    def model_setup_balanced (self):
        # Load BERT with a classification
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base',
            num_labels = 2,  # for binary classification.
            output_attentions = False,  #  attentions weights.
            output_hidden_states = False,  #  all hidden-states.
        )

        #  Set up device and move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(self.device)
        
        # Compute class weights
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(self.train_data['labels']),
                                             y=self.train_data['labels'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Set up the criterion with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training configuration
        self.optimizer = AdamW(self.model.parameters(),
                  lr = 2e-5,  # args.learning_rate
                  eps = 1e-8  # args.adam_epsilon
                )

        # Schedule the learning rate
        self.num_epochs = 3 #recommanded 2 to 4
        total_steps = len(self.train_dataloader) * self.num_epochs # Total number of training steps

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, # Default value
                                            num_training_steps = total_steps)

    
    def train(self):
        self.model.train()  # Set model to training mode
        total_train_loss = 0
        total_train_accuracy = 0

        for step, batch in enumerate(self.train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)  # Move batch to the appropriate device
            b_input_ids, b_input_mask, b_labels = batch

            self.model.zero_grad()  # Clear previous gradients

            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()  # Perform backpropagation
            self.optimizer.step()  # Update parameters
            self.scheduler.step()  # Update learning rate

            total_train_loss += loss.item()

            #training acuracy
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += np.mean(np.argmax(logits, axis=1) == label_ids)

            # Print loss every batch
            print(f'Batch {step+1}/{len(self.train_dataloader)}, Loss: {loss.item()}')

        average_train_loss = total_train_loss / len(self.train_dataloader)  # Calculate the average loss over the training data
        average_train_accuracy = total_train_accuracy / len(self.train_dataloader) #average accuracy

        return average_train_loss, average_train_accuracy

    def train_with_weights(self):
        self.model.train()  # Set model to training mode
        total_train_loss = 0
        total_train_accuracy = 0

        for step, batch in enumerate(self.train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)  # Move batch to the appropriate device
            b_input_ids, b_input_mask, b_labels = batch

            self.model.zero_grad()  # Clear previous gradients

            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Compute loss using the criterion that has been adjusted for class weights
            loss = self.criterion(outputs.logits, b_labels)
            loss.backward()  # Perform backpropagation
            self.optimizer.step()  # Update parameters
            self.scheduler.step()  # Update learning rate

            total_train_loss += loss.item()

            #training acuracy
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += np.mean(np.argmax(logits, axis=1) == label_ids)

            # Print loss every batch
            print(f'Batch {step+1}/{len(self.train_dataloader)}, Loss: {loss.item()}')

        average_train_loss = total_train_loss / len(self.train_dataloader)  # Calculate the average loss over the training data
        average_train_accuracy = total_train_accuracy / len(self.train_dataloader) #average accuracy

        return average_train_loss, average_train_accuracy

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in self.validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():  # Do not compute gradients
               outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            total_eval_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += np.mean(np.argmax(logits, axis=1) == label_ids)

        average_eval_loss = total_eval_loss / len(self.validation_dataloader)
        average_eval_accuracy = total_eval_accuracy / len(self.validation_dataloader)

        return average_eval_loss, average_eval_accuracy

    def training_pipeline_no_weights(self):
        
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-------------------------------')

            train_loss, train_accuracy = self.train()
            print(f'Train loss: {train_loss}')
            print(f'Train Accuracy: {train_accuracy}')
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            #  validation 
            val_loss, val_accuracy = self.validate()
            print(f'Validation Loss: {val_loss}')
            print(f'Validation Accuracy: {val_accuracy}')
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        #saving model
        self.model.save_pretrained('./roberta_model_no_weights')
        print("Model saved")

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot for Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'o-', label='Train Loss', linewidth=2, markersize=5)
        plt.plot(val_losses, 'o-', color='red', label='Validation Loss', linewidth=2, markersize=5)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot for Training and Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'o-', label='Train Accuracy', linewidth=2, markersize=5)
        plt.plot(val_accuracies, 'o-', color='red', label='Validation Accuracy', linewidth=2, markersize=5)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def training_pipeline_with_weights(self):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-------------------------------')

            train_loss, train_accuracy = self.train_with_weights()
            print(f'Train loss: {train_loss}')
            print(f'Train Accuracy: {train_accuracy}')
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            #  validation 
            val_loss, val_accuracy = self.validate()
            print(f'Validation Loss: {val_loss}')
            print(f'Validation Accuracy: {val_accuracy}')
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        #saving model
        self.model.save_pretrained('/content/drive/MyDrive/DeepLearning/Model_BALANCED_roberta_FINAL_lr_2e-5_3ep')
        print("Model saved")

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot for Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'o-', label='Train Loss', linewidth=2, markersize=5)
        plt.plot(val_losses, 'o-', color='red', label='Validation Loss', linewidth=2, markersize=5)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot for Training and Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'o-', label='Train Accuracy', linewidth=2, markersize=5)
        plt.plot(val_accuracies, 'o-', color='red', label='Validation Accuracy', linewidth=2, markersize=5)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def test(self):
        self.model.eval()  # Set model to evaluation mode
        total_eval_loss = 0

        # Initialize lists to store true and predicted labels for all batches
        all_labels = []
        all_preds = []

        for batch in self.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():  # Do not compute gradients
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            total_eval_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1)

            # Append batch prediction results
            all_labels.extend(label_ids)
            all_preds.extend(preds)

        average_eval_loss = total_eval_loss / len(self.test_dataloader)

        # Compute accuracy and other classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['Non_hope_speech', 'Hope_speech'])

        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hope Speech', 'Hope Speech'], yticklabels=['Non-Hope Speech', 'Hope Speech'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

        return average_eval_loss, accuracy, report

    def run_fromsaved(self, path):
      # Path where the model was saved

      #follow standard preprocessing steps
      self.preprocess()
      self.tokenize()
      self.model_setup()

      # Load the model
      self.model = RobertaForSequenceClassification.from_pretrained(path)
      # Assuming you are using the same device (e.g., CUDA) as before
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model.to(self.device)  # Move model to the appropriate device

      test_loss, test_accuracy, test_report = self.test() #test set evaluation
      print(f'Test Loss: {test_loss}')
      print(f'Test Accuracy: {test_accuracy}')
      print("Test Classification Report:")
      print(test_report)

    def run_fromscratch_no_weights(self):
      #follow steps of:
      self.preprocess()
      self.tokenize()
      self.model_setup()
      self.training_pipeline_no_weights() #function to fun from scratch
      test_loss, test_accuracy, test_report = self.test() #test set evaluation
      print(f'Test Loss: {test_loss}')
      print(f'Test Accuracy: {test_accuracy}')
      print("Test Classification Report:")
      print(test_report)

    def run_fromscratch_balanced(self):
      #follow steps of:
      self.preprocess()
      self.tokenize()
      self.model_setup_balanced() #set up with weights 
      self.training_pipeline_with_weights() #function to fun from scratch with weights
      test_loss, test_accuracy, test_report = self.test() #test set evaluation
      print(f'Test Loss: {test_loss}')
      print(f'Test Accuracy: {test_accuracy}')
      print("Test Classification Report:")
      print(test_report)