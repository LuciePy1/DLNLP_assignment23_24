
import re
import nltk
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


class HopeDetection_deathRow:

    #initialising class instance and calling the loading function
    def __init__(self, path_dataset):
        # Load the DataFrame
        self.df = pd.read_csv(path_dataset, encoding="utf-8-sig")
        self.clean_dataset()
    
    def clean_dataset(self):
        # Fill any NaN with "No statement given"
        self.df.iloc[:, 2] = self.df.iloc[:, 2].fillna('No statement given.')

        # Define phrases of an absence of a last statement
        absence_of_last_statements = {
            'No',
            'None',
            'None.',
            'This offender declined to make a last statement.',
            'This inmate declined to make a last statement.',
            'No last statement given.',
            'No statement was made.',
            'No statement given.',
            'Last Statement',
            'No last statement.',
            'Spoken: No.'
        }

        # Remove entries that match the absence statements 
        self.df = self.df[~self.df.iloc[:, 2].apply(lambda x: x.strip() in absence_of_last_statements)]
        # Print the length of the dataframe after removal
        # print(f"Number of entries after removal: {len(self.df)}")

        # Convert the 'Last Statement' column to a list
        self.last_statements = self.df.iloc[:, 2].tolist()

        # for statement in  self.last_statements:
        #     if len(statement) < 60 and statement:  # Ensure the statement is not empty
        #         print(statement)

    def data_visualisation(self):
        # process the text for NLP analysis
        self.processed_statements = self.process_text(self.last_statements)
        # Calculate frequency distribution
        self.frequency_distribution = self.calculate_frequency(self.processed_statements)

    def process_text(self, statements):
        processed_statements = []
        lemmatizer = WordNetLemmatizer()
        my_stopwords = set(stopwords.words('english') + ['offender','declined','statement','none'])

        for statement in statements:
            # Cleaning and tokenization
            tokens = word_tokenize(re.sub("[^a-zA-Z]", " ", statement.lower()))
            # Removing stopwords and lemmatizing
            clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in my_stopwords]
            processed_statements.append(" ".join(clean_tokens))

        return processed_statements

    def calculate_frequency(self, processed_statements):
        all_words = []
        for text in processed_statements:
            all_words.extend(word_tokenize(text))
        word_freq_dist = nltk.FreqDist(all_words)
        mostcommon = word_freq_dist.most_common(350)
        # convert it into dict structure to create a word cloud 
        mostcommon = dict((x, y) for x, y in mostcommon)
        print (mostcommon)
        #words cloud for most 250 frequent word
        wordcloud_q = WordCloud(width=1600, height=800,
                                background_color='white',
                                stopwords=set(STOPWORDS),
                                max_words=250,
                                ).generate_from_frequencies(mostcommon)
        fig = plt.figure(1, figsize=(20,20))
        plt.imshow(wordcloud_q)
        plt.axis('off')
        plt.show()

        return word_freq_dist
    
    def prepare_death_row_data(self):
        # Preprocess the data
        nltk.download('stopwords')
        stop_words = stopwords.words('english')

        processed_texts = []
        for statement in self.last_statements:
            # Convert to lowercase
            words = statement.lower().split()
            # Remove stopwords and any non-word characters
            filtered_words = [re.sub(r"[^\w\s]", '', word) for word in words if word.lower() not in stop_words]
            # Join words to form the cleaned up statement
            cleaned_statement = ' '.join(filtered_words)
            processed_texts.append(cleaned_statement)

        # # Print sample of processed statements to check the output
        # print("Sample processed statements:")
        # for sample in processed_texts[:5]: #first 5
        #     print(sample)
        #     print("")
        
        # Tokenize the text
        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        tokenized_data = [
            tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=60,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            ) for text in processed_texts
        ]

        # Extract input IDs and attention masks
        input_ids = torch.cat([item['input_ids'] for item in tokenized_data], dim=0)
        attention_masks = torch.cat([item['attention_mask'] for item in tokenized_data], dim=0)

        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        return dataloader
    
    def prepare_and_predict(self, model, device):
        # Tokenize and prepare DataLoader
        dataloader = self.prepare_death_row_data()

        # Prediction
        model.eval()  # Set the model to evaluation mode
        all_logits = []

        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)  # to the device (GPU/CPU)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits
                all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Applying softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(all_logits, dim=1)
        probabilities = probabilities.detach().cpu().numpy()  # Move probabilities to CPU

        predictions = np.argmax(probabilities, axis=1)

        # Build a DataFrame for visualization
        results_df = pd.DataFrame({
            'Statement': self.last_statements,
            'Predicted Class': predictions,
            'Probability Non-hope': probabilities[:, 0],
            'Probability Hope': probabilities[:, 1]
        })

        return results_df
    
    def apply_transfered_model(self):
        self.clean_dataset()

        # Load the model
        model_path = 'A/roberta_model_no_weights'
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare data and predict
        results_df = self.prepare_and_predict(model, device)

        # Display the results
        # pd.set_option('display.max_colwidth', None)  # Set to None to display the full text of columns
        # print(results_df.head())  # Print the first few results for review

        # Calculate the count of each predicted class and visualize
        class_counts = results_df['Predicted Class'].value_counts()
        plt.figure(figsize=(8, 6))  

        # First subplot for 'Hope' probability distribution
        plt.subplot(1, 2, 1)  
        sns.boxplot(data=results_df[results_df['Predicted Class'] == 1]['Probability Hope'], color='lightblue')
        plt.title('Probability Distribution for "Hope"')
        plt.ylabel('Probability')

        # Second subplot for 'Non-hope' probability distribution
        plt.subplot(1, 2, 2) 
        sns.boxplot(data=results_df[results_df['Predicted Class'] == 0]['Probability Non-hope'], color='lightblue')
        plt.title('Probability Distribution for "Non-hope"')
        plt.ylabel('Probability')

        plt.tight_layout() 
        plt.show()
        pd.reset_option('display.max_colwidth')

        # Print examples of 'Hope' and 'Non-hope' statements
        hope_examples = results_df[results_df['Predicted Class'] == 1].head(3)
        non_hope_examples = results_df[results_df['Predicted Class'] == 0].head(3)

        print("\nExamples of 'Hope' statements:")
        for index, row in hope_examples.iterrows():
            print(f" - {row['Statement']} (Prob: {row['Probability Hope']:.2f})")

        print("\nExamples of 'Non-hope' statements:")
        for index, row in non_hope_examples.iterrows():
            print(f" - {row['Statement']} (Prob: {row['Probability Non-hope']:.2f})")

        # Calculate the count of each predicted class
        class_counts = results_df['Predicted Class'].value_counts()
        print("Distribution of Predicted Classes:")
        print(class_counts)

        # Filter results for 'Hope' predictions 
        hope_results_df = results_df[results_df['Predicted Class'] == 1]
        # Filter results for 'Non-hope' predictions 
        non_hope_results_df = results_df[results_df['Predicted Class'] == 0]

        # Calculate avg and sd for 'Hope' 
        average_probability_hope = hope_results_df['Probability Hope'].mean()
        std_dev_probability_hope = hope_results_df['Probability Hope'].std()

        # Calculate avg and sd 'Non-hope' 
        average_probability_non_hope = non_hope_results_df['Probability Non-hope'].mean()
        std_dev_probability_non_hope = non_hope_results_df['Probability Non-hope'].std()

        # Print the results
        print("\nAverage Probability for 'Hope' Class:")
        print(average_probability_hope)
        print("\nStandard Deviation for 'Hope' Class:")
        print(std_dev_probability_hope)

        print("\nNumber of 'Hope' Predictions:")
        print(hope_results_df.shape[0])

        print("\nAverage Probability for 'Non-hope' Class:")
        print(average_probability_non_hope)
        print("\nStandard Deviation for 'Non-hope' Class:")
        print(std_dev_probability_non_hope)

        print("\nNumber of 'Non-hope' Predictions:")
        print(non_hope_results_df.shape[0])



