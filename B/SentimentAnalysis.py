import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

class SentimentAnalysis_class:

    #initialising class instance and calling the loading function
    def __init__(self, path_dataset):
        # Load the DataFrame
        self.df = pd.read_csv(path_dataset, encoding="utf-8-sig")
        self.clean_dataset()
    
    def clean_dataset(self):
        # Fill any NaN with "No statement given"
        self.df.iloc[:, 2] = self.df.iloc[:, 2].fillna('No statement given.')

        # Define phrases of absence of a last statement
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

        # Convert 'Last Statement' column to list
        self.last_statements = self.df.iloc[:, 2].tolist()

        # for statement in  self.last_statements:
        #     if len(statement) < 60 and statement:  # Ensure the statement is not empty
        #         print(statement)


    def plot_statement_length_distribution(self):
        # Calculate lengths of each statement
        lengths = [len(statement.split()) for statement in self.last_statements]
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=10, color='lightblue', alpha=0.7, range=(0, 400), edgecolor='black', rwidth=0.9)
        plt.title('Distribution of Statement Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.xlim(0, 400)  # Set the limits for x-axis
        plt.show()


    def perform_sentiment_analysis(self):

        device = 0 if torch.cuda.is_available() else -1  # device 0 for CUDA, -1 for CPU
        # Initialize a zero-shot classification pipeline 
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

        # Define the sentiment classes
        sentiment_classes = ["positive", "negative", "neutral"]

        # Classify
        results = [classifier(statement, candidate_labels=sentiment_classes) for statement in self.last_statements]
        return results

    def visualize_sentiment_results(self, results):
         # Extracting labels and scores
        labels = [result['labels'][0] for result in results]  # Get the top label 
        scores = [result['scores'][0] for result in results]  # Get the top score 

        # Create DataFrame 
        df_results = pd.DataFrame({'Sentiment': labels, 'Score': scores})

        # statistics for sentiment scores
        print("Descriptive Statistics for Sentiment Scores:")
        print(df_results.groupby('Sentiment')['Score'].describe())

        # Setting up the plot grid
        sns.set(style="whitegrid", palette="pastel", color_codes=True)
        plt.figure(figsize=(10, 5))  

        # First subplot - boxplot
        plt.subplot(1, 2, 1)  
        bp = sns.boxplot(x='Sentiment', y='Score', data=df_results, color='lightblue')
        for i, artist in enumerate(bp.artists):
            artist.set_edgecolor('black')

        plt.title('Box Plot of Sentiment Scores by Sentiment', fontsize=16)
        plt.xlabel('Sentiment', fontsize=14)
        plt.ylabel('Score', fontsize=14)

        # Second subplot - countplot
        plt.subplot(1, 2, 2)  
        cp = sns.countplot(x='Sentiment', data=df_results, color='lightblue')
        for patch in cp.patches:
            patch.set_edgecolor('black')

        plt.title('Distribution of Sentiments', fontsize=16)
        plt.xlabel('Sentiment', fontsize=14)
        plt.ylabel('Count', fontsize=14)

        plt.tight_layout()
        plt.show()

    
