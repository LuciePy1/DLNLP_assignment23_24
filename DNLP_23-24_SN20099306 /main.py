# main.py
import pandas as pd
from A.HopeEDI_Bert_model import Hope_with_BERT_class
from A.HopeEDI_Roberta_model import Hope_with_Roberta_class
from B.Death_row_data_scraping import data_acquisition_last_statements
from B.Transfer_learning_deathRow import HopeDetection_deathRow
from B.SentimentAnalysis import SentimentAnalysis_class

def main():
        
    # Specify the path to the datasets
    hopeEDI_train_path = 'Datasets/english_hope_train.csv'
    hopeEDI_dev_path ='Datasets/english_hope_dev.csv'
    deathrow_path = "Datasets/death_row_information.csv"

    # Create an instance for each
    hope_bert = Hope_with_BERT_class(hopeEDI_train_path, hopeEDI_dev_path)
    hope_roberta = Hope_with_Roberta_class(hopeEDI_train_path, hopeEDI_dev_path)
    death_row_hope = HopeDetection_deathRow(deathrow_path)
    sa = SentimentAnalysis_class(deathrow_path)

    while True:
        # Display menu    
        print("DEEP LEARNING FOR NLP ASSIGNMENT:")
        print("WELCOME")

        print("Please Select an option to run a Task:")
        print("1. Task A - Hope Speech Detection with Transformer models")
        print("2. Task B - Transfer learning & Analysis of Death Row Statements")
        print("3. Exit")

        # Get user input
        choice = input("Enter your choice (1-3): ")

        # Perform tasks based on user's choice
        if choice == '1': #TASK A

            print("0. Visualise Dataset")
            print("1. Train/Test Bert model from scratch")
            print("2. Train/Test Roberta model from scratch")
            print("3. Train/Test Bert model with balanced weights from scratch")
            print("4. Run pre-saved Bert model(note please run 1. first)")
            print("5. Run pre-saved Roberta model (please run 2. first)")
            print("6. Run pre-saved Bert balanced model (please run 3. first)")

            # Get user input
            choice2 = input("Enter your choice (0-6): ")

            if choice2 == '0':
                # Call the method 
                print("Visualisation of HopeEDI split and labels")
                hope_bert.data_visualisation()
                break

            elif choice2 == '1':
                # TASK A - Train/Test Bert model from scratch 
                hope_bert.run_fromscratch_no_weights()
                break

            elif choice2 == '2':
                # TASK A - Train/Test roberta model from scratch 
                hope_roberta.run_fromscratch_no_weights()
                break
            
            elif choice2 == '3':
                # TASK A - Train/Test Bert model with balanced weights from scratch
                hope_bert.run_fromscratch_balanced()
                break

            elif choice2 == '4':
                # Test Bert model pre-saved (note the model needs to be in the task A folder)
                model_path = 'A/bert_model_no_weights'
                hope_bert.run_fromsaved(model_path)
                break

            elif choice2 == '5':
                # Test roberta model pre-saved (note the model needs to be in the task A folder)
                model_path = 'A/roberta_model_no_weights'
                hope_roberta.run_fromsaved(model_path)
                break
        
            elif choice2 == '6':
                # Test bert model with weights pre-saved (note the model needs to be in the task A folder)
                model_path = 'A/bert_model_with_weights'
                hope_bert.run_fromsaved(model_path)
                break

        elif choice == '2': #TASK B

            print("0. Visualise Dataset")
            print("1. Re-collect dataset from Texas Department Death Row info page")
            print("2. Transfer Learning - Sentiment Analysis")
            print("3. Transfer Learning - HopeSpeech Detection (please run TASK A - 2. first)")

            # Get user input
            choice3 = input("Enter your choice (0-3): ")
   
            if choice3 == '0':
                # print the first few rows of the death row dataset
                df = pd.read_csv(deathrow_path, encoding="utf-8-sig")
                df.head()

                death_row_hope.data_visualisation()
                sa.plot_statement_length_distribution()
                break

            elif choice3 == '1':
                #this class and function collects the data from the Texas Department of Criminal Justice death row information page
                newdataset = data_acquisition_last_statements()
                break
                
            elif choice3 == '2':
                #Sentiment Analysis
                results = sa.perform_sentiment_analysis()
                sa.visualize_sentiment_results(results)

                break

            elif choice3 == '3':
                #Transfer learning
                death_row_hope.apply_transfered_model()
                break

        elif choice == '3':
            print("Thank you for grading my assignment! Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
