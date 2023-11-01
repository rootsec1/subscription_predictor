# Import the Model and PreProcess classes from model.py and preprocess.py
from model import Model
from preprocess import PreProcess

def main():
    # Create an instance of PreProcess with the specified dataset directory
    preprocess_instance = PreProcess(dataset_root_dir="../data/features")
    
    # Clean the data and obtain the training dataframe
    training_df = preprocess_instance.clean_data()

    # Create an instance of the Model with the training dataframe
    model_instance = Model(training_df=training_df)
    
    # Train the model and obtain accuracy and ROC AUC
    acc, roc = model_instance.predict_and_return_accuracy_and_roc()
    
    # Print the obtained accuracy and ROC AUC
    print("Accuracy: ", acc)
    print("Area under ROC curve: ", roc)

if __name__ == '__main__':
    # Execute the main function if this script is run as the main program
    main()
