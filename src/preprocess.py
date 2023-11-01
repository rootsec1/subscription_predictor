# Import the pandas library
import pandas as pd

# Create a class for data preprocessing
class PreProcess():
    dataset_root_dir: str = None
    __accounts_df__: pd.DataFrame = pd.DataFrame()
    __subscriptions_df__: pd.DataFrame = pd.DataFrame()
    __training_data_df__: pd.DataFrame = pd.DataFrame()

    def __init__(self, dataset_root_dir: str):
        # Initialize the class with a dataset root directory
        self.dataset_root_dir = dataset_root_dir
        # Load the account, subscriptions, and training data into dataframes
        self.__accounts_df__ = self.__load_data__("account.csv")
        self.__subscriptions_df__ = self.__load_data__("subscriptions.csv")
        self.__training_data_df__ = self.__load_data__("train.csv")
        
    def __load_data__(self, filename: str):
        try:
            # Load data from a CSV file with UTF-8 encoding
            return pd.read_csv(f"{self.dataset_root_dir}/{filename}")
        except UnicodeDecodeError:
            # If there is an encoding error, try loading with Latin-1 encoding
            return pd.read_csv(f"{self.dataset_root_dir}/{filename}", encoding="latin-1")
    
    def clean_data(self):
        # Clean and preprocess the data
        self.__clean_account_data__()
        self.__clean_subscription_data__()
        merged_df = self.__merge_accounts_with_subscriptions__()
        merged_df = self.__merge_training_df_into_final_df__(merged_df)
        return merged_df
    
    def __clean_account_data__(self):
        # Remove unnecessary columns and fill NaN values with 0
        self.__accounts_df__.drop(columns=["shipping.zip.code", "shipping.city", "relationship", "first.donated"], inplace=True)
        self.__accounts_df__.fillna(0, inplace=True)

    def __clean_subscription_data__(self):
        # Select specific columns and calculate the number of subscriptions per account
        self.__subscriptions_df__ = self.__subscriptions_df__[["account.id", "package", "price.level", "subscription_tier"]]
        self.__subscriptions_df__["num_subscriptions"] = self.__subscriptions_df__.groupby("account.id")["account.id"].transform("count")

    def __merge_accounts_with_subscriptions__(self):
        # Merge account and subscription data, remove duplicates, and fill NaN values with 0
        merged_df = pd.merge(self.__accounts_df__, self.__subscriptions_df__, on="account.id", how="left")
        merged_df = merged_df.drop_duplicates(subset=['account.id'])
        merged_df.fillna(0, inplace=True)
        return merged_df
        
    def __merge_training_df_into_final_df__(self, merged_df: pd.DataFrame):
        # Merge the training data into the final merged dataframe
        final_train_df = self.__training_data_df__.merge(merged_df, on='account.id', how='left')
        return final_train_df
