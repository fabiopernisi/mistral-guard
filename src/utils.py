import pandas as pd
import os

def save_to_csv(df: pd.DataFrame, file_name: str, file_path: str = "../data"):
    """
    Save DataFrame to CSV file.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    df.to_csv(f"{file_path}/{file_name}.csv", index=False)

def get_first_prompt(df: pd.DataFrame, keyword: str = "Assistant"):
    """
    Cut any full prompt up to the first occurence of the keyword for all dataframe columns.
    Example:  
    \n\nHuman: What kind of noises did dinosaurs make?\n\nAssistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say.
    -> \n\nHuman: What kind of noises did dinosaurs make?\n\n
    """
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[:x.find(keyword)] if keyword in x else x)
    return df