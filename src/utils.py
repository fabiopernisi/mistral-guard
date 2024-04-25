import pandas as pd
import os
from typing import List
from tqdm import tqdm

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

def get_batch_completion(model, tokenizer, texts: List[str], batch_size = 4):
    """
    Load corresponding model from Hugging Face and generate completions for given text.
    """
    generated = list()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation = True, padding=True, return_tensors="pt").to("cuda")
        # TODO: Talk about parameters, i.e. top_k, max_length, etc.
        outputs = model.generate(**inputs, max_length = 1024)
        gen = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        print(gen)
        generated.extend(gen)
    return generated