from datasets import load_dataset
import pandas as pd
from utils import save_to_csv, get_first_prompt, get_batch_completion
from transformers import AutoTokenizer, AutoModelForCausalLM

hh_dataset = load_dataset("Anthropic/hh-rlhf")

# should process hh-rlhf dataset and create a final csv with the dataset, and save it into \data

# Transform dataset to pandas DataFrame
hh_train = pd.DataFrame(hh_dataset["train"]) # two columns: "chosen" and "rejected"

# Simple first implementation of process_data that just gets the first human prompt. Note that doing this will result in "chosen" and "rejected" columns
# containing exactly the same data (the first human prompt)
def process_data(df: pd.DataFrame):
    df = get_first_prompt(df, "Assistant")
    tokenizer = AutoTokenizer.from_pretrained("alpindale/WizardLM-2-8x22B")
    model = AutoModelForCausalLM.from_pretrained("alpindale/WizardLM-2-8x22B")
    df["generated"] = get_batch_completion(model, tokenizer, df["rejected"].tolist())
    return df

hh_train_processed = process_data(hh_train)
save_to_csv(hh_train_processed, "hh_train_processsed")