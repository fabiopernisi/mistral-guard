from datasets import load_dataset
import pandas as pd
from utils import save_to_csv, get_first_prompt, get_batch_completion
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

#hh_dataset = load_dataset("Anthropic/hh-rlhf")

# should process hh-rlhf dataset and create a final csv with the dataset, and save it into \data

# Transform dataset to pandas DataFrame
#hh_train = pd.DataFrame(hh_dataset["train"]) # two columns: "chosen" and "rejected"

# Simple first implementation of process_data that just gets the first human prompt. Note that doing this will result in "chosen" and "rejected" columns
# containing exactly the same data (the first human prompt)
def process_data(df: pd.DataFrame):
    # df = get_first_prompt(df, "Assistant")
    tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/WizardLM-33B-V1.0-Uncensored")
    tokenizer.padding_side = "left"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/WizardLM-33B-V1.0-Uncensored", device_map = "auto", load_in_4bit = True, torch_dtype = torch.bfloat16)
    print("Finished loading model.")
    df["model_completion"] = "NaN"
    step_size = 4
    for index in tqdm(range(0, len(df), step_size), desc = "Generating completions"):
        end_index = min(index + step_size, len(df))
        batch_prompts = df["prompt"][index:end_index].tolist()
        batch_completions = get_batch_completion(model, tokenizer, batch_prompts, batch_size = step_size)
        df.loc[index:end_index-1, "model_completion"] = batch_completions
        save_to_csv(df, "merged_unsafe_prompts_ongoing")
    return df

merged_unsafe_prompts = pd.read_csv("../data/merged_unsafe_prompts.csv")
merged_unsafe_prompts_processed = process_data(merged_unsafe_prompts)
save_to_csv(merged_unsafe_prompts_processed, "merged_unsafe_prompts_completion")