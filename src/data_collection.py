import pandas as pd
from utils import save_to_csv, post_process_model_completion, get_batch_completion
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

def process_data(df: pd.DataFrame):
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
        save_to_csv(df, "../data/new_prompts/all_additional_prompts_ongoing")
    return df

merged_unsafe_prompts = pd.read_csv("../data/new_prompts/all_additional_prompts")
merged_unsafe_prompts_processed = process_data(merged_unsafe_prompts)
dataset_postprocessed = post_process_model_completion(merged_unsafe_prompts_processed, "model_completion")
save_to_csv(dataset_postprocessed, "../data/new_prompts/all_additional_prompts_completion")