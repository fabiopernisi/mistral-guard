from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import pandas as pd
import numpy as np
from typing import List, Dict

login()

model_id = "meta-llama/LlamaGuard-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

dataset = pd.read_csv('../data/lcj_completion_begin_end_125_safe_demos_00.csv') # sample dataset

def build_chats(dataset: pd.DataFrame) -> List[List[Dict]]:
    """
    Bulid chats for Llama Guard evaluation
    """
    chats = []
    instructions = dataset['instruction'].tolist()
    for instruction in instructions:
        chat = [{"role": "user", "content": instruction}]
        chats.append(chat)
    return chats

chats = build_chats(dataset)

def moderate_with_template(chat: List[Dict]):
    """
    Function to evaluate single chat using Llama Guard
    """
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def evaluate_chats(chats: List[List[Dict]]):
    """
    Function to evaluate multiple chats using Llama Guard
    """
    responses = []
    for chat in chats:
        response = moderate_with_template(chat)
        responses.append(response)
    return responses