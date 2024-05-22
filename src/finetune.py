import torch
import os
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils import data_provider
from tqdm import tqdm
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.sloth:
            from unsloth import FastLanguageModel
            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/mistral-7b-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
                max_seq_length = config.max_seq_length,
                dtype = torch.bfloat16,
                load_in_4bit = True,
                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # Let's first try LorA  
        if config.lora & config.sloth:
            self.model = FastLanguageModel.get_peft_model(
                model,
                r = config.lora_dim, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = config.lora_alpha,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
        self.model.to(self.device)
    
    def _select_optim(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
    def _formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"""<s>### You are an helpful AI Model. Classify the question into one of these categories. 
            Question: {example['prompt'][i]}\n ### Answer: {example['label'][i]} + {self.tokenizer.eos_token}"""
            output_texts.append(text)
        return output_texts

    def run(self):
        train_data, _ = data_provider("train", self.config.batch_size)
        val_data, _ = data_provider("validation", self.config.batch_size)
        response_template = " ### Answer:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir="first_model_checkpoint/",
            # num_train_epochs=3,
            # max_steps = 100,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            evaluation_strategy="epoch",
            # evaluation_strategy="steps",
            # eval_steps=20,
            optim="adamw_8bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            # tf32=True,
            # max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=False
        )
        trainer = SFTTrainer(
            self.model,
            train_dataset=train_data,
            val_dataset=val_data,
            formatting_func=self._formatting_prompts_func,
            data_collator=collator,
            max_seq_length=self.config.max_seq_length,
            args = training_args
        )
        trainer.train()

    # Good prompts and completions: https://huggingface.co/datasets/yahma/alpaca-cleaned
    # https://exnrt.com/blog/ai/mistral-7b-fine-tuning/
    # https://medium.com/@sujathamudadla1213/difference-between-trainer-class-and-sfttrainer-supervised-fine-tuning-trainer-in-hugging-face-d295344d73f7#:~:text=Use%20Trainer%3A%20If%20you%20have,experience%20with%20efficient%20memory%20usage.
    # def train(self):
    #     train_loader = data_provider("train", self.config.batch_size)
    #     optimizer = self._select_optim()
    #     # scheduler
    #     train_steps = len(train_loader)
    #     total_steps = self.config.epochs * train_steps
    #     train_loss = []
    #     self.model.train()
    #     for epoch in range(self.config.epochs):
    #         epoch_loss = 0
    #         iter = 0
    #         for i, (X, y) in enumerate(train_loader):
    #             iter += 1
    #             X, y = self.tokenizer.tokenize(X, padding = True, return_tensors = "pt").to(self.device), self.tokenizer.tokenize(y, padding = True, return_tensors = "pt").to(self.device)
    #             optimizer.zero_grad()
    #             with torch.cuda.amp.autocast():
    #                 outputs = self.model(X, labels = y)

def config():
    def __init__(self, model_name, batch_size = 4, lr = 2e-4, epochs = 3, max_seq_length = 32678, lora = True, lora_dim = 16, lora_alpha = 16, sloth = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.lora = lora
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.sloth = sloth

config = config(model_name = "mistralai/Mistral-7B-v0.1", batch_size = 2)
trainer = Trainer(config)
trainer.run()