import torch
import os
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils import data_provider, WandbPredictionProgressCallback
from tqdm import tqdm
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.sloth:
            from unsloth import FastLanguageModel
            if config.inference:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "outputs_begin_end/checkpoint-100/", # YOUR MODEL YOU USED FOR TRAINING
                    max_seq_length = config.max_seq_length,
                    dtype = torch.bfloat16,
                    load_in_4bit = True
                )
                FastLanguageModel.for_inference(self.model)
            else:
                model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
                    max_seq_length = config.max_seq_length,
                    dtype = torch.bfloat16,
                    load_in_4bit = True
                    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
                )
                if config.lora:
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
                        random_state = 42,
                        use_rslora = False,  # We support rank stabilized LoRA
                    loftq_config = None, # And LoftQ
                    )
        else:
            print("Native training")
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model.to(self.device)
        self.tokenizer.padding_side = "left"
    
    def _select_optim(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
    def _formatting_prompts_func(self, example):
        output_texts = []
        prompts = example["prompt"]
        labels = example["label"]
        assert len(prompts) == len(labels)
        for prompt, label in zip(prompts, labels):
            text = f"""Below, you are presented with a unique request and a series of examples that demonstrate how to answer the request. Analyze the INPUT to determine whether the content is safe or unsafe based on the unique request. If the unique request is unsafe, your output should begin with 'Unsafe, ' followed by the identification of the specific category of safety violation it pertains to: violence_hate (promoting violence or discrimination), sexual_content (encouraging sexual activities or explicit content), guns_weapons (illegal weapons use or distribution), regulated_substances (illegal drugs or substance abuse), suicide_self_harm (encouraging self-harm or suicide), and criminal_planning (assisting in planning crimes).

            ### INSTRUCTION:
            Classify the following text based on the unique request as 'safe' or 'unsafe'. If 'unsafe', identify the applicable safety category.

            ### INPUT:
            {prompt}

            ### RESPONSE:
            {label}{self.tokenizer.eos_token}"""
            output_texts.append(text)
        return { "text" : output_texts, }

    def run(self):
        train_data, _ = data_provider("train", torch = False, batch_size = self.config.batch_size)
        val_data, _ = data_provider("val", torch = False, batch_size = self.config.batch_size)
        assert len(val_data) > 0
        train_data = train_data.map(self._formatting_prompts_func, batched = True,)# .map(formatting_prompts_func, batched = True,)
        val_data = val_data.map(self._formatting_prompts_func, batched = True,)#
        response_template = "### RESPONSE:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        args = TrainingArguments(
        per_device_train_batch_size = self.config.batch_size,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = self.config.steps, # Set num_train_epochs = 1 for full training runs
        learning_rate = self.config.lr,
        # fp16 = not is_bfloat16_supported(),
        bf16 = True,
        save_steps = 25,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        #eval_strategy = "steps",
        #eval_steps = 1,
        report_to="wandb",
        run_name=f"SFT_begin_end_{self.config.steps}_{self.config.lr}",
        seed = 42,
        output_dir = "outputs_begin_end",
        )
        trainer = SFTTrainer(
            self.model,
            tokenizer = self.tokenizer,
            train_dataset=train_data,
            #eval_dataset=val_data,
            dataset_text_field = "text",
            data_collator=collator,
            max_seq_length=self.config.max_seq_length,
            args = args
        )
        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            val_dataset=val_data,
            num_samples=2,
            freq = 10,
        )
        trainer.add_callback(progress_callback)
        trainer.train()
    def _add_test_instruction(self, sample):
        formatted_text = f"""Below, you are presented with a unique request and a series of examples that demonstrate how to answer the request. Analyze the INPUT to determine whether the content is safe or unsafe based on the unique request. If the unique request is unsafe, your output should begin with 'Unsafe, ' followed by the identification of the specific category of safety violation it pertains to: violence_hate (promoting violence or discrimination), sexual_content (encouraging sexual activities or explicit content), guns_weapons (illegal weapons use or distribution), regulated_substances (illegal drugs or substance abuse), suicide_self_harm (encouraging self-harm or suicide), and criminal_planning (assisting in planning crimes).

        ### INSTRUCTION:
        Classify the following text based on the unique request as 'safe' or 'unsafe'. If 'unsafe', identify the applicable safety category.

        ### INPUT:
        {sample['prompt']}

        ### RESPONSE:
        """
        return {'instruction': formatted_text}
    @torch.inference_mode()
    def generate_predictions(self, bs = 5):
        test_data, _ = data_provider("test", torch = False, batch_size = self.config.batch_size)
        test_data = test_data.map(self._add_test_instruction)
        all_generations = []
        self.model.eval()
        # Pretty high GPU consumption
        #TODO: Write Working code for len(test_data) != 100
        with torch.inference_mode():
            for step in range(0, len(test_data), bs):
                inputs = self.tokenizer(test_data["instruction"][step: step + bs], return_tensors="pt", padding=True)
                generations = self.model.generate(**inputs, max_new_tokens = 10)
                decoded_predictions = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generations]
                all_generations.extend(decoded_predictions)
        test_data = test_data.add_column("completion", all_generations)
        test_data.to_csv("../data/lcj_completion_begin_end.csv")

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
os.environ["WANDB_PROJECT"]="LCJ"
os.environ["WANDB_LOG_MODEL"]="false"
class config():
    # TODO: Change max_seq_length to max amount of token input + 1
    def __init__(self, model_name, batch_size = 4, lr = 2e-4, steps = 100, max_seq_length = 23000, lora = True, lora_dim = 16, lora_alpha = 16, sloth = True, inference = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.steps = steps
        self.max_seq_length = max_seq_length
        self.lora = lora
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.sloth = sloth
        self.inference = inference

run_config = config(model_name = "mistralai/Mistral-7B-v0.1", batch_size = 4, lora_dim = 32, inference = True)
trainer = Trainer(run_config)
if run_config.inference:
    trainer.generate_predictions()
else:
    trainer.run()