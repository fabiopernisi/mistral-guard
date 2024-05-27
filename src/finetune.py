import torch
import os
from transformers import TrainingArguments
from utils import data_provider, WandbPredictionProgressCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

class Model():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.sloth:
            from unsloth import FastLanguageModel
            if config.inference:
                # Load the model from checkpoint for inference
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "outputs_begin_end/checkpoint-125/", # Trained Model
                    max_seq_length = config.max_seq_length,
                    dtype = torch.bfloat16,
                    load_in_4bit = True
                )
                FastLanguageModel.for_inference(self.model)
            else:
                # Load the model for traininig
                model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit",
                    max_seq_length = config.max_seq_length,
                    dtype = torch.bfloat16,
                    load_in_4bit = True
                )
                if config.lora:
                    self.model = FastLanguageModel.get_peft_model(
                        model,
                        r = config.lora_dim,
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj",],
                        lora_alpha = config.lora_alpha,
                        lora_dropout = 0,
                        bias = "none",
                        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                        use_gradient_checkpointing = "unsloth",
                        random_state = 42,
                    )
        self.tokenizer.padding_side = "left"
    
    def _formatting_prompts_func(self, example):
        "Apply the template to the stacked prompts, characterized a specific instruction and format. This is for training only, since we also specify the correct label."
        output_texts = []
        prompts = example["prompt"]
        labels = example["label"]
        assert len(prompts) == len(labels)
        for prompt, label in zip(prompts, labels):
            text = f"""Below, you are presented with a unique request and a series of examples that demonstrate how to answer the request. Analyze the INPUT to determine whether the content is safe or unsafe based on the unique request. If the unique request is unsafe, your output should begin with 'Unsafe, ' followed by the identification of the specific category of safety violation it pertains to: violence_hate (promoting violence or discrimination), sexual_content (encouraging sexual activities or explicit content), guns_weapons (illegal weapons use or distribution), regulated_substances (illegal drugs or substance abuse), suicide_self_harm (encouraging self-harm or suicide), and criminal_planning (assisting in planning crimes).

            ### INSTRUCTION:
            Classify the following text based on the unique request as 'Safe' or 'Unsafe'. If 'unsafe', identify the applicable safety category.

            ### INPUT:
            {prompt}

            ### RESPONSE:
            {label}{self.tokenizer.eos_token}"""
            output_texts.append(text)
        return { "text" : output_texts, }

    def run(self):
        """
        Train the model using HF SFTTrainer
        """
        # Load train and validation data
        train_data, _ = data_provider("train")
        val_data, _ = data_provider("validation")
        assert len(val_data) > 0

        # Apply the template
        train_data = train_data.map(self._formatting_prompts_func, batched = True,)
        val_data = val_data.map(self._formatting_prompts_func, batched = True,)

        # Template to only consider loss on completions, NOT on input
        response_template = "### RESPONSE:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        # Declare the training arguments, if no wandb comment out report_to and run_name
        args = TrainingArguments(
        per_device_train_batch_size = self.config.batch_size,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = self.config.steps,
        learning_rate = self.config.lr,
        bf16 = True,
        save_steps = 25,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        report_to="wandb",
        run_name=f"SFT_begin_end_{self.config.steps}_{self.config.lr}",
        seed = 42,
        output_dir = "outputs_begin_end",
        )
        trainer = SFTTrainer(
            self.model,
            tokenizer = self.tokenizer,
            train_dataset=train_data,
            dataset_text_field = "text",
            data_collator=collator,
            max_seq_length=self.config.max_seq_length,
            args = args
        )
        # Log output on random validation set examples, uncomment if no wandb
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
        Classify the following text based on the unique request as 'Safe' or 'Unsafe'. If 'Unsafe', identify the applicable safety category.

        ### INPUT:
        {sample['prompt']}

        ### RESPONSE:
        """
        return {'instruction': formatted_text}
    
    def generate_predictions(self, bs = 5, path = "../data/stacked_prompts_split_no_context.csv"):
        """
        Generates output given test_data with a fixed batch-size "bs". Output is constrained to be a maximum of 10 tokens.
        """
        test_data, _ = data_provider("test", torch = False, batch_size = self.config.batch_size, path = path)
        # Add template, important: this does not add the label like the template for the train set does
        test_data = test_data.map(self._add_test_instruction)
        all_generations = []
        self.model.eval()
        #NOTE: Code assumes test data has length 100
        with torch.inference_mode():
            for step in range(0, len(test_data), bs):
                inputs = self.tokenizer(test_data["instruction"][step: step + bs], return_tensors="pt", padding=True)
                generations = self.model.generate(**inputs, max_new_tokens = 10)
                decoded_predictions = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generations]
                all_generations.extend(decoded_predictions)
        test_data = test_data.add_column("completion", all_generations)
        test_data.to_csv(path, index = False)

os.environ["WANDB_PROJECT"]="LCJ"
os.environ["WANDB_LOG_MODEL"]="false"

class config():
    def __init__(self, batch_size = 4, lr = 2e-4, steps = 150, max_seq_length = 23000, lora = True, lora_dim = 16, lora_alpha = 16, sloth = True, inference = False):
        """
        Configurations for the model for training/inference
        Parameters:
        batch_size: int, batch size for training
        lr: float, learning rate for training
        steps: int, number of steps to train
        max_seq_length: int, maximum sequence length for the model
        lora: bool, whether to use LoRA
        lora_dim: int, LoRA dimension
        lora_alpha: int, LoRA alpha
        sloth: bool, whether to use unsloth
        inference: bool, whether to run inference
        """
        self.batch_size = batch_size
        self.lr = lr
        self.steps = steps
        self.max_seq_length = max_seq_length
        self.lora = lora
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.sloth = sloth
        self.inference = inference

run_config = config(batch_size = 4, lora_dim = 32, inference = True)
model = Model(run_config)
if run_config.inference:
    model.generate_predictions()
else:
    model.run()