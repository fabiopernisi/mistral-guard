import pandas as pd
from datasets import load_dataset
import numpy as np

np.random.seed(42)

def create_stacked_prompt(train_prompts, good_requests, prompt_amount, test_requests = None):
    stacked_prompts, labels = [], []
    for idx in range(prompt_amount):
        prompt = ""
        demonstrations = np.random.randint(2, 15)
        sampled_idx = np.random.choice(len(train_prompts), size=demonstrations, replace=False)
        flag = 0
        p = np.random.random()
        if p > 0.5:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\n\n\n"
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\n\n\n "
                    label = "Unsafe, " + train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx]['prompt'] + "\n\n\n"
                    label = "Unsafe, " + test_requests.iloc[idx]["safety_category"]
            flag = 1

        for i in sampled_idx[:len(sampled_idx) - 2]:
            prompt += "USER: " + train_prompts.iloc[i]["prompt"] + "\nASSISTANT: " + train_prompts.iloc[i]["edited_unsafe_completion"] + "\n\n"
        if flag == 0:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\n"
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\n"
                    label = "Unsafe, " + train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx]['prompt'] + "\n"
                    label = "Unsafe, " + test_requests.iloc[idx]["safety_category"]
        # if len(prompt) > prompt_size[1] or len(prompt) < prompt_size[0]:
        #     continue
        stacked_prompts.append(prompt)
        labels.append(label)
    return stacked_prompts, labels

def create_mixed_stacked_prompt(train_prompts, good_requests, prompt_amount, safe_percentage, alpaca_instructions, alpaca_output, test_requests = None):
    stacked_prompts, labels = [], []
    for idx in range(prompt_amount):
        prompt = ""
        # demonstrations = np.random.randint(2, 8)
        demonstrations = 1
        sampled_idx = np.random.choice(len(train_prompts), size=demonstrations, replace=False)
        flag = 0
        p = np.random.random()
        if p > 0.5:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\n\n\n"
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\n\n\n "
                    label = "Unsafe, " + train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx]['prompt'] + "\n\n\n"
                    label = "Unsafe, " + test_requests.iloc[idx]["safety_category"]
            flag = 1

        for i in sampled_idx[:len(sampled_idx) - 2]:
            q = np.random.random()
            if q >= safe_percentage:
                prompt += "USER: " + train_prompts.iloc[i]["prompt"] + "\nASSISTANT: " + train_prompts.iloc[i]["edited_unsafe_completion"] + "\n\n"
            else:
                prompt += "USER: " + alpaca_instructions.iloc[i] + "\nASSISTANT: " + alpaca_output.iloc[i] + "\n\n"
        if flag == 0:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\n"
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\n"
                    label = "Unsafe, " + train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx]['prompt'] + "\n"
                    label = "Unsafe, " + test_requests.iloc[idx]["safety_category"]
        # if len(prompt) > prompt_size[1] or len(prompt) < prompt_size[0]:
        #     continue
        stacked_prompts.append(prompt)
        labels.append(label)
    return stacked_prompts, labels
        

def main():
    all_prompts = pd.read_csv("../data/single_train_prompts.csv")
    prompt_amount = 1000

    train_prompts = all_prompts[all_prompts['split'] == 'train']
    valid_prompts = all_prompts[all_prompts['split'] == 'validation']
    test_prompts = all_prompts[all_prompts['split'] == 'test']

    alpaca = load_dataset("yahma/alpaca-cleaned")
    alpaca = pd.DataFrame(alpaca["train"])
    alpaca = alpaca[alpaca['input'] == ''].sample(len(all_prompts))
    alpaca_instructions = alpaca["instruction"]
    alpaca_output = alpaca["output"]

    # prompt_size = [8192, 20024]
    dataset = load_dataset("argilla/databricks-dolly-15k-curated-en")
    dataset = pd.DataFrame(dataset["train"])
    dataset["request"] = dataset["original-context"] + "\n" + dataset["original-instruction"]
    all_good_requests = dataset["request"].sample(prompt_amount // 2)
    train_alpaca_instructions, train_alpaca_output = alpaca_instructions[:len(train_prompts) + len(valid_prompts)], alpaca_output[:len(train_prompts) + len(valid_prompts)]
    test_alpaca_instructions, test_alpaca_output = alpaca_instructions[900:], alpaca_output[900:]
    train_good_requests = all_good_requests[:400]
    valid_good_requests = all_good_requests[400:450]
    test_good_requests = all_good_requests[450:]

    safe_p = 0.0
    train_stacked_prompts, train_labels = create_mixed_stacked_prompt(train_prompts, train_good_requests, int(0.8 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output)
    val_stacked_prompts, val_labels = create_mixed_stacked_prompt(train_prompts, valid_good_requests, int(0.1 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output)
    test_stacked_prompts, test_labels = create_mixed_stacked_prompt(pd.concat([train_prompts, valid_prompts]), test_good_requests, int(0.1 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output, test_requests = test_prompts)
    stacked_prompts = train_stacked_prompts + val_stacked_prompts + test_stacked_prompts
    labels = train_labels + val_labels + test_labels
    stacked = pd.DataFrame(columns = ["prompt", "label"], index = range(len(stacked_prompts)))
    stacked['split'] = 'train'
    stacked.loc[stacked.index > stacked.index.max()*0.8, 'split'] = 'validation'
    stacked.loc[stacked.index > stacked.index.max()*0.9, 'split'] = 'test'
    stacked["prompt"], stacked["label"] = stacked_prompts, labels
    stacked = stacked.sample(frac=1).reset_index(drop=True)
    stacked.to_csv(f"../data/stacked_prompts_split_no_context.csv", index=False)
if __name__ == "__main__":
    main()