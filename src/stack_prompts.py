import pandas as pd
from datasets import load_dataset
import numpy as np

def create_stacked_prompt(train_prompts, good_requests, prompt_size, prompt_amount):
    stacked_prompts, labels = [], []
    for idx in range(prompt_amount):
        prompt = ""
        demonstrations = np.random.randint(45, 60)
        sampled_idx = np.random.choice(len(train_prompts), size=demonstrations, replace=False)
        for i in sampled_idx[:len(sampled_idx) - 2]:
            prompt += "USER: " + train_prompts.iloc[i]["prompt"] + "\nASSISTANT: " + train_prompts.iloc[i]["edited_unsafe_completion"] + "\n\n"
        if idx < prompt_amount // 2:
            prompt += "USER: " + good_requests.iloc[idx] + "\nASSISTANT: "
            label = "Safe"
        else:
            prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\nASSISTANT: "
            label = train_prompts.iloc[sampled_idx[-1]]["safety_category"]
        # if len(prompt) > prompt_size[1] or len(prompt) < prompt_size[0]:
        #     continue
        stacked_prompts.append(prompt)
        labels.append(label)
    return stacked_prompts, labels
        

def main():
    train_prompts = pd.read_csv("../data/single_train_prompts.csv")
    prompt_size = [8192, 20024]
    prompt_amount = 1000
    dataset = load_dataset("argilla/databricks-dolly-15k-curated-en")
    dataset = pd.DataFrame(dataset["train"])
    dataset["request"] = dataset["original-context"] + "\n" + dataset["original-instruction"]
    good_requests = dataset["request"].sample(prompt_amount // 2)
    stacked_prompts, labels = create_stacked_prompt(train_prompts, good_requests, prompt_size, prompt_amount)
    print(len(stacked_prompts))
    print(len(labels))
    stacked_df = pd.DataFrame(columns = ["prompt", "label"], index = range(len(stacked_prompts)))
    stacked_df["prompt"], stacked_df["label"] = stacked_prompts, labels
    stacked_df.to_csv("../data/stacked_prompts.csv", index=False)
if __name__ == "__main__":
    main()