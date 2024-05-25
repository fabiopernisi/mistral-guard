import pandas as pd
from datasets import load_dataset
import numpy as np


def create_stacked_prompt(train_prompts, good_requests, prompt_amount, test_requests = None):
    stacked_prompts, labels = [], []
    for idx in range(prompt_amount):
        prompt = ""
        demonstrations = np.random.randint(45, 60)
        sampled_idx = np.random.choice(len(train_prompts), size=demonstrations, replace=False)
        flag = 0
        p = np.random.random()
        if p > 0.5:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\nASSISTANT: "
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\nASSISTANT: "
                    label = train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx] + "\nASSISTANT: "
                    label = test_requests.iloc[idx]["safety_category"]
            flag = 1

        for i in sampled_idx[:len(sampled_idx) - 2]:
            prompt += "USER: " + train_prompts.iloc[i]["prompt"] + "\nASSISTANT: " + train_prompts.iloc[i]["edited_unsafe_completion"] + "\n\n"
        if flag == 0:
            if idx < prompt_amount // 2:
                prompt += "USER: " + good_requests.iloc[idx] + "\nASSISTANT: "
                label = "Safe"
            else:
                if test_requests is None:
                    prompt += "USER: " + train_prompts.iloc[sampled_idx[-1]]["prompt"] + "\nASSISTANT: "
                    label = train_prompts.iloc[sampled_idx[-1]]["safety_category"]
                else:
                    prompt += "USER: " + test_requests.iloc[idx] + "\nASSISTANT: "
                    label = test_requests.iloc[idx]["safety_category"]
        # if len(prompt) > prompt_size[1] or len(prompt) < prompt_size[0]:
        #     continue
        stacked_prompts.append(prompt)
        labels.append(label)
    return stacked_prompts, labels
        

def main():
    all_prompts = pd.read_csv("../data/single_train_prompts.csv")

    train_prompts = all_prompts[all_prompts['split'] == 'train']
    valid_prompts = all_prompts[all_prompts['split'] == 'validation']
    test_prompts = all_prompts[all_prompts['split'] == 'test']

    # prompt_size = [8192, 20024]
    prompt_amount = 1000
    dataset = load_dataset("argilla/databricks-dolly-15k-curated-en")
    dataset = pd.DataFrame(dataset["train"])
    dataset["request"] = dataset["original-context"] + "\n" + dataset["original-instruction"]
    all_good_requests = dataset["request"].sample(prompt_amount // 2)
    train_good_requests = all_good_requests[:400]
    valid_good_requests = all_good_requests[400:450]
    test_good_requests = all_good_requests[450:]

    train_stacked_prompts, train_labels = create_stacked_prompt(train_prompts, train_good_requests, 0.8 * prompt_amount)
    val_stacked_prompts, val_labels = create_stacked_prompt(train_prompts, valid_good_requests, 0.1 * prompt_amount)
    test_stacked_prompts, test_labels = create_stacked_prompt(pd.concat([train_prompts, valid_prompts]), test_good_requests, 0.1 * prompt_amount, test_requests = test_prompts)
    stacked_prompts = train_stacked_prompts + val_stacked_prompts + test_stacked_prompts
    labels = train_labels + val_labels + test_labels
    stacked_df = pd.DataFrame(columns = ["prompt", "label"], index = range(len(stacked_prompts)))
    stacked_df["prompt"], stacked_df["label"] = stacked_prompts, labels
    stacked_df.to_csv("../data/stacked_prompts.csv", index=False)
if __name__ == "__main__":
    main()