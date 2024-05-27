import pandas as pd
from datasets import load_dataset
import numpy as np

np.random.seed(42)

def create_mixed_stacked_prompt(train_prompts, good_requests, prompt_amount, safe_percentage, alpaca_instructions, alpaca_output, test_requests = None):
    """
    This function is used to create a long context prompt consisting of demonstrations and a USER request.
    Parameters:
    train_prompts: DataFrame, the training prompts
    good_requests: DataFrame, the safe requests
    prompt_amount: int, the number of prompts to generate
    safe_percentage: float, the percentage of safe demonstrations
    alpaca_instructions: DataFrame, the alpaca instructions
    alpaca_output: DataFrame, the alpaca outputs
    test_requests: DataFrame, the test requests
    Returns:
    stacked_prompts: list, the generated prompts

    More explanation:
    For the desired number of prompts, we generate a prompt with demonstrations and a unique request in the following way:
    We start by setting the range of demonstrations depending on desidered context length, i.e. between 45 and 60 demonstrations for long context and between 2 and 8 for short context.
    We then sample uniformly the number of demonstrations from the given range and then sample that many indices over the range of the training prompts.

    Then we sample "p", between 0 and 1. If p > 0.5, we place the unique request in the beginning of the prompt, otherwise at the end.
    We also fix that if the index is less than half of the prompt amount, we place a safe request, otherwise we place an unsafe request.
    All of this means that we will have 50 safe and 50 unsafe requests, and in expectation 25 of each should be at the top and bottom of the prompt.
    For the train set, the last sampled index corresponds to the request of which we record the safety category as the label for the whole prompt, given that we place an unsafe request.#

    Regarding demonstrations, "safe_p" dictates the percentage of safe demonstrations. We sample "q" between 0 and 1, and if q >= safe_p, we place a safe demonstration, otherwise we place an unsafe demonstration.
    The demonstrations are built by preprending "USER" to the instruction and "ASSISTANT" to the output.

    For the test set, the demonstrations will be sampled from the training and validation prompts, as the demonstrations should not
    impact the label of the prompt according to our definiton. The request will be sampled from the test requests, meaning that the model
    has never seen the request before. This is to ensure that the model has never seen the requests in the test set before.
    
    """
    stacked_prompts, labels = [], []
    for idx in range(prompt_amount):
        prompt = ""
        # demonstrations = np.random.randint(45, 60) # long context
        # demonstrations = np.random.randint(2, 8) # short context
        demonstrations = 1 # no context
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
        stacked_prompts.append(prompt)
        labels.append(label)
    return stacked_prompts, labels
        
def main():
    all_prompts = pd.read_csv("../data/single_train_prompts.csv")
    prompt_amount = 1000

    # Splits of single prompt completion demonstrations
    train_prompts = all_prompts[all_prompts['split'] == 'train']
    valid_prompts = all_prompts[all_prompts['split'] == 'validation']
    test_prompts = all_prompts[all_prompts['split'] == 'test']

    # Load dataset for safe question-answer demonstrations
    alpaca = load_dataset("yahma/alpaca-cleaned")
    alpaca = pd.DataFrame(alpaca["train"])
    alpaca = alpaca[alpaca['input'] == ''].sample(len(all_prompts))
    alpaca_instructions = alpaca["instruction"]
    alpaca_output = alpaca["output"]

    # Load dataset for safe requests
    dataset = load_dataset("argilla/databricks-dolly-15k-curated-en")
    dataset = pd.DataFrame(dataset["train"])
    dataset["request"] = dataset["original-context"] + "\n" + dataset["original-instruction"]
    all_good_requests = dataset["request"].sample(prompt_amount // 2)
    train_good_requests = all_good_requests[:400]
    valid_good_requests = all_good_requests[400:450]
    test_good_requests = all_good_requests[450:]

    # Set the safe percentage
    safe_p = 0.0

    train_stacked_prompts, train_labels = create_mixed_stacked_prompt(train_prompts, train_good_requests, int(0.8 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output)
    # val is not used anywhere; just for training validation so no need to pass test_requests
    val_stacked_prompts, val_labels = create_mixed_stacked_prompt(train_prompts, valid_good_requests, int(0.1 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output)
    test_stacked_prompts, test_labels = create_mixed_stacked_prompt(pd.concat([train_prompts, valid_prompts]), test_good_requests, int(0.1 * prompt_amount),
                            safe_percentage = safe_p, alpaca_instructions = alpaca_instructions, alpaca_output = alpaca_output, test_requests = test_prompts)
    
    # Build dataset from list
    stacked_prompts = train_stacked_prompts + val_stacked_prompts + test_stacked_prompts
    labels = train_labels + val_labels + test_labels
    stacked = pd.DataFrame(columns = ["prompt", "label"], index = range(len(stacked_prompts)))
    # [80, 10, 10] train test split
    stacked['split'] = 'train'
    stacked.loc[stacked.index > stacked.index.max()*0.8, 'split'] = 'validation'
    stacked.loc[stacked.index > stacked.index.max()*0.9, 'split'] = 'test'
    stacked["prompt"], stacked["label"] = stacked_prompts, labels

    # stacked = stacked.sample(frac=1).reset_index(drop=True) # Sampling is done by huggingface
    stacked.to_csv(f"../data/stacked_prompts_split_no_context.csv", index=False)
if __name__ == "__main__":
    main()