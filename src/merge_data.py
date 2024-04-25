import json
import pandas as pd

def load_json_prompts(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        df = pd.DataFrame(data['instructions'], columns=['prompt'])
        if 'tags' in data:
            df['label'] = data['tags']
        else:
            df['label'] = pd.NA
    return df

def load_csv_prompts(file_path, has_labels=True):
    if has_labels:
        df = pd.read_csv(file_path, usecols=['Behavior', 'Tags'])
        df.rename(columns={'Behavior': 'prompt', 'Tags': 'label'}, inplace=True)
    else:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['prompt']
        df['label'] = pd.NA  # assign empty column for labels
    return df

def main():
    json_files = [
        "../data/I-CoNa.json",
        "../data/I-MaliciousInstructions.json",
        "../data/I-PhysicalSafetyUnsafe.json"
    ]
    csv_files = [
        ("../data/harmbench_behaviors_text_all.csv", True),  # CSV with labels
        ("../data/transfer_expriment_behaviors.csv", False)  # CSV without labels
    ]
    
    frames = [load_json_prompts(f) for f in json_files]
    
    for file_path, has_labels in csv_files:
        frames.append(load_csv_prompts(file_path, has_labels))
    
    final_df = pd.concat(frames, ignore_index=True)
    
    final_df.to_csv("../data/merged_unsafe_prompts.csv", index=False)

if __name__ == "__main__":
    main()
