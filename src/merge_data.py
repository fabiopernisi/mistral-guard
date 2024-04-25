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

def load_csv_prompts(file_path):
    df = pd.read_csv(file_path, usecols=['Behavior', 'Tags'])
    df.rename(columns={'Behavior': 'prompt', 'Tags': 'label'}, inplace=True)
    return df

def main():
    json_files = [
        "../data/I-CoNa.json",
        "../data/I-MaliciousInstructions.json",
        "../data/I-PhysicalSafetyUnsafe.json"
    ]
    csv_file = "../data/harmbench_behaviors_text_all.csv"
    
    frames = [load_json_prompts(f) for f in json_files]
    
    csv_data = load_csv_prompts(csv_file)
    
    # Append all data frames
    final_df = pd.concat(frames + [csv_data], ignore_index=True)
    
    # Save to /data folder
    final_df.to_csv("../data/combined_unsafe_prompts.csv", index=False)

if __name__ == "__main__":
    main()
