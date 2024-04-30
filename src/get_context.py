import pandas as pd

def get_context(source, target):

    context_prompts = []

    source_df = pd.read_csv(source)
    target_df = pd.read_csv(target, usecols=['Behavior', 'Tags', 'ContextString'])

    # print(target_df[target_df['Behavior'].duplicated(keep=False)])

    for prompt in source_df['prompt'].unique():
        for index, row in target_df.iterrows():
            if prompt == row['Behavior']: 
                context_prompts.append(row['Behavior'] + "\n\n" + row['ContextString'])

    return context_prompts

def main():
    source = "../data/context/DATASET_context.csv"
    target = "../data/harmbench_behaviors_text_all.csv"
    context_prompts = get_context(source, target)

    final_df = pd.DataFrame({'context_prompts': context_prompts})
    final_df.to_csv("../data/context/context_prompts.csv", index= False)

if __name__ == "__main__":
    main()
    