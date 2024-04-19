from datasets import load_dataset

hh_dataset = load_dataset("Anthropic/hh-rlhf")

# should process hh-rlhf dataset and create a final csv with the dataset, and save it into \data