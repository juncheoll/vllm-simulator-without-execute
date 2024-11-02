import pandas as pd
from datasets import load_dataset
import json
from transformers import LlamaTokenizer
import ast

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token



df = pd.read_csv('squad_sorted.csv')

def tokenize_text_for_input(text):
    if pd.isna(text):
        return []
    
    conversation = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
    tokens = [tokens[0]] + tokens[4:-5]
    return tokens

def tokenize_text_for_output(text):
    if pd.isna(text):
        return []
    
    conversation = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
    tokens = tokens[4:-4] + [2]
    return tokens

def tokens_to_json(tokens):
    return json.dumps(tokens)


df['Input_Tokens'] = df['Input_Text'].apply(tokenize_text_for_input)
df['Output_Tokens'] = df['Output_Text'].apply(tokenize_text_for_output)

df['Input_Tokens'] = df['Input_Tokens'].apply(tokens_to_json)
df['Output_Tokens'] = df['Output_Tokens'].apply(tokens_to_json)

df = df[['Request Id', 'Input_Text', 'Output_Text', 'Input_Tokens', 'Output_Tokens', 'context_count']]

output_file = "squad_sorted-p1.csv"
df.to_csv(output_file, index=False)

print('성공')