import pandas as pd
from datasets import load_dataset
import json
from transformers import LlamaTokenizer
import ast

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def literal_eval(text):
    return ast.literal_eval(text)['conversations']

def conversations(a):
    return a['conversations']

def distribute_conversations(row):
    rows = []
    
    for j in range(1, len(row['train']), 2):
        sub_train = row['train'][j-1:j+1]
        sequence = j // 2 + 1
        
        rows.append({
            'id': row['id'],
            'sequence': sequence,
            'Input_Text': sub_train[0]['content'].replace('\n', ''),
            'Output_Text': sub_train[1]['content'].replace('\n', '')
        })
    return pd.DataFrame(rows)


df = pd.read_csv('topical-chat-json.csv')
df['train'] = df['train'].apply(literal_eval)
split_df = df.apply(distribute_conversations, axis=1)
df = pd.concat(split_df.values)



def tokenize_text_for_input(text):
    if pd.isna(text):
        return []
    
    conversation = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
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

df['Request Id'] = df['id'].astype(str).str.zfill(4) + df['sequence'].astype(str).str.zfill(2)
df = df[['Request Id', 'id', 'sequence', 'Input_Text', 'Output_Text', 'Input_Tokens', 'Output_Tokens']]

output_file = "topical-chat-json-p.csv"
df.to_csv(output_file, index=False)

print('성공')