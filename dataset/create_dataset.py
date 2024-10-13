import pandas as pd
from datasets import load_dataset
import json
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_text_for_input(text):
    if pd.isna(text):
        return []
    
    conversation = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
    tokens = [tokens[0]] + tokens[4:-4]
    return tokens

def tokenize_text_for_output(text):
    if pd.isna(text):
        return []
    
    conversation = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
    tokens = tokens[4:-5] + [2]
    return tokens

def tokens_to_json(tokens):
    return json.dumps(tokens)


dataset = load_dataset("gopalkalpande/bbc-news-summary", split='train')
df = pd.DataFrame(dataset)

if 'File_path' in df.columns:
    df = df.drop(columns=['File_path'])
else:
    print("Warning: 'File_path' 컬럼 존재하지 않음.")

df.reset_index(drop=True, inplace=True)
df.index = df.index + 1
df.index.name = 'Request Id'

df = df.rename(columns={
    'Articles' : 'Input_Content',
    'Summaries' : 'Output_Content'
})

df['Input_Tokens'] = df['Input_Content'].apply(tokenize_text_for_input)
df['Output_Tokens'] = df['Output_Content'].apply(tokenize_text_for_output)

df['Input_Tokens'] = df['Input_Tokens'].apply(tokens_to_json)
df['Output_Tokens'] = df['Output_Tokens'].apply(tokens_to_json)

output_file = "bbc-news-summary-local.csv"
df.to_csv(output_file, index=True)

print('성공')

