import time
from vllm import LLM, SamplingParams, CSV_Assist
import torch


def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")
    return end_time - start_time

# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='meta-llama/Llama-3.2-3B',
    #model='lmsys/longchat-7b-16k',
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
    load_format='bitsandbytes',
    enable_prefix_caching=True,
    max_model_len=4096
)

sampling_params = SamplingParams(temperature=0, max_tokens=100)


total_time = 0

csv = CSV_Assist()
file_path = '../dataset/shuffled_squad.csv'
csv = CSV_Assist()
csv.read_csv(file_path)
prompt_list = csv.get_rows_input_text()


total_time += get_generation_time(
    llm,
    sampling_params,
    prompt_list)

print(f'total time: {total_time}')