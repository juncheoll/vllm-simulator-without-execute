import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, CSV_Assist
from vllm.sequence import SamplerOutput, ExecuteModelRequest
from vllm.utils import FlexibleArgumentParser, Device
from vllm.inputs.data import TokensPrompt

import torch
from typing_extensions import TypedDict

import time
from typing import Dict, Tuple, List

import logging
from datetime import datetime

from create_vector import (generate_vectors, generate_vectors_same, insert_prefix_vector, generate_mixed_vectors)

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%m-%d %H:%M:%S',
    level=logging.INFO
)

def log_message(message):
    current_time = datetime.now().strftime('%m-%d %H:%M:%S')
    logging.info(f'{message}')

min_tokens = 0
max_tokens = None

file_path = '../dataset/topical-chat-json-p.csv'
csv = CSV_Assist()
csv.read_csv(file_path)

class Data():
    def __init__(self, request_id: str, sequence: int, input_tokens: List[int], output_tokens: List[int]):
        
        self.request_id: str = request_id
        self.sequence: int = sequence
        self.input_tokens: List[int] = input_tokens
        self.output_tokens: List[int] = output_tokens


num_request = 1024
dataset: Dict[int, List[Data]] = {}


def create_test_dataset():
    i = 0
    for id, group in csv.df.groupby('id'):
        data_list = []
        for _, row in group.iterrows():
            data_list.append(Data(
                request_id=str(row.name).zfill(6),
                sequence=row['sequence'],
                input_tokens=row['Input_Tokens'],
                output_tokens=row['Output_Tokens']
            ))
        
        dataset[id] = data_list
        i += 1
        if i == num_request:
            return



sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=min_tokens, max_tokens=max_tokens)

def next_query(request_id: str) -> Tuple[str, List[int]] | Tuple[None, None]:
    id = int(request_id[0:4])
    sequence = int(request_id[4:6])
    
    if sequence == dataset[id][-1].sequence:
        return None, None
    
    new_input_tokens = []
    
    data_list = dataset[id]
    
    for i in range(sequence+1):
        new_input_tokens.extend(data_list[i].input_tokens)
        new_input_tokens.extend(data_list[i].output_tokens)
    
    return data_list[sequence].request_id, new_input_tokens
    
    

def running(engine: LLMEngine, finished_log, last_sampled_log):
    i = 0
    while engine.has_unfinished_requests():
        i += 1
        #print(f'step: {i}')
        request_outputs, outputs, request = engine.simulate_step(csv, execute=True)
        cpu_rate = engine.scheduler[0].get_prefix_cache_hit_rate(device=Device.CPU)
        gpu_rate = engine.scheduler[0].get_prefix_cache_hit_rate(device=Device.GPU)
        #print(f'CPU_Rate : {cpu_rate}, GPU_Rate : {gpu_rate}')

        request_id_of_ModelRequest = []
        parent_seq_id_of_sample = []
        request_id_of_RequestOutput = []

        if len(request.finished_requests_ids) != 0:
            finished_log.extend(request.finished_requests_ids)

        #print(len(request.blocks_to_swap_in), len(request.blocks_to_swap_out), sep='\t')

        if request.last_sampled_token_ids is not None:
            last_sampled_log.append(request.last_sampled_token_ids)

        for seq_group_meta in request.seq_group_metadata_list:
            request_id_of_ModelRequest.append(seq_group_meta.request_id)

        output = outputs[0]
        #print(output)
        for o in output.outputs:
            sample = o.samples[0]
            parent_seq_id_of_sample.append(sample.parent_seq_id)
            #print(f'logprobs : {sample.logprobs}')


        
        for request_output in request_outputs:
            request_id_of_RequestOutput.append(request_output.request_id)
            if request_output.finished:
                #print(request_output.outputs)
                next_id, next_input = next_query(request_output.request_id)
                if next_input is not None:
                    #print(f'add, {request_output.request_id}, {next_id}, {next_input}')
                    engine.add_request(next_id,
                                       TokensPrompt({
                                           'prompt_token_ids': next_input}),
                                        sampling_params)
                    


        if len(request_id_of_ModelRequest) == len(parent_seq_id_of_sample) == len(request_id_of_RequestOutput) and all(a == b == c for a, b, c in zip(request_id_of_ModelRequest, parent_seq_id_of_sample, request_id_of_RequestOutput)):
            print('error error!')
            print(f'request_id_of_ModelRequest : {request_id_of_ModelRequest}')
            print(f'parent_seq_id_of_sample : {parent_seq_id_of_sample}')
            print(f'request_id_of_RequestOutput : {request_id_of_RequestOutput}')

def process_requests(engine: LLMEngine):
    """Continuously process a list of prompts and handle the outputs."""
    stat_logger = engine.stat_loggers
    stat_logger_logging = stat_logger.get('logging')
    scheduler = engine.scheduler[0]
    print(f'watermark:{scheduler.block_manager.watermark}')

    finished_log = []
    last_sampled_log = []
    
    for id, data_list in dataset.items():
        engine.add_request(data_list[0].request_id,
                           TokensPrompt({
                               'prompt_token_ids':data_list[0].input_tokens}),
                           sampling_params)
        

    running(engine, finished_log, last_sampled_log)

    print(f'finished_log : {len(finished_log)}')
    print(f'last_sampled_log : {last_sampled_log}')

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs(#model="meta-llama/Llama-2-7b-chat-hf",
                             model="facebook/opt-1.3b",
                             #model='openai-community/gpt2-medium',
                             #model='meta-llama/Llama-3.2-1B',
                             #dtype=torch.bfloat16,
                             #trust_remote_code=True,
                             #quantization="bitsandbytes",
                             #load_format="bitsandbytes",
                             #kv_cache_dtype="fp8",
                             #quantization_param_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json",
                             swap_space=16,
                             preemption_mode='swap',
                             )
    engine_args.gpu_memory_utilization = 0.9
    engine_args.enable_prefix_caching = True
    #engine_args.max_num_seqs = 32

    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    stat_logger = engine.stat_loggers
    stat_logger_logging = stat_logger.get('logging')
    stat_logger_prometheus = stat_logger.get('prometheus')
    stat_logger_logging.local_interval = 1
    create_test_dataset()
    start_time = time.time()
    process_requests(engine)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"executed in: {execution_time:.6f} seconds")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
