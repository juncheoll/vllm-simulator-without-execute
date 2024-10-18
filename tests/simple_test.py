import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.sequence import SamplerOutput, ExecuteModelRequest
from vllm.utils import FlexibleArgumentParser
from vllm.inputs.data import TokensPrompt

import torch
from typing_extensions import TypedDict

import time

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


start = 1
end = 29889
lenght = 50
num_vectors = 128

min_tokens = 0
max_tokens = 128



def create_test_prompts() -> List[Tuple[List[int], SamplingParams]]:
    prompts = []
    vectors = generate_vectors(start, end, lenght, num_vectors)
    for i in range(len(vectors)):
        vectors[i].insert(0, start)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=min_tokens, max_tokens=max_tokens)
    return [(TokensPrompt({'prompt_token_ids':vector}), sampling_params) for vector in vectors]


def create_test_prompts_v2() -> List[Tuple[List[int], SamplingParams]]:
    prompts = []
    vectors = generate_vectors(start, end, lenght, num_vectors)
    for i in range(len(vectors)):
        vectors[i].insert(0, start)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=min_tokens, max_tokens=max_tokens)
    output = []

    for i in range(len(vectors)):
        sampling_params.min_tokens += 1
        sampling_params.max_tokens += 1
        output.append((TokensPrompt({'prompt_token_ids':vectors[i]}), sampling_params))

    return output

def running(engine, finished_log, last_sampled_log):
    while engine.has_unfinished_requests():

        request_outputs, outputs, request = engine.step()

        request_id_of_ModelRequest = []
        parent_seq_id_of_sample = []
        request_id_of_RequestOutput = []

        if len(request.finished_requests_ids) != 0:
            finished_log.extend(request.finished_requests_ids)

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
                print(request_output.outputs)


        if len(request_id_of_ModelRequest) == len(parent_seq_id_of_sample) == len(request_id_of_RequestOutput) and all(a == b == c for a, b, c in zip(request_id_of_ModelRequest, parent_seq_id_of_sample, request_id_of_RequestOutput)):
            print('error error!')
            print(f'request_id_of_ModelRequest : {request_id_of_ModelRequest}')
            print(f'parent_seq_id_of_sample : {parent_seq_id_of_sample}')
            print(f'request_id_of_RequestOutput : {request_id_of_RequestOutput}')

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[List[int], SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    stat_logger = engine.stat_loggers
    stat_logger_logging = stat_logger.get('logging')
    scheduler = engine.scheduler[0]
    print(f'watermark:{scheduler.block_manager.watermark}')

    finished_log = []
    last_sampled_log = []
    request_id = 0 

    for i in range(int(len(test_prompts)/2)):
        vector, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id), vector, sampling_params)
        request_id += 1

    running(engine, finished_log, last_sampled_log)

    while test_prompts:
        vector, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id), vector, sampling_params)
        request_id += 1

    running(engine, finished_log, last_sampled_log)

    print(f'finished_log : {finished_log}')
    print(f'last_sampled_log : {last_sampled_log}')

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-chat-hf", # "facebook/opt-125m" 
                             dtype=torch.bfloat16,
                             trust_remote_code=True,
                             quantization="bitsandbytes",
                             load_format="bitsandbytes",
                             kv_cache_dtype="fp8",
                             #quantization_param_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json",
                             swap_space=16,
                             preemption_mode='swap',
                             )
    engine_args.gpu_memory_utilization = 0.7
    engine_args.enable_prefix_caching = True
    #engine_args.max_num_seqs = 32

    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    stat_logger = engine.stat_loggers
    stat_logger_logging = stat_logger.get('logging')
    stat_logger_prometheus = stat_logger.get('prometheus')
    stat_logger_logging.local_interval = 10

    test_prompts = create_test_prompts_v2()
    start_time = time.time()
    process_requests(engine, test_prompts)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"executed in: {execution_time:.6f} seconds")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
