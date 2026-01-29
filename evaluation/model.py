# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import vllm
from prompts import PROMPT_RETRIEVAL, PROMPT_NO_RETRIEVAL, BOX_FORMAT

from vllm.lora.request import LoRARequest

class InstructModel:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", lora_path=None, decode_batch_size=8, vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.85):
        self.initialize_models(model_name, lora_path, decode_batch_size, vllm_tensor_parallel_size, vllm_gpu_memory_utilization)

    def initialize_models(self, model_name, lora_path, decode_batch_size, vllm_tensor_parallel_size, vllm_gpu_memory_utilization):
        self.model_name = model_name
        self.lora_path = lora_path

        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=vllm_tensor_parallel_size, 
            gpu_memory_utilization=vllm_gpu_memory_utilization, 
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=bool(lora_path),
            max_lora_rank=64 if lora_path else None, # Buffer for rank 16
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.batch_size = decode_batch_size  


    def get_batch_size(self) -> int:
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any], top_p=0.9, temperature=0.6, max_new_tokens=1024, n_answer=32, max_seq_length=32768, is_rag=True) -> List[List[str]]:

        queries = batch["query"]
        references = batch["retrieved_chunks"] if is_rag else [[] for _ in range(len(batch["query"]))]
        query_times = batch["query_time"]

        formatted_prompts_ids = self.format_prompts(queries, query_times, references, is_rag)

        max_prompt_tokens = max_seq_length - max_new_tokens
        
        formatted_prompts = []
        prompt_lengths = []
        for prompt_ids in formatted_prompts_ids:
            prompt_token_count = len(prompt_ids)
            if prompt_token_count > max_prompt_tokens:
                prompt_ids = prompt_ids[:(max_prompt_tokens-9)] + prompt_ids[-9:]
            prompts = self.tokenizer.decode(prompt_ids)
            formatted_prompts.append(prompts)
            prompt_lengths.append(len(prompt_ids))

        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=n_answer,
                top_p=top_p,
                temperature=temperature,
                skip_special_tokens=True,
                max_tokens=max_new_tokens,
            ),
            use_tqdm = False,
            lora_request=LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None
        )

        answers = [] 
        for response in responses:
            prompt_answers = []
            for output in response.outputs:
                prompt_answers.append(output.text)
            answers.append(prompt_answers)
            
        return formatted_prompts, prompt_lengths, answers

    def batch_generate_answer_with_tokens(self, batch: Dict[str, Any], top_p=0.9, temperature=0.6, max_new_tokens=1024, n_answer=32, max_seq_length=32768, is_rag=True):
        queries = batch["query"]
        references = batch["retrieved_chunks"] if is_rag else [[] for _ in range(len(batch["query"]))]
        query_times = batch["query_time"]

        formatted_prompts_ids = self.format_prompts(queries, query_times, references, is_rag)

        max_prompt_tokens = max_seq_length - max_new_tokens

        formatted_prompts = []
        prompt_lengths = []
        for prompt_ids in formatted_prompts_ids:
            prompt_token_count = len(prompt_ids)
            if prompt_token_count > max_prompt_tokens:
                prompt_ids = prompt_ids[:(max_prompt_tokens-9)] + prompt_ids[-9:]
            prompts = self.tokenizer.decode(prompt_ids)
            formatted_prompts.append(prompts)
            prompt_lengths.append(len(prompt_ids))

        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=n_answer,
                top_p=top_p,
                temperature=temperature,
                skip_special_tokens=True,
                max_tokens=max_new_tokens,
                logprobs=1,  # Enable logprobs for the top 1 token
            ),
            use_tqdm=False
        )

        answers = []
        response_token_ids = []
        response_tokens = []
        response_lengths = []
        response_logprobs = []
        response_cumulative_logprobs = []
        for response in responses:
            prompt_answers = []
            prompt_token_ids = []
            prompt_tokens = []
            prompt_lengths_out = []
            prompt_logprobs = []
            prompt_cumulative_logprobs = []
            for output in response.outputs:
                prompt_answers.append(output.text)
                if hasattr(output, "token_ids") and output.token_ids is not None:
                    out_token_ids = output.token_ids
                else:
                    out_token_ids = self.tokenizer.encode(output.text, add_special_tokens=False)
                prompt_token_ids.append(out_token_ids)
                try:
                    out_tokens = self.tokenizer.convert_ids_to_tokens(out_token_ids)
                except Exception:
                    out_tokens = []
                prompt_tokens.append(out_tokens)
                prompt_lengths_out.append(len(out_token_ids))
                
                # Extract logprobs if available
                if hasattr(output, "logprobs") and output.logprobs is not None:
                    # Extract logprobs for each token position as (token, prob) tuples
                    token_logprobs = []
                    for i, token_logprob_dict in enumerate(output.logprobs):
                        if token_logprob_dict and i < len(out_tokens):
                            # Get the logprob for the actual token (first entry is usually the chosen token)
                            first_token_id = list(token_logprob_dict.keys())[0]
                            logprob_value = token_logprob_dict[first_token_id].logprob
                            token_text = out_tokens[i] if i < len(out_tokens) else f"token_{first_token_id}"
                            token_logprobs.append((token_text, logprob_value))
                        else:
                            token_text = out_tokens[i] if i < len(out_tokens) else f"token_{i}"
                            token_logprobs.append((token_text, None))
                    prompt_logprobs.append(token_logprobs)
                else:
                    # Create tuples with token text and None probability
                    token_logprobs = [(out_tokens[i] if i < len(out_tokens) else f"token_{i}", None) for i in range(len(out_token_ids))]
                    prompt_logprobs.append(token_logprobs)
                
                # Extract cumulative logprob if available
                if hasattr(output, "cumulative_logprob") and output.cumulative_logprob is not None:
                    prompt_cumulative_logprobs.append(output.cumulative_logprob)
                else:
                    prompt_cumulative_logprobs.append(None)
                    
            answers.append(prompt_answers)
            response_token_ids.append(prompt_token_ids)
            response_tokens.append(prompt_tokens)
            response_lengths.append(prompt_lengths_out)
            response_logprobs.append(prompt_logprobs)
            response_cumulative_logprobs.append(prompt_cumulative_logprobs)

        return formatted_prompts, prompt_lengths, answers, response_token_ids, response_tokens, response_lengths, response_logprobs, response_cumulative_logprobs

    def format_prompts(self, queries, query_times, references, is_rag=True):

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            
            formatted_refs = []
            total_length = 0
            idx = 1
            for reference in references[_idx]:
                if 'bge_score' in reference and reference['bge_score'] is not None:
                    if reference['bge_score'] < 2:
                        continue
                elif 'oracle_score' in reference and reference['oracle_score'] is not None:
                    if reference['oracle_score'] < 3:
                        continue
                ref_text = f"<DOC>\nDocument [{idx}]: {reference['chunk_text']}\n</DOC>"
                if total_length + len(ref_text) > 80000:
                    break
                formatted_refs.append(ref_text)
                total_length += len(ref_text)
                idx += 1

            formatted_references = "\n".join(formatted_refs)
            user_message = f"### Question\n{query}\n"
            user_message += f"### Query Time\n{query_time}\n"

            if is_rag:
                task_message = PROMPT_RETRIEVAL + '\n' + BOX_FORMAT
                user_message += f"### References\n{formatted_references}"
            else:
                task_message = PROMPT_NO_RETRIEVAL + '\n' + BOX_FORMAT

            final_message = task_message + '\n' + user_message
            
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": final_message},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts
