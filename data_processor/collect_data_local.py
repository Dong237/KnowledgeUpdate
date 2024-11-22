#!/usr/bin/env python

"""
data_collection.py

This script performs data collection and processing tasks by automatically looping through
all existing .txt files in a folder and generating Q&A pairs based on the documents. 

The above process is performed sequentially for each document, and requires a local LLM.

Methods are inspired by the following works: 
    [Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning](https://arxiv.org/abs/2404.00213)
    [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)

"""

import os
import re
import sys
import warnings
import logging
import argparse
import jieba
from tqdm import tqdm
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.helper import (
    read_txt_file, 
    llm_chat_local,
    jdump,
    closest_power_of_2,
    setup_logging
)
from utils.prompts import (
    SYSTEM_PROMPT_DATA_GEN,
    SYSTEM_PROMPT_ROLE_GEN,
    THEME_SUMMARIZATION, 
    FACT_DISTILLATION,
    ROLE_GENERATION,
    ROLE_BASED_QA_DIVERSIFY,
    FACT_BASED_QA_GEN_SKIP,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script for processing text into SFT-ready format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/txt_data",
        help="Path to the directory containing the .txt data files."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/hf_model",
        help="Path to the pre-trained model for generating data from .txt files."
    )
    parser.add_argument(
        "--chunk_size_by_token",
        type=int,
        default=512,
        help="Chunk size measured by tokens, smaller chunk size leads to finer granularity when extracting (more facts being extracted)"
    )
    parser.add_argument(
        "--qa_amount_per_fact",
        type=int,
        default=10,
        help="Number of QA pairs to generate per fact, default to 10 as in Mecklenburg et al, 2024 ."
    )
    parser.add_argument(
        "--role_amount_per_fact",
        type=int,
        default=3,
        help="Number of roles to assign per fact, roles are played by an LLM to augment/diversify the QA pairs for a given fact"
    )
    parser.add_argument(
        "--json_save_dir",
        type=str,
        default="datasets/qa_pairs.json",
        help="Path to save the final json dataset for finetuning"
    )
    return parser.parse_args()


def initialize_model_and_tokenizer(model_name_or_path: str):
    """Initializes and returns the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def _chunking_by_token_size(content: str, overlap_token_size=64, max_token_size=1024):
    tokens = list(jieba.cut(content))
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = ''.join(tokens[start: start + max_token_size])
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        })
    return results


def _parse_diversified_qa_response(diversified_qa_response: str) -> list:
    elements = re.findall(
        r'Q:.*?A:.*?(?=\d+\.|$)', 
        diversified_qa_response, 
        re.DOTALL
    )
    elements = [element.strip() for element in elements]
    return elements


def _parse_numbered_elements(response: str) -> list:
    return re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', response)


def _generate_qa_pairs_from_one_txt_file(data_path: str, data_name: str, qa_amount_per_fact: int, 
                                         role_amount_per_fact: int, chunk_size_by_token: int,
                                         model, tokenizer) -> List[dict]:
    logging.info(f"Collecting data from the single txt file: {data_name}")
    data = read_txt_file(os.path.join(data_path, data_name))
    overlap_token_size = closest_power_of_2(chunk_size_by_token)
    chunking_results = _chunking_by_token_size(
        content=data, 
        overlap_token_size=overlap_token_size,
        max_token_size=chunk_size_by_token 
    )

    qa_pairs_all_chunks = []
    facts = []
    for chunk in tqdm(chunking_results, desc="Processing chunks..."):
        content = chunk["content"]

        logging.info("Summarizing the theme of the chunk")
        theme_response = llm_chat_local(
            model, 
            tokenizer,
            user_prompt=THEME_SUMMARIZATION.format(passage=content), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        if len(theme_response) == 0:
            logging.warning("No theme found for the chunk")
            continue
        logging.info("Extracting facts from the chunk")
        facts_response = llm_chat_local(
            model, 
            tokenizer,
            user_prompt=FACT_DISTILLATION.format(
                theme=theme_response, 
                passage=content
            ), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        parsed_facts = _parse_numbered_elements(facts_response)

        for fact in tqdm(parsed_facts, desc="Processing facts in the chunk"):
            logging.info("Generating a standard Q&A pair based on the fact")
            qa_response = llm_chat_local(
                model, 
                tokenizer,
                user_prompt=FACT_BASED_QA_GEN_SKIP.format(
                    theme=theme_response, 
                    fact=fact
                ), 
                system_prompt=SYSTEM_PROMPT_DATA_GEN
            )
            if qa_response != "SKIP":
                logging.info("Generating possible roles")
                roles_response = llm_chat_local(
                    model, 
                    tokenizer,
                    user_prompt=ROLE_GENERATION.format(
                        amount=role_amount_per_fact,
                        theme=theme_response
                    ),
                    system_prompt=SYSTEM_PROMPT_ROLE_GEN
                )
                parsed_roles_response = _parse_numbered_elements(roles_response)
                
                role_amount_per_fact_actual = len(parsed_roles_response)
                assert qa_amount_per_fact > role_amount_per_fact_actual, "Each role should generate at least one QA pair"
                scheduler = divmod(qa_amount_per_fact, role_amount_per_fact_actual)
                qa_amount_per_role_schedule = [scheduler[0]] * (role_amount_per_fact_actual - 1) + [scheduler[0] + scheduler[1]]

                for idx, role in enumerate(parsed_roles_response): 
                    logging.info("Diversifying the standard Q&A pair given roles")   
                    diversified_qa_response = llm_chat_local(
                        model,
                        tokenizer,
                        user_prompt=ROLE_BASED_QA_DIVERSIFY.format(
                            role=role,
                            theme=theme_response,
                            amount=qa_amount_per_role_schedule[idx],
                            qa_pair=qa_response,
                            fact=fact
                        ), 
                        system_prompt=SYSTEM_PROMPT_DATA_GEN
                    )
                    qa_pairs_per_role = _parse_diversified_qa_response(diversified_qa_response)
                    facts.extend([fact] * len(qa_pairs_per_role))
                    qa_pairs_all_chunks.extend(qa_pairs_per_role)
            else:
                warnings.warn(f"The fact {fact} is too broad or ambiguous to generate any Q&A pairs. Skipping.")
                continue

    qa_pairs_json = []
    for idx, (qa, fact) in enumerate(zip(qa_pairs_all_chunks, facts)):
        try:
            q, a = qa.split('\n')
        except ValueError:
            logging.warning(f"Q&A pair {qa} will be omitted due to splitting failure.")
            continue
        q = q.replace('Q:', '').strip()
        a = a.replace('A:', '').strip()
        qa_pairs_json.append({
            "id": f"identity_{idx}",
            "conversations": [
                {"from": "user", "value": q},
                {"from": "assistant", "value": a}
            ],
            "fact": fact,
            "file_name": data_name
        })
    logging.info(f"Processing for the file {data_name} finished.")
    return qa_pairs_json


def generate_qa_pairs_from_folder(data_path: str, qa_amount_per_fact: int, role_amount_per_fact: int, 
                                  chunk_size_by_token: int, json_save_dir: str, model, tokenizer):
    files = os.listdir(data_path)
    json_data_list = []

    for file in tqdm(files, desc="Processing files..."):
        data_list = _generate_qa_pairs_from_one_txt_file(
            data_path, 
            file,
            qa_amount_per_fact, 
            role_amount_per_fact,
            chunk_size_by_token,
            model,
            tokenizer
        )
        json_data_list.extend(data_list)
    
    jdump(json_data_list, json_save_dir)
    logging.info(f"Saved final JSON data to {json_save_dir}")


if __name__ == "__main__":
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    setup_logging()
    args = _parse_args()
    
    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path)
    
    generate_qa_pairs_from_folder(
        data_path=args.data_path,
        qa_amount_per_fact=args.qa_amount_per_fact, 
        role_amount_per_fact=args.role_amount_per_fact,
        chunk_size_by_token=args.chunk_size_by_token,
        json_save_dir=args.json_save_dir,
        model=model,
        tokenizer=tokenizer
    )
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
