#!/usr/bin/env python

import os
import re
import sys
import traceback
import warnings
import logging
import argparse
from tqdm import tqdm
from typing import List
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import jieba  # Import jieba for tokenization

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import (
    read_txt_file, 
    llm_chat, 
    jdump,
    closest_power_of_2,
    setup_logging,
    periodic_save,
    consumer,
)
from prompts import (
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing of files."
    )
    return parser.parse_args()

def _chunking_by_token_size(content: str, overlap_token_size=64, max_token_size=1024):
    # Tokenize the content using jieba
    tokens = list(jieba.cut(content))
    results = []
    
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = ''.join(tokens[start : start + max_token_size])
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
                                         role_amount_per_fact: int, chunk_size_by_token: int) -> List[dict]:
    logging.info(f"Collecting data from the single txt file: {data_name}")
    data = read_txt_file(os.path.join(data_path, data_name))

    assert qa_amount_per_fact > role_amount_per_fact, "Each role should generate at least one QA pair"
    scheduler = divmod(qa_amount_per_fact, role_amount_per_fact)
    qa_amount_per_role_schedule = [scheduler[0]]*(role_amount_per_fact-1) + [scheduler[0]+scheduler[1]]

    # No need to use a Hugging Face tokenizer, as we're using jieba now
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
        theme_response = llm_chat(
            user_prompt=THEME_SUMMARIZATION.format(passage=content), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        if len(theme_response) == 0:
            logging.warning("No theme found for the chunk")
            continue
        logging.info("Extracting facts from the chunk")
        facts_response = llm_chat(
            user_prompt=FACT_DISTILLATION.format(
                theme=theme_response, 
                passage=content
            ), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        parsed_facts = _parse_numbered_elements(facts_response)

        for fact in tqdm(parsed_facts, desc="Processing facts in the chunk"):
            logging.info("Generating a standard Q&A pair based on the fact")
            qa_response = llm_chat(
                user_prompt=FACT_BASED_QA_GEN_SKIP.format(
                    theme=theme_response, 
                    fact=fact
                ), 
                system_prompt=SYSTEM_PROMPT_DATA_GEN
            )
            if qa_response != "SKIP" and len(qa_response) > 0:
                logging.info("Generating possible roles")
                roles_response = llm_chat(
                    user_prompt=ROLE_GENERATION.format(
                        amount=role_amount_per_fact,
                        theme=theme_response
                    ),
                    system_prompt=SYSTEM_PROMPT_ROLE_GEN
                )
                parsed_roles_response = _parse_numbered_elements(roles_response)

                for idx, role in enumerate(parsed_roles_response): 
                    logging.info("Diversifying the standard Q&A pair given roles")   
                    diversified_qa_response = llm_chat(
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
                    facts.extend([fact]*len(qa_pairs_per_role))
                    qa_pairs_all_chunks.extend(qa_pairs_per_role)
            else:
                message = f"The fact {fact} is too broad or ambiguous to generate any Q&A pairs. It will be skipped"
                warnings.warn(message)
                continue

    qa_pairs_json = []
    for idx, (qa, fact) in enumerate(zip(qa_pairs_all_chunks, facts)):
        try:
            q, a = qa.split('\nA: ')
        except ValueError as e:
            logging.error(f"The following exception occurred while splitting the Q&A pair")
            logging.warning(f"Q&A pair {qa} will be omitted in the final dataset due to splitting failure")
            continue
        q = q.replace('Q: ', '')
        qa_pairs_json.append({
            "id": f"identity_{idx}",
            "conversations": [
                {"from": "user", "value": q},
                {"from": "assistant", "value": a}
            ],
            "fact": fact,
            "file_name": data_name
        })
    logging.info(f"Processing for the file {data_name} finished")
    return qa_pairs_json


def generate_qa_pairs_from_folder(
    data_path: str, 
    qa_amount_per_fact: int, 
    role_amount_per_fact: int, 
    chunk_size_by_token: int, 
    json_save_dir: str, 
    num_workers: int
    ):
    
    files = os.listdir(data_path)
    json_data_queue = Queue(maxsize=10000)  # Limit queue size to 10k items
    json_data_list = []
    
    # Start periodic save thread
    save_thread = threading.Thread(target=periodic_save, args=(json_data_list, json_save_dir))
    save_thread.daemon = True  # Set as daemon so it stops when the main program exits
    save_thread.start()

    # Start consumer thread to collect data
    consumer_thread = threading.Thread(target=consumer, args=(json_data_queue, json_data_list))
    consumer_thread.start()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                _generate_qa_pairs_from_one_txt_file,
                data_path,
                file,
                qa_amount_per_fact,
                role_amount_per_fact,
                chunk_size_by_token
            ): file for file in files
        }

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files..."):
            file = future_to_file[future]
            try:
                data_list = future.result()
                for item in data_list:
                    json_data_queue.put(item)  # Add data to queue
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
                logging.error("".join(traceback.format_exception(None, e, e.__traceback__)))

    # Stop the consumer thread by sending a sentinel value
    json_data_queue.put(None)  # Signal the consumer to stop
    consumer_thread.join()  # Wait for the consumer thread to finish
    
    # Wait for the periodic save thread to finish saving the final data
    save_thread.join()

    # Save final data to JSON
    jdump(json_data_list, json_save_dir)
    logging.info(f"Saved final JSON data to {json_save_dir}")


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    generate_qa_pairs_from_folder(
        data_path=args.data_path,
        qa_amount_per_fact=args.qa_amount_per_fact, 
        role_amount_per_fact=args.role_amount_per_fact,
        chunk_size_by_token=args.chunk_size_by_token, 
        json_save_dir=args.json_save_dir, 
        num_workers=args.num_workers
    )
