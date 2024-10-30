import os
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import warnings
import argparse
from tqdm import tqdm
from typing import List
from utils import read_txt_file, llm_chat, jdump
from transformers import AutoTokenizer
from prompts import (
    system_prompt_data_gen,
    system_prompt_eval,
    system_prompt_role_gen,
    theme_summarization, 
    fact_distillation,
    role_generation,
    role_based_qa_diversify,
    fact_based_qa_gen_skip,
    answer_accuracy
    )


def parse_args():
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
        "--data_name",
        type=str,
        default="data.txt",
        help="Name of the data file to be processed."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/hf_model",
        help="Path to the pre-trained model or model identifier for initializing the tokenizer."
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


def _chunking_by_token_size(
    content: str, 
    tokenizer: AutoTokenizer,
    overlap_token_size=128, 
    max_token_size=1024, 
):
    tokens = tokenizer.encode(content)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = tokenizer.decode(
            tokens[start : start + max_token_size]
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def _parse_diversified_qa_response(diversified_qa_response: str) -> list:
    """
    Parse this type of string '1. Q: xxx？\nA: xxx。\n2. Q: xxx？\nA: xx \n3... '
    into ["Q: xxx？\nA: xxx。", "Q: xxx？\nA: xxx。", ...]
    """
    elements = re.findall(
        r'Q:.*?A:.*?(?=\d+\.|$)', 
        diversified_qa_response, 
        re.DOTALL
        )
    # Remove extra whitespace from each element
    elements = [element.strip() for element in elements]
    return elements

def _parse_numbered_elements(response: str) -> list:
    """
    Parse this type of string '1.  xxx\n2. yyy\n3... '
    into ["xxx", "yyy", ...]
    """
    return re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', response)


def main(
        data: str, 
        model_name_or_path: str,
        qa_amount_per_fact: int, 
        role_amount_per_fact: int,
        json_save_dir: str,
        ) -> List[str]:
    """
    Generate fact-based Q&A pairs given a text .txt file, and save them according
    to the json format required for fine-tuning the LLM

    Args:
        data: The text data in string format.
        model_name_or_path: Name of the tokenizer, could be a local dir or hf repo.
        qa_amount_per_fact: The numebr of Q&A pairs to be generated for each fact, default \
        is 10 as in Mecklenburg et al, 2024.
        role_amount_per_fact: The number of roles to be played by the LLM in order to augment \
        the Q&A pair for a certain fact from multiple perspectives.
        json_save_dir: Path to save the final json data for finetuning task.
    """

    assert qa_amount_per_fact > role_amount_per_fact, "Each role should generate at least one QA pair"
    scheduler = divmod(qa_amount_per_fact, role_amount_per_fact)
    qa_amount_per_role_schedule = [scheduler[0]]*(role_amount_per_fact-1) + [scheduler[0]+scheduler[1]]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        )

    chunking_results = _chunking_by_token_size(
        content=data, 
        tokenizer=tokenizer,
        max_token_size=tokenizer.model_max_length # TODO maybe I should decrease this number in case LLM gives incomplete amount of facts for one big article
        )


    qa_pairs_all_chunks = []
    for chunk in tqdm(chunking_results, desc="Processing chunks..."):
        chunk_tokens = chunk["tokens"]
        content = chunk["content"]

        # extracting facts from chunked texts
        theme_response = llm_chat(
            user_prompt=theme_summarization.format(passage=content), 
            system_prompt=system_prompt_data_gen
            )
        facts_response = llm_chat(
            user_prompt=fact_distillation.format(
                theme=theme_response, 
                passage=content
                ), 
            system_prompt=system_prompt_data_gen
        )
        parsed_facts = _parse_numbered_elements(facts_response)

        qa_pairs_all_facts = []
        for fact in tqdm(parsed_facts, desc="Processing facts in the chunk..."):
            # generate a qa pair from the fact as example
            qa_response = llm_chat(
                user_prompt=fact_based_qa_gen_skip.format(
                    theme=theme_response, 
                    fact=fact
                    ), 
                system_prompt=system_prompt_data_gen
                )
            # only proceed when the fact is not too broad
            if qa_response!="SKIP":
                # generate different roles
                roles_response = llm_chat(
                    user_prompt=role_generation.format(
                        amount=role_amount_per_fact,
                        theme=theme_response
                    ),
                    system_prompt=system_prompt_role_gen
                    )
                parsed_roles_response = _parse_numbered_elements(roles_response)

                for idx, role in enumerate(parsed_roles_response):    
                    diversified_qa_response = llm_chat(
                        user_prompt=role_based_qa_diversify.format(
                            role=role,
                            theme=theme_response,
                            amount=qa_amount_per_role_schedule[idx],
                            qa_pair=qa_response,
                            fact=fact
                            ), 
                        system_prompt=system_prompt_data_gen
                    )
                    qa_pairs_per_role = _parse_diversified_qa_response(diversified_qa_response)
                    qa_pairs_all_facts.extend(qa_pairs_per_role)
            else:
                message = f"The fact {fact} is too broad or ambiguous to generate any Q&A pairs. will be skipped"
                warnings.warn(message)
                continue
        qa_pairs_all_chunks.extend(qa_pairs_all_facts)
    
    
    qa_pairs_json = []
    for idx, qa in enumerate(qa_pairs_all_chunks):
        q, a = qa.split('\nA: ')
        q = q.replace('Q: ', '')
        # TODO this json format may probably need adjustment upon delivery
        qa_pairs_json.append(
            {
            "id": f"identity_{idx}",
            "conversations": [
                {
                    "from": "user",
                    "value": q
                },
                {
                    "from": "assistant",
                    "value": a
                }
            ]
            }
        )
    jdump(qa_pairs_json, json_save_dir)
    return qa_pairs_all_chunks, qa_pairs_json


if __name__ == "__main__":
    args = parse_args()
    # generating fact-based qa pair from one single txt file
    qa_pairs_all_chunks, qa_pairs_json = main(
        data=read_txt_file(
            os.path.join(args.data_path, args.data_name)
            ), 
        model_name_or_path=args.model_name_or_path,
        qa_amount_per_fact=args.qa_amount_per_fact, 
        role_amount_per_fact=args.role_amount_per_fact,
        json_save_dir=args.json_save_dir,
        )
    print(qa_pairs_all_chunks)