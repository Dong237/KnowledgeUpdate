"""
This script processes and evaluates model responses to determine the accuracy and comparative win rate
between a fine-tuned model and a base model, using LLM-based prompts for scoring. The script loads
response data, formats evaluation prompts, and interacts with an LLM to generate accuracy and win rate 
evaluations, with configurable retry and delay parameters.

Main Components:

1. `_parse_args`: Parses command-line arguments for paths, filenames, and retry configurations.

2. `_call_llm_with_max_attempts`: Calls the LLM with retry attempts and exponential backoff.

3. `_get_acc_eval_response` and `_get_win_rate_eval_response`: Format and send accuracy or win rate 
   prompts for individual responses to the LLM.

4. `_evaluate`: Main evaluation function that runs either accuracy or win rate evaluations depending on 
   the input data structure, with results saved to disk.

5. `main`: Entry point that manages concurrent execution of evaluation tasks with a thread pool.
"""


import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import time
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Union, Literal
from concurrent.futures import ThreadPoolExecutor
from utils import (
    llm_chat,
    jload, 
    jdump,
    setup_logging
)
from prompts import (
    SYSTEM_PROMPT_EVAL,
    ANSWER_ACCURACY,
    WIN_RATE
)


FOLDER_TO_SAVE = "eval_results"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script for processing text into SFT-ready format"
        )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets",
        help="Path to the directory containing the model answers."
    )
    parser.add_argument(
        "--data_name_sft",
        type=str,
        default="answers_sft_model_7B.json",
        help="Path to the file that contains the finetuned model's answers"
    )
    parser.add_argument(
        "--data_name_base",
        type=str,
        default="answers_base_model_7B.json",
        help="Path to the file that contains the base model's answers"
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of attempts of calling the LLM for each evaluation."
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=2,
        help="Delay between retry attempts in seconds."
    )
    return parser.parse_args()


def _call_llm_with_max_attempts(
        user_prompt, 
        max_attempts, 
        retry_delay
        ) -> Union[int, str]:
    attempts = 0
    while attempts < max_attempts:
        eval_response = llm_chat(
            user_prompt=user_prompt, 
            system_prompt=SYSTEM_PROMPT_EVAL
            )
        if eval_response in ["0", "1"]:
            return eval(eval_response)
        logging.info("Evaluation output is misformatted, trying reevaluating...")
        attempts += 1
        time.sleep(retry_delay * (2 ** attempts))  # Exponential backoff
    return "SKIP"


def _get_acc_eval_response(
        answer, 
        max_attempts=5, 
        retry_delay=2,
        ): 
    user_prompt = ANSWER_ACCURACY.format(
        fact=answer["fact"], 
        question=answer["conversations"][0]["value"],
        answer=answer["conversations"][1]["value"]
        )
    eval_result = _call_llm_with_max_attempts(
        user_prompt, 
        max_attempts=max_attempts, 
        retry_delay=retry_delay
        )
    return eval_result


def _get_win_rate_eval_response(
        answer_sft,
        answer_base, 
        max_attempts=5, 
        retry_delay=1,
        ):
    user_prompt = WIN_RATE.format(
        fact=answer_sft["fact"], 
        question=answer_sft["conversations"][0]["value"],
        answer_sft=answer_sft["conversations"][1]["value"],
        answer_base=answer_base["conversations"][1]["value"]
        )
    eval_result = _call_llm_with_max_attempts(
        user_prompt, 
        max_attempts=max_attempts, 
        retry_delay=retry_delay
        )
    return eval_result


def _evaluate(
    data_path: str,
    data_name: Union[dict, str],
    max_attempts: int = 5, 
    retry_delay: int = 2,
    verbose: bool = True,
    eval_method: Literal["optimistic", "pessimistic"] = "optimistic",
    ) -> Optional[List[int]]:
    """
    Evaluates model responses for accuracy or win rate, calling an LLM to judge response quality.
    
    Parameters:
    ----------
    data_path : str
        Path to the directory containing model responses.
    data_name : Union[dict, str]
        Data identifier specifying the response file(s). If a dictionary, performs a win rate comparison 
        between `answers_sft` and `answers_base`. If a string, performs accuracy evaluation on that file.
    max_attempts : int, optional
        Maximum number of retry attempts when calling the LLM (default is 5).
    retry_delay : int, optional
        Delay in seconds between retry attempts, with exponential backoff applied (default is 2).
    verbose : bool, optional
        If True, prints evaluation accuracy at the end of the process (default is True).
    eval_method : Literal["pos", "neg"], optional
        Determines the calculation method for accuracy: "optimistic" counts "SKIP" as 1 while "pessimistic"
        counts only 1s.    
    Returns:
    -------
    Optional[List[int]]
        List of evaluation results for each response; 1 for correct, 0 for incorrect, or "SKIP" for errors.
        In the case of Win Rate evaluation, 1 means a win for the SFT model and 0 otherwise.
        
    Raises:
    ------
    TypeError
        If `data_name` is neither a dictionary nor a string.
    """

    eval_results = []
    data_path = Path(data_path)  # Convert to Path object
    if isinstance(data_name, dict):
        logging.info("Start the evaluation of win rate...")
        answers_sft = jload(data_path / data_name["answers_sft"])
        answers_base = jload(data_path / data_name["answers_base"])
        suffix1 = "_win_rate"
        suffix2 = data_name["answers_sft"].split("answers_sft")[-1]
        for answer_sft, answer_base in tqdm(
            zip(answers_sft, answers_base), 
            desc="Evaluating the win rate..."):
            eval_result = _get_win_rate_eval_response(
                answer_sft,
                answer_base, 
                max_attempts=max_attempts, 
                retry_delay=retry_delay,
                )
            eval_results.append(eval_result)

    elif isinstance(data_name, str):
        logging.info("Start the evaluation of accuracy...")
        answers = jload(data_path / data_name)
        suffix1 = "_acc"
        suffix2 = data_name.split("answers")[-1]
        for answer in tqdm(answers, desc="Evaluating accuracy.."):
            eval_result = _get_acc_eval_response(
                answer, 
                max_attempts=max_attempts, 
                retry_delay=retry_delay
                )
            eval_results.append(eval_result)

    else:
        raise TypeError("data_name can only be either a dict or a string")
    
    jdump(eval_results, os.path.join(FOLDER_TO_SAVE,"eval"+suffix1+suffix2+".json"))

    if verbose:
        if eval_method=="optimistic":

            result = 1-eval_results.count(0)/len(answers)
        else:
            result = eval_results.count(1)/len(answers)
        print(f"Evaluation accuracy is {eval_results.count(1)/len(answers):.4%}")


def main():
    """
    Main function that orchestrates concurrent execution of evaluation tasks.
    
    It sets up logging, parses command-line arguments, and defines evaluation tasks
    for accuracy and win rate comparisons. Each task is executed concurrently
    using a thread pool for efficient parallel processing.
    """

    setup_logging()
    args = _parse_args()
    data_name_win_rate = {
        "data_name_sft": args.data_name_sft,
        "data_name_base": args.data_name_base
    }
    
    # Define the evaluation tasks with relevant parameters
    tasks = [
        (args.data_path, args.data_name_sft, args.max_attempts, args.retry_delay),
        (args.data_path, args.data_name_base, args.max_attempts, args.retry_delay),
        (args.data_path, data_name_win_rate, args.max_attempts, args.retry_delay),
    ]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_evaluate, *task) for task in tasks]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()