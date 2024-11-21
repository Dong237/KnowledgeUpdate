import requests
import os 
import re
import json
import netrc
import io
import math
import time
import torch
import logging
import colorlog
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


## general loading and saving for txt and json files
def read_txt_file(file, encoding='utf-8', strip_lines=True):
    content = []
    with _make_r_io_base(file, mode='r') as f:
        if not isinstance(f, io.TextIOWrapper):
            f = io.TextIOWrapper(f, encoding=encoding)
        content = f.read()
    return content


def write_txt_file(file_path, content, encoding='utf-8'):
    if content is None:
        raise ValueError("Content must be provided when writing to a file.")
    if isinstance(content, list):
        content = "\n".join(content)
    with _make_w_io_base(file_path, mode="w") as f:
        if not isinstance(f, io.TextIOWrapper):
            f = io.TextIOWrapper(f, encoding=encoding)
        f.write(content)


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(
            obj, 
            f, 
            ensure_ascii=False,
            indent=indent, 
            default=default
            )
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def llm_chat_local(model, tokenizer, user_prompt, system_prompt):
        messages = [
            {
                "role": "system", 
                "content": system_prompt
                },
            {
                "role": "user", 
                "content": user_prompt
                },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # make sure to use greedy decoding
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.generation_config.top_k=None
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=512,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# Define rate limit (e.g., 10 requests per minute)
CALLS = 100
PERIOD = 5  # seconds

# Apply both rate limit and retry decorators
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)  # Ensure no more than 10 requests per minute 
@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=4, max=40)
)
def llm_chat(user_prompt, system_prompt):
    api_key = "Zhiyan123"
    model_name = "zhiyan-v2.6-chat-int8"
    url = f'http://192.168.200.212:8100/v1'
    client = OpenAI(api_key=api_key, base_url=url)
    messages = [
            {
                "role": "system", 
                "content": system_prompt
                },
            {
                "role": "user", 
                "content": user_prompt
                },
        ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages, #[{"role": "user", "content": user_prompt}],
            stream=True,
            temperature=0.2,
            top_p=0.7,
        )
        
        full_ans = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_ans += chunk.choices[0].delta.content

        return full_ans.strip()

    except Exception as e:
        print(f"发生错误: {e}")
        return ""
    

## Utilities for logging
def is_wandb_logged_in():
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False
    
    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    return bool(auth)


def setup_logging():
    # Set up file handler
    file_handler = logging.FileHandler("app.log")  
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
            )
        )

    # Set up color console handler using colorlog
    color_handler = colorlog.StreamHandler()
    color_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the lowest log level to capture all messages
    logger.addHandler(file_handler)
    logger.addHandler(color_handler)


## Data Processing
def closest_power_of_2(A):
    """This is for finding a proper overlap token size when chunking"""
    target = A // 10
    lower_exp = int(math.log2(target))
    upper_exp = lower_exp + 1
    lower_power = 2 ** lower_exp
    upper_power = 2 ** upper_exp
    
    if abs(target - lower_power) <= abs(target - upper_power):
        return lower_power
    else:
        return upper_power
    
    
# Function to periodically save data
def periodic_save(json_data_list, json_save_dir, save_interval=900):
    """Save data every `save_interval` seconds."""
    while True:
        time.sleep(save_interval)
        if json_data_list:  # Check if there's any data to save
            jdump(json_data_list, json_save_dir)
            logging.info(f"Saved partial data to {json_save_dir} at {time.ctime()}")


def consumer(json_data_queue, json_data_list):
    while True:
        item = json_data_queue.get()
        if item is None:  # Sentinel value to stop the consumer thread
            break
        json_data_list.append(item)
        json_data_queue.task_done()
        

## For evaluation
def extract_rating(text):
    match = re.search(r'\b[01]\b', text.strip())
    return match.group() if match else None