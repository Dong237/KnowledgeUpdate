import requests
import os 
import json
import netrc
import io
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


## Getting resposne from taiyi-130B model
# TODO adapt the retry mechanism according to the different purposes of generation
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=40)
)
def llm_chat(
        user_prompt, 
        url="http://10.124.61.13:9200", #太乙模型 130b
        system_prompt='Your are a helpful assistant',
        temperature=0, 
        top_p=0.95, 
        max_tokens=5120, 
        presence_penalty=0, 
        frequency_penalty=1
        ):
    url += "/v1/chat/completions"
    payload = {
        "messages": [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                "role": "user",
                "content": "{}".format(user_prompt)
            }
        ],
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "presence_penalty" : presence_penalty,
        "frequency_penalty" : frequency_penalty,
    }
    headers = {"Content-Type": "application/json",}
    response = requests.post(url, json=payload, headers=headers)
    result = response.json() 
    # return dict(
    #     response=result["choices"][0]["message"]["content"],
    #     **result["usage"]
    # )
    return result["choices"][0]["message"]["content"]


## Utilities for logging
def is_wandb_logged_in():
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False
    
    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    return bool(auth)