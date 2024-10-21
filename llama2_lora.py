import os
import time
import fire
import torch
import wandb
import numpy as np
from typing import List
from datasets import load_dataset

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training
)
from transformers import ( # noqa: F402
    LlamaForCausalLM, 
    LlamaTokenizer, 
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback
)
        

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
        ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:{output}"
        )
        }
REMOVE_COLUMNS = ["instruction", "output", "input"]

def generate_prompt(data_point):  
    if data_point["input"]:
        return PROMPT_DICT["prompt_input"].format_map(data_point)
    else:
        return PROMPT_DICT["prompt_no_input"].format_map(data_point)

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "data/sample.json",
        output_dir: str = "./output",
        use_cache: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        lr_scheduler_type: str = "linear",
        warmup_steps: int = 100,
        optim:str = "adamw_torch",
        fp16: bool = False, # will cause error if it is not the opposite of 'load_in_8bit' (a trade-off between memory and speed, at least when training on V100)
        load_in_8bit: bool = True,
        cutoff_len: int = 256,
        early_stopping_patience=1e9, # set to a large number to disable
        use_gradient_checkpointing: bool = True,
        val_set_size: int = 2000,
        save_strategy: str = "steps",
        save_total_limit: int = 2,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj"], # ["query_key_value",],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = True,  # faster, but produces an odd training loss curve
        # logging params
        logging_strategy: str = "steps",
        logging_steps: int = 10,
        report_to: str = "tensorboard",        
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "true",
        # resume training
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Finetuning model with params:\n"
            "==============================\n"

            "model/data params:\n"
            "------------------\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"use_cache: {use_cache}\n"

            "training hyperparams:\n"
            "---------------------\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"optim: {optim}\n"
            f"fp16: {fp16}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"early_stopping_patience: {early_stopping_patience}\n"
            f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
            f"val_set_size: {val_set_size}\n"
            f"save_strategy: {save_strategy}\n"
            f"save_total_limit: {save_total_limit}\n"
            f"eval_step: {eval_step}\n"
            f"save_step: {save_step}\n"

            "lora hyperparams:\n"
            "-----------------\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"

            "llm hyperparams:\n"
            "----------------\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"

            "wandb params:\n"
            "-------------\n"
            f"logging_strategy: {logging_strategy}\n"
            f"logging_steps: {logging_steps}\n"
            f"report_to: {report_to}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        )

    ### DDP initialization
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/LLaMA-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    ### wandb logging initialization
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and 
        len(os.environ["WANDB_PROJECT"]) > 0
    )
    if use_wandb:
        report_to = "wandb"
        if len(wandb_run_name)==0:
            wandb_run_name = time.strftime(
                '%Y-%m-%d-%H:%M:%S %p %Z', 
                time.gmtime(time.time())
                )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            wandb.init(project=wandb_project, name=wandb_run_name)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_WATCH"] = wandb_watch
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    ### initialization of model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit, 
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # if load_in_8bit:
    #     model.half()

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            return_token_type_ids=False
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        # if train_on_inputs=False, only ouput will remain in "labels"
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                                                IGNORE_INDEX
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    ### model preparation
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()  
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    ### data preparation
    if data_path.endswith(".json"):  
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    tokenized_data = data.shuffle().map(
        generate_and_tokenize_prompt,
        remove_columns=REMOVE_COLUMNS,
        )
    if val_set_size > 0:
        split_data=  tokenized_data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42,
            )

    ### training preparation
    training_args = TrainingArguments(
        # training
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        fp16=fp16,
        # logging
        logging_dir=output_dir,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        report_to=report_to,
        run_name=wandb_run_name,
        # evaluating and saving
        evaluation_strategy=save_strategy if val_set_size > 0 else "no", # align both strategies for the purpose of loading best model at the end
        save_strategy=save_strategy, 
        eval_steps=eval_step if val_set_size > 0 else None,
        save_steps=save_step if save_strategy == "steps" else None,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True, # the default metric is loss
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        )
    
    class PerplexityCallback(TrainerCallback):
        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            eval_perplexity = np.exp(state.log_history[-1]['eval_loss'])
            # state.log_history[-1].update({"eval_perplexity": eval_perplexity})
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                wandb.log({"eval_perplexity": eval_perplexity}, step=state.global_step)

    trainer = Trainer(
        model=model,
        train_dataset=split_data["train"] if val_set_size > 0 else tokenized_data,
        eval_dataset=split_data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=data_collator,
        callbacks=[
            PerplexityCallback,
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ]
    )
    model.config.use_cache = use_cache

    # with torch.autocast("cuda"): 
    # adding the above context will cause loss stays constant (0 in most cases) for some reason
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)