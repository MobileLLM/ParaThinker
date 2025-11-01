import torch
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer

from custom_data_collator import ParallelCoTsDataCollator
from datasets import load_from_disk, Dataset, DatasetDict

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras import logging
from llamafactory.hparams import DataArguments, ModelArguments, FinetuningArguments, GeneratingArguments, get_train_args
from llamafactory.model.loader import load_model, load_tokenizer
from custom_run_sft import custom_run_sft
from transformers import TrainingArguments

logger = logging.get_logger(__name__)

def tokenize_strings(string_list: List[str], tokenizer: AutoTokenizer) -> List[List[int]]:
    tokenized_strings = []
    for string in string_list:
        # Tokenize and get input_ids (token IDs)
        token_ids = tokenizer.encode(string, add_special_tokens=False)
        tokenized_strings.append(token_ids)
    
    return tokenized_strings

def load_openr1_math_dataset(
    dataset_path: str,
    tokenizer,
    add_special_tokens: bool = True,
) -> Dict[str, Dataset]:
    """Load custom processed OpenR1-Math dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
        
    # Add special tokens for parallel CoTs reasoning
    if add_special_tokens:
        special_tokens = [
            "<think1>", "</think1>", "<think2>", "</think2>", "<think3>", "</think3>",
            "<think4>", "</think4>", "<think5>", "</think5>", "<think6>", "</think6>",
            "<think7>", "</think7>", "<think8>", "</think8>", "<summary>", "</summary>",
            "<vllm_pad>"
        ]
        
        tokens_to_add = []
        for token in special_tokens:
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                tokens_to_add.append(token)
                
        if tokens_to_add:
            logger.info(f"Adding {len(tokens_to_add)} special tokens to the tokenizer")
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            
    # Process the dataset structure
    if isinstance(dataset, Dataset):  # If it's a single dataset
        dataset.shuffle(seed=42)
        dataset_dict = {
            "train_dataset": dataset,
        }
    else:  # If it's a DatasetDict with splits
        dataset_dict = {
            "train_dataset": dataset["train"] if "train" in dataset else dataset,
            "eval_dataset": dataset["validation"] if "validation" in dataset else None
        }
    
    for split, ds in dataset_dict.items():
        if ds is not None:
            logger.info(f"Split: {split}, Size: {len(ds)}")
            
    return dataset_dict


def main():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    model_args.resize_vocab = True
    finetuning_args.pure_bf16 = True
    finetuning_args.plot_loss = True
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    
    # Load custom dataset instead of using get_dataset
    dataset = load_openr1_math_dataset(
        dataset_path=data_args.dataset[0],
        tokenizer=tokenizer,
        add_special_tokens=True,
    )
    
    # Get model for training
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # Determine if we should use flash attention
    attn_implementation = getattr(model.config, "attn_implementation", None)
    if attn_implementation is None:
        attn_implementation = getattr(model.config, "_attn_implementation", "eager")
    
    # Create custom data collator with appropriate parameters
    data_collator = ParallelCoTsDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        block_diag_attn=True,
        max_length=model_args.model_max_length,
        compute_dtype=torch.bfloat16,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    
    debug_tokenizer = True
    if debug_tokenizer:
        strings = [
            "<think1>", "</think1>", "<think2>", "</think2>", "<think3>", "</think3>",
            "<think4>", "</think4>", "<think5>", "</think5>", "<think6>", "</think6>",
            "<think7>", "</think7>", "<think8>", "</think8>", "<summary>", "</summary>",
            "<vllm_pad>"
        ]
        print(f"Special Token IDs: {tokenize_strings(strings, tokenizer)}")
    
    # Run the SFT training with custom data collator
    custom_run_sft(
        model_args=model_args, 
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=None,
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset,
        data_collator=data_collator
    )
    
if __name__ == "__main__":
    main()
