from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras import logging
from llamafactory.extras.ploting import plot_loss

def custom_run_sft(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
    callbacks=None,
    model=None,
    tokenizer=None,
    data_collator=None,
    dataset_dict=None
):
    """Custom version of run_sft that accepts pre-loaded components"""
    
    tokenizer_module = {"tokenizer": tokenizer}
    tokenizer = tokenizer_module["tokenizer"]
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)
    
    # Use provided dataset or get standard dataset
    dataset_module = dataset_dict
    
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        processor=tokenizer_module.get("processor", None),
        **dataset_module,
        **tokenizer_module
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
            