from transformers import AutoModelForCausalLM, AutoConfig, T5ForConditionalGeneration
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
    PeftConfig,
)


def load_ordinary_model(
    path_to_model_weights: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    model_max_length: int = 1024,
) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(path_to_model_weights, 
                                                       load_in_4bit=load_in_4bit, 
                                                       load_in_8bit=load_in_8bit, 
                                                       model_max_length=model_max_length)

    return model


def load_peft_model(
    path_to_weights: str,
    LoRA: bool = True,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    model_max_length: int = 1024,
) -> PeftModel:
    

    model = load_ordinary_model(
        path_to_model_weights=path_to_weights,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        model_max_length=model_max_length,
    )

    if LoRA:
        # lora config
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM  # CAUSAL_LM - gpt-like, SEQ_2_SEQ_LM- t5
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        model.use_cache = False

    return model


def load_pretrained_peft_model(
    path_to_model_weights: str,
    path_to_peft_weights: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    model_max_length=1024,
) -> PeftModel:
    

    model = PeftModel.from_pretrained(
        load_ordinary_model(
            path_to_model_weights=path_to_model_weights,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            model_max_length=model_max_length,
        ),
        path_to_peft_weights,
        is_trainable=True,
    )

    model.print_trainable_parameters()
    model.enable_input_require_grads()

    return model
