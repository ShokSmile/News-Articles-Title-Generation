from transformers import AutoModelForCausalLM, AutoConfig
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
    model_type: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    model_max_length: int = 4096,
) -> AutoModelForCausalLM:
    """
    :param path_to_model_weights: path to model weights
    :param model_type:
    1. "alpaca";
    2. "mpt";
    3. "xgen";
    4. "llama"
    :param load_in_8bit: 8 bit quantization
    :param load_in_4bit: 4 bit quantization
    :param model_max_length: model max length
    :return: AutoModelForCasualLM
    """
    assert model_type in list(["alpaca", "mpt", "xgen", "llama"]), print("Check ")

    if model_type != "alpaca" and model_type != "llama":
        config = AutoConfig.from_pretrained(
            path_to_model_weights,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

        config.max_seq_len = model_max_length
        model = AutoModelForCausalLM.from_pretrained(
            path_to_model_weights, config=config, trust_remote_code=True
        )

    elif model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(
            path_to_model_weights,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            path_to_model_weights,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    model.use_cache = False

    return model


def load_peft_model(
    path_to_weights: str,
    model_type: str,
    LoRA: bool = True,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    model_max_length: int = 4096,
) -> PeftModel:
    assert model_type in list(["alpaca", "mpt", "xgen", "llama"]), print("Check ")

    model = load_ordinary_model(
        path_to_model_weights=path_to_weights,
        model_type=model_type,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        model_max_length=model_max_length,
    )

    target_modules = {
        "mpt": ["Wqkv"],
        "alpaca": ["q_proj", "v_proj"],
        "llama": ["q_proj", "v_proj"],
        "xgen": ["q_proj", "v_proj"],
    }

    if LoRA:
        # lora config
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules[model_type],
            # ["q_proj", "v_proj"] -- xgen , ["Wqkv"] -- mpt7, ["q", "v"] - t5
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # CAUSAL_LM - xgen and mpt, SEQ_2_SEQ_LM- t5
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        model.use_cache = False

    return model


def load_pretrained_peft_model(
    path_to_model_weights: str,
    path_to_peft_weights: str,
    model_type: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    model_max_length=4096,
) -> PeftModel:
    assert model_type in list(["alpaca", "mpt", "xgen", "llama"]), print(
        "Check model type!"
    )

    model = PeftModel.from_pretrained(
        load_ordinary_model(
            path_to_model_weights=path_to_model_weights,
            model_type=model_type,
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
