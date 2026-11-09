
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(model_name):
    """Load the pretrained tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no padding or EOS token.")

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_name, tokenizer):
    """Load the pretrained causal language model."""

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def load_model_and_tokenizer(model_name):
    """Load both model and tokenizer."""

    tokenizer = load_tokenizer(model_name)

    model = load_model(model_name, tokenizer)

    return model, tokenizer

