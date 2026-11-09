from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(model_name):
    """Load the model tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_name):
    """Load the pretrained language model."""

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model


def load_model_and_tokenizer(model_name):
    """Load both model and tokenizer."""

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    return model, tokenizer
