from functools import reduce
from modules import shared
from modules import extra_networks
from modules import prompt_parser
from modules import sd_hijack

# taken from modules/ui.py 
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/ui.py
def get_token_count(text, steps, is_positive: bool = True, return_tokens = False):
    """ Get token count and max length for a given prompt text. If return_tokens is True, return the tokens as well. 
    Returns:
        token_count: int - The total number of tokens in the prompt text
        max_length: int - The maximum length of the prompt text
    """
    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]

    token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return token_count, max_length


def tokenize_prompt(text):
    """ Tokenize the given prompt text using the current clip model. 
    Arguments:
        text: str - The prompt text to tokenize

    Returns:
        tokens: list[int] - If return_tokens is True, return the tokenized prompt as well
    
    """
    if isinstance(text, str):
        prompts = [text]

    clip = getattr(sd_hijack.model_hijack, 'clip', None)
    if clip is None:
        return None, None
    batch_chunks, token_count = clip.process_texts(prompts)
    return batch_chunks, token_count


def decode_tokenized_prompt(tokens):
    """ Decode the given tokenized prompt using the current clip model. 
    Arguments:
        tokens: list[int] - The tokenized prompt to decode
    Returns:
        a list of tuples containing the token index, token, and decoded token
    
    """
    clip = getattr(sd_hijack.model_hijack, 'clip', None)
    if clip is None:
        return None
    decoded_prompt = [
        [token_idx, token, clip.tokenizer.decoder[token]] for token_idx, token in enumerate(tokens)
    ]
    return decoded_prompt