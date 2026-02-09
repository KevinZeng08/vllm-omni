from typing import Literal, TypedDict

from vllm.inputs import EmbedsPrompt, SingletonPrompt, TextPrompt, TokensPrompt


class ParsedStrPrompt(TypedDict):
    type: Literal["str"]
    content: str


class ParsedTextPrompt(TypedDict):
    type: Literal["text"]
    content: TextPrompt


class ParsedTokensPrompt(TypedDict):
    type: Literal["tokens"]
    content: TokensPrompt


class ParsedEmbedsPrompt(TypedDict):
    type: Literal["embeds"]
    content: EmbedsPrompt


ParsedSingletonPrompt = ParsedStrPrompt | ParsedTextPrompt | ParsedTokensPrompt | ParsedEmbedsPrompt


def parse_singleton_prompt_omni(
    prompt: SingletonPrompt,
) -> ParsedSingletonPrompt:
    """Parse a singleton prompt into a typed parsed prompt.

    Handles omni-specific prompt types including tokens prompts with
    embeddings and additional information. Supports string, text,
    tokens, and embeddings prompts.

    Args:
        prompt: Singleton prompt to parse. Can be a string, TextPrompt,
            TokensPrompt (with optional prompt_embeds and additional_information),
            or EmbedsPrompt.

    Returns:
        ParsedSingletonPrompt containing the parsed prompt with type information

    Raises:
        TypeError: If the prompt type is not supported
    """
    if isinstance(prompt, str):
        return {"type": "str", "content": prompt}
    if isinstance(prompt, list):
        if not all(isinstance(token, int) for token in prompt):
            raise TypeError("Token prompt should be a list of integers")
        return {
            "type": "tokens",
            "content": TokensPrompt(prompt_token_ids=prompt),
        }
    if isinstance(prompt, dict):
        # Priority tokens: When both tokens and embeds exist, keep both and
        # follow the tokens path.
        if "prompt_token_ids" in prompt:
            return {"type": "tokens", "content": prompt}
        if "prompt_embeds" in prompt:
            return {"type": "embeds", "content": prompt}
        if "prompt" in prompt:
            return {"type": "text", "content": prompt}
    raise TypeError("inputs must be a string, list of token IDs, TextPrompt, TokensPrompt, or EmbedsPrompt")
