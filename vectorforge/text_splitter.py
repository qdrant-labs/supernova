import tiktoken


def split_text(text: str, max_tokens: int = 8192, model: str = "text-embedding-3-small") -> list[str]:
    """
    Split text into chunks that fit within the token limit.
    Returns a list of text pieces. If the text fits, returns [text].
    """
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text, allowed_special="all")

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoder.decode(chunk_tokens))

    return chunks