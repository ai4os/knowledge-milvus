"""
Utils for text chunking.
In the future we can maybe use a dedicated library.
"""


def basic_chunking(text, chunk_size=5000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def semantic_chunking(text, chunk_size=5000, overlap=500):
    """
    [GEMINI GENERATED!]

    Splits text into chunks respecting semantic boundaries (paragraphs, sentences).

    Priority of splitting:
    1. Double newlines (\n\n) - Paragraphs
    2. Single newlines (\n)   - Lines
    3. Sentence endings (. ! ?)
    4. Space ( )              - Words
    5. Hard limit             - Characters (fallback)
    """

    # We want to build chunks up to chunk_size
    # We will accumulate smaller pieces until we hit the limit

    # Step 1: Split into atomic pieces (paragraphs or sentences)
    # This regex splits by double newlines, or sentence endings if no newlines
    # It keeps the delimiters.
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    # Recursive splitting helper
    def recursive_split(text, separators):
        final_chunks = []
        if not separators:
            return [
                text
            ]  # Fallback: return as is (very unlikely to hit this with empty string sep)

        sep = separators[0]
        # Regex to split and keep separator attached to the previous chunk
        # If sep is special char, escape it. If sep is "", just return list of chars
        if sep == "":
            return list(text)

        try:
            # Split by separator
            splits = text.split(sep)
            # Re-attach separators to the end of the split (except the last one)
            splits = [s + sep for s in splits[:-1]] + [splits[-1]]
        except Exception:
            return [text]

        new_chunks = []
        current_chunk = ""

        for split in splits:
            # If a single split is already too big, recurse down to next separator
            if len(split) > chunk_size:
                if current_chunk:
                    new_chunks.append(current_chunk)
                    current_chunk = ""
                # Recurse
                sub_chunks = recursive_split(split, separators[1:])
                new_chunks.extend(sub_chunks)
            elif len(current_chunk) + len(split) > chunk_size:
                # If adding this split exceeds size, save current and start new
                new_chunks.append(current_chunk)
                current_chunk = split
            else:
                current_chunk += split

        if current_chunk:
            new_chunks.append(current_chunk)

        return new_chunks

    # Get the initial splits
    raw_chunks = recursive_split(text, separators)

    # Step 2: Post-process to handle overlap and strict sizing
    # The recursive splitter groups things well, but we need to enforce the overlap
    # manually if we want a "sliding window" of semantic chunks.

    final_chunks = []
    current_chunk = ""

    # This is a simplified sliding window approach for semantic blocks
    # A true sliding window with overlap on variable-sized semantic blocks is complex.
    # Here is a robust approximation:

    # We iterate through the raw semantic blocks
    for i, block in enumerate(raw_chunks):
        if len(current_chunk) + len(block) <= chunk_size:
            current_chunk += block
        else:
            final_chunks.append(current_chunk.strip())

            # Create overlap: backtrack to include previous blocks until overlap size is met
            overlap_buffer = ""
            backtrack_idx = i - 1
            while backtrack_idx >= 0 and len(overlap_buffer) < overlap:
                overlap_buffer = raw_chunks[backtrack_idx] + overlap_buffer
                backtrack_idx -= 1

            current_chunk = overlap_buffer + block

    if current_chunk:
        final_chunks.append(current_chunk.strip())

    return final_chunks
