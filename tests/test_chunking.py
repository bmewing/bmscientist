from bmscientist.chunking import TextChunker


def test_chunker_splits_long_text_with_overlap():
    text = " ".join(f"token{i}" for i in range(400))
    chunker = TextChunker(chunk_size=120, chunk_overlap=20)

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert chunks[0][-10:] in chunks[1]


def test_chunker_returns_empty_list_for_empty_text():
    assert TextChunker().chunk_text("") == []

