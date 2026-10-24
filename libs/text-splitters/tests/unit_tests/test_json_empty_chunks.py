"""JSON chunk boundaries must not manufacture empty documents."""

import json
from typing import Any

import pytest

from langchain_text_splitters.json import RecursiveJsonSplitter


@pytest.mark.parametrize("min_chunk_size", [0, 1, 2])
@pytest.mark.parametrize(
    "data",
    [
        {"text": "x" * 30},
        {"outer": {"inner": {"text": "x" * 30}}},
        {"outer": {"empty": {}, "text": "x" * 30}},
    ],
)
def test_no_empty_chunks_before_oversized_value(
    min_chunk_size: int, data: dict[str, Any]
) -> None:
    splitter = RecursiveJsonSplitter(max_chunk_size=20, min_chunk_size=min_chunk_size)
    chunks = splitter.split_json(data)
    assert chunks
    assert all(chunks)
    documents = splitter.create_documents([data], metadatas=[{"source": "input"}])
    assert [json.loads(doc.page_content) for doc in documents] == chunks
    assert all(doc.metadata == {"source": "input"} for doc in documents)


def test_empty_dict_values_are_preserved() -> None:
    splitter = RecursiveJsonSplitter(max_chunk_size=5, min_chunk_size=0)
    assert splitter.split_json({"a": {}, "b": {}}) == [{"a": {}}, {"b": {}}]


def test_empty_input_still_has_no_chunks() -> None:
    assert RecursiveJsonSplitter(min_chunk_size=0).split_json({}) == []


def test_complete_values_are_not_lost_at_chunk_boundaries() -> None:
    data = {str(i): "x" * 30 for i in range(3)}
    splitter = RecursiveJsonSplitter(max_chunk_size=20, min_chunk_size=0)
    assert splitter.split_json(data) == [{key: value} for key, value in data.items()]
