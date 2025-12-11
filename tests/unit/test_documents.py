from __future__ import annotations

from langchain_core.documents import Document

from app import documents


def test_load_documents_falls_back_to_general_knowledge(monkeypatch, tmp_path):
    def fake_loader() -> list[Document]:
        return [Document(page_content="Question: Q?\nAnswer: A!", metadata={"source": "hf"})]

    monkeypatch.setattr(documents, "load_general_knowledge_documents", fake_loader)

    loaded = documents.load_documents(data_path=str(tmp_path))

    assert loaded
    assert loaded[0].page_content == "Question: Q?\nAnswer: A!"
    assert loaded[0].metadata["source"] == "hf"
