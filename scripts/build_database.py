from __future__ import annotations

import argparse
from pathlib import Path

from src.chunking import dataframe_to_documents
from src.embeddings import EmbeddingModel
from src.parser import parse_ebm_xml_to_dataframe
from src.vector_store import EbmVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local FAISS database from EBM XML.")
    parser.add_argument("--xml", required=True, help="Path to the official EBM XML.")
    parser.add_argument("--store", required=True, help="Output directory for the FAISS store.")
    parser.add_argument("--model", default=None, help="Optional sentence-transformers model name.")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    store_dir = Path(args.store)
    embedding_model = EmbeddingModel(args.model) if args.model else EmbeddingModel()

    df = parse_ebm_xml_to_dataframe(str(xml_path))
    documents = dataframe_to_documents(df)
    store, embeddings = EbmVectorStore.build(documents, embedding_model=embedding_model)
    store.save(store_dir, embeddings=embeddings)


if __name__ == "__main__":
    main()

