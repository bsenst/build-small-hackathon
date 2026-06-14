from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import dataframe_to_documents
from src.embeddings import EmbeddingModel
from src.parser import filter_df_by_fachgruppe, parse_ebm_xml_to_dataframe
from src.vector_store import EbmVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local FAISS database from EBM XML.")
    parser.add_argument("--xml", required=True, help="Path to the official EBM XML.")
    parser.add_argument("--store", required=True, help="Output directory for the FAISS store.")
    parser.add_argument("--model", default=None, help="Optional sentence-transformers model name.")
    parser.add_argument("--fachgruppe-filter", action="store_true", help="Filter to Fachgruppe 001 only (for full EBM downloads).")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    store_dir = Path(args.store)
    embedding_model = EmbeddingModel(args.model) if args.model else EmbeddingModel()

    df = parse_ebm_xml_to_dataframe(str(xml_path))
    
    # Apply Fachgruppe filter only if requested
    if args.fachgruppe_filter:
        print("Applying Fachgruppe 001 filter...")
        df = filter_df_by_fachgruppe(df)
        if df.empty:
            raise ValueError(
                "No Fachgruppe 001 documents found in the provided XML. "
                "Please check the XML file or remove the --fachgruppe-filter flag."
            )
    
    if df.empty:
        raise ValueError(
            "No documents found in the provided XML. "
            "Please provide a valid KBV EBM XML file."
        )
    
    print(f"Building FAISS store from {len(df)} documents...")
    documents = dataframe_to_documents(df)
    store, embeddings = EbmVectorStore.build(documents, embedding_model=embedding_model)
    store.save(store_dir, embeddings=embeddings)


if __name__ == "__main__":
    main()

