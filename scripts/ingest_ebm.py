from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import dataframe_to_search_corpus
from src.parser import filter_df_by_fachgruppe, parse_ebm_xml_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the EBM XML into structured artifacts.")
    parser.add_argument("--xml", required=True, help="Path to the official EBM XML.")
    parser.add_argument("--output", required=True, help="Directory for generated artifacts.")
    parser.add_argument("--fachgruppe-filter", action="store_true", help="Filter to Fachgruppe 001 only (for full EBM downloads).")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    
    print(f"Processing {len(df)} documents...")
    df.to_parquet(output_dir / "ebm.parquet", index=False)
    df.to_json(output_dir / "ebm.jsonl", orient="records", lines=True, force_ascii=False)

    corpus = dataframe_to_search_corpus(df)
    (output_dir / "ebm_documents.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in corpus),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
