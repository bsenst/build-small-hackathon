from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking import dataframe_to_search_corpus
from src.parser import parse_ebm_xml_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the EBM XML into structured artifacts.")
    parser.add_argument("--xml", required=True, help="Path to the official EBM XML.")
    parser.add_argument("--output", required=True, help="Directory for generated artifacts.")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_ebm_xml_to_dataframe(str(xml_path))
    df.to_parquet(output_dir / "ebm.parquet", index=False)
    df.to_json(output_dir / "ebm.jsonl", orient="records", lines=True, force_ascii=False)

    corpus = dataframe_to_search_corpus(df)
    (output_dir / "ebm_documents.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in corpus),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
