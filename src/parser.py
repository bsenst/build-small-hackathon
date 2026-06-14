from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd


NAMESPACES = {"ehd": "urn:ehd/001", "go": "urn:ehd/go/001"}


def get_text_content(elem: ET.Element | None) -> str | None:
    """Extract all text recursively from an XML element."""
    if elem is None:
        return None

    text = " ".join(t.strip() for t in elem.itertext() if t.strip())
    return text if text else None


def parse_ebm_xml_to_dataframe(xml_path: str) -> pd.DataFrame:
    """Parse EBM XML into a pandas DataFrame with one row per GNR."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows: list[dict[str, Any]] = []

    for gnr in root.findall("./ehd:body/go:gnr_liste/go:gnr", namespaces=NAMESPACES):
        row: dict[str, Any] = {
            "code": gnr.get("V"),
            "use": gnr.get("USE"),
            "valid_from": gnr.get("VT"),
        }

        legende = gnr.find("./go:allgemein/go:legende", namespaces=NAMESPACES)
        if legende is not None:
            kurztext = legende.find("go:kurztext", namespaces=NAMESPACES)
            quittungstext = legende.find("go:quittungstext", namespaces=NAMESPACES)
            langtext = legende.find("go:langtext", namespaces=NAMESPACES)
            kap_bez = legende.find("go:kap_bez", namespaces=NAMESPACES)

            row["short_text"] = kurztext.get("V") if kurztext is not None else None
            row["receipt_text"] = quittungstext.get("V") if quittungstext is not None else None
            row["long_text"] = get_text_content(langtext)

            if kap_bez is not None:
                bereich = kap_bez.find("go:bereich", namespaces=NAMESPACES)
                kapitel = kap_bez.find("go:kapitel", namespaces=NAMESPACES)
                abschnitt = kap_bez.find("go:abschnitt", namespaces=NAMESPACES)
                row["chapter_code"] = kap_bez.get("V")
                row["chapter_name"] = kap_bez.get("DN")
                row["bereich"] = bereich.get("DN") if bereich is not None else None
                row["kapitel"] = kapitel.get("DN") if kapitel is not None else None
                row["abschnitt"] = abschnitt.get("DN") if abschnitt is not None else None

        row["service_period"] = (
            gnr.find("./go:allgemein/go:gueltigkeit/go:service_tmr", namespaces=NAMESPACES).get("V")
            if gnr.find("./go:allgemein/go:gueltigkeit/go:service_tmr", namespaces=NAMESPACES) is not None
            else None
        )
        row["effective_period"] = (
            gnr.find("./go:allgemein/go:gueltigkeit/go:effective_tmr", namespaces=NAMESPACES).get("V")
            if gnr.find("./go:allgemein/go:gueltigkeit/go:effective_tmr", namespaces=NAMESPACES) is not None
            else None
        )

        notes: list[str] = []
        for note in gnr.findall("./go:allgemein/go:anmerkungen_liste/go:anmerkung", namespaces=NAMESPACES):
            txt = get_text_content(note)
            if txt:
                notes.append(txt)
        row["notes"] = notes

        bewertung = gnr.find("./go:allgemein/go:bewertung_liste/go:bewertung", namespaces=NAMESPACES)
        if bewertung is not None:
            row["points"] = bewertung.get("V")
            row["unit"] = bewertung.get("U")
            lt = bewertung.find("go:leistung_typ", namespaces=NAMESPACES)
            row["leistung_typ"] = lt.get("V") if lt is not None else None

        fachgruppen: list[str] = []
        for fg in gnr.findall(".//go:fachgruppe_liste//go:fachgruppe", namespaces=NAMESPACES):
            value = fg.get("V")
            if value:
                fachgruppen.append(value)
        row["fachgruppen"] = fachgruppen

        exclusions: list[dict[str, str | None]] = []
        for ex in gnr.findall("./go:regel/go:ausschluss_liste/go:bezugsraum", namespaces=NAMESPACES):
            for ex_gnr in ex.findall("./go:gnr_liste/go:gnr", namespaces=NAMESPACES):
                exclusions.append(
                    {
                        "code": ex_gnr.get("V"),
                        "description": ex_gnr.get("DN"),
                    }
                )
        row["exclusions"] = exclusions

        gkv_types: list[str] = []
        for gkv in gnr.findall("./go:vdx/go:gkv_kontenart_liste/go:gkv_kontenart", namespaces=NAMESPACES):
            value = gkv.get("V")
            if value:
                gkv_types.append(value)
        row["gkv_account_types"] = gkv_types

        rows.append(row)

    return pd.DataFrame(rows)


def filter_df_by_fachgruppe(df: pd.DataFrame, fachgruppe: str = "001") -> pd.DataFrame:
    # Allow skipping the fachgruppe filter (useful for demo XML fallback)
    import os

    if os.environ.get("SKIP_FG_FILTER") == "1":
        return df.reset_index(drop=True)

    return df[df["fachgruppen"].apply(lambda x: isinstance(x, list) and fachgruppe in x)].reset_index(drop=True)

