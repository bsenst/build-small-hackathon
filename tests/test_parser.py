from __future__ import annotations

from pathlib import Path

from src.parser import parse_ebm_xml_to_dataframe


def test_parse_ebm_xml_to_dataframe(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<ehd:root xmlns:ehd="urn:ehd/001" xmlns:go="urn:ehd/go/001">
  <ehd:body>
    <go:gnr_liste>
      <go:gnr V="01100" USE="1" VT="20240101">
        <go:allgemein>
          <go:legende>
            <go:kurztext V="Unvorhergesehene Inanspruchnahme I"/>
            <go:quittungstext V="QText"/>
            <go:langtext>Langtext der Leistung.</go:langtext>
            <go:kap_bez V="01" DN="Kapitel 01">
              <go:bereich DN="Bereich A"/>
              <go:kapitel DN="Kapitelname"/>
              <go:abschnitt DN="Abschnittname"/>
            </go:kap_bez>
          </go:legende>
          <go:anmerkungen_liste>
            <go:anmerkung>Anmerkung 1</go:anmerkung>
          </go:anmerkungen_liste>
          <go:bewertung_liste>
            <go:bewertung V="196" U="PUNKTE">
              <go:leistung_typ V="GKV"/>
            </go:bewertung>
          </go:bewertung_liste>
        </go:allgemein>
        <go:regel>
          <go:ausschluss_liste>
            <go:bezugsraum>
              <go:gnr_liste>
                <go:gnr V="01101" DN="Ausschluss eins"/>
              </go:gnr_liste>
            </go:bezugsraum>
          </go:ausschluss_liste>
        </go:regel>
        <go:vdx>
          <go:gkv_kontenart_liste>
            <go:gkv_kontenart V="A"/>
          </go:gkv_kontenart_liste>
        </go:vdx>
      </go:gnr>
    </go:gnr_liste>
  </ehd:body>
</ehd:root>
""",
        encoding="utf-8",
    )

    df = parse_ebm_xml_to_dataframe(str(xml))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["code"] == "01100"
    assert row["short_text"] == "Unvorhergesehene Inanspruchnahme I"
    assert row["points"] == "196"
    assert row["notes"] == ["Anmerkung 1"]
    assert row["fachgruppen"] == []
    assert row["exclusions"][0]["code"] == "01101"

