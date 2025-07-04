import requests
import xml.etree.ElementTree as ET
import os


def search_pmcs(query, start_date, max_results=5):
    """
    Suche PMC-Artikel zu einem bestimmten Thema und Datum.
    Gibt eine Liste von PMCIDs zurück.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{query} AND ({start_date}[PDAT] : 3000[PDAT])",
        "retmode": "xml",
        "retmax": max_results,
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    pmcids = [id_elem.text for id_elem in root.findall(".//Id")]
    return pmcids


def fetch_pmc_fulltext(pmcid):
    """
    Holt den Volltext eines PMC-Artikels im XML-Format.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    return response.content if response.status_code == 200 else None


def save_xml(pmcid, xml_data, save_dir="xmls"):
    """
    Speichert den XML-Inhalt in eine Datei.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"PMC{pmcid}.xml")
    with open(path, "wb") as f:
        f.write(xml_data)
    print(f"✅ XML gespeichert: {path}")


def pipeline(query, start_date, max_results=5):
    """
    Führt den kompletten Prozess aus: Suche → XML holen → Speichern.
    """
    print(f"🔍 Suche nach PMC-Artikeln zu: '{query}' ab {start_date}")
    pmcids = search_pmcs(query, start_date, max_results)
    if not pmcids:
        print("❌ Keine Ergebnisse gefunden.")
        return

    for pmcid in pmcids:
        print(f"\n📄 Lade PMC{pmcid}")
        xml = fetch_pmc_fulltext(pmcid)
        if xml:
            save_xml(pmcid, xml)
        else:
            print(f"⚠️  Fehler beim Abrufen von PMC{pmcid}")


if __name__ == "__main__":
    searchterm = '("neuroendocrine tumour"[Title/Abstract] OR "NET"[Title/Abstract])'
    # ENET: 27April2023
    # ESMO:  6 April 2020
    start_date = "2023/04/27"
    max_results = 5

    pipeline(searchterm, start_date, max_results)
