"""
Generate per-chapter bash scripts to download cited papers from open-access sources.

Parses refs.bib, maps citation keys to chapters via tex files, extracts arXiv IDs
and URLs, queries Semantic Scholar for open-access PDFs, and produces one download
script per chapter in scripts/download_ch*.sh.

Usage:
    python scripts/generate_download_scripts.py

Requires internet access for Semantic Scholar API lookups. Results are cached
in scripts/.paper_cache.json to avoid repeated API calls.
"""

import re
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB_PATH = ROOT / "docs" / "refs.bib"
CACHE_PATH = ROOT / "scripts" / ".paper_cache.json"

# Chapter directories and their tex files
CHAPTERS = {
    "ch00_introduction": ["tex/intro.tex", "tex/abstract.tex"],
    "ch01_history": ["tex/history.tex"],
    "ch02_planning_learning": ["tex/planning_learning.tex", "tex/planning_learning_alt.tex"],
    "ch03_rl_structural_est": ["tex/rl_in_se.tex"],
    "ch04_inverse_rl": ["tex/irl.tex"],
    "ch05_rl_in_games": ["tex/marl.tex"],
    "ch06_bandits": ["tex/se_in_rl_full.tex"],
    "ch07_applications": ["tex/applications.tex"],
    "ch08_rlhf": ["tex/rlhf.tex"],
    "ch09_conclusion": ["tex/conclusion.tex"],
}


def parse_bib(bib_path):
    """Parse a .bib file into a dict of {key: {field: value}}."""
    entries = {}
    text = bib_path.read_text(encoding="utf-8", errors="replace")

    # Split into entries
    entry_pattern = re.compile(
        r"@(\w+)\s*\{([^,]+),\s*(.*?)\n\}",
        re.DOTALL,
    )
    for match in entry_pattern.finditer(text):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        body = match.group(3)

        fields = {"_type": entry_type}
        # Parse fields: name = {value} or name = "value"
        field_pattern = re.compile(
            r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}|(\w+)\s*=\s*\"([^\"]*)\""
        )
        for fm in field_pattern.finditer(body):
            if fm.group(1):
                fields[fm.group(1).lower()] = fm.group(2).strip()
            elif fm.group(3):
                fields[fm.group(3).lower()] = fm.group(4).strip()

        entries[key] = fields

    return entries


def extract_arxiv_id(entry):
    """Try to extract an arXiv ID from journal, note, url, or eprint fields."""
    for field in ["journal", "note", "url", "eprint", "archiveprefix"]:
        val = entry.get(field, "")
        # Match patterns like arXiv:2209.15174 or arxiv.org/abs/2209.15174
        m = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)", val, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", val, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def extract_url(entry):
    """Extract a direct URL if present."""
    return entry.get("url", "")


def extract_doi(entry):
    """Extract DOI if present."""
    return entry.get("doi", "")


def extract_citations_from_tex(tex_path):
    """Extract all citation keys from a tex file."""
    if not tex_path.exists():
        return set()
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # Match \cite{...}, \citet{...}, \citep{...}, \citeauthor{...}, etc.
    keys = set()
    for m in re.finditer(r"\\cite[tp]?\*?\{([^}]+)\}", text):
        for k in m.group(1).split(","):
            keys.add(k.strip())
    return keys


def sanitize_filename(title, key):
    """Create a filesystem-safe filename from title or key."""
    if title:
        name = re.sub(r"[^a-zA-Z0-9\s-]", "", title)
        name = re.sub(r"\s+", "_", name.strip())
        name = name[:60].rstrip("_")
        return name + ".pdf"
    return key + ".pdf"


# Known arXiv IDs for papers in this survey. Manually curated for reliability.
KNOWN_ARXIV_IDS = {
    # Foundational RL
    "Mnih2015": "1312.5602",
    "mnih2015": "1312.5602",
    "Schulman2015": "1502.05477",
    "Schulman2017": "1707.06347",
    "Haarnoja2018": "1801.01290",
    "Kakade2001": "cs/0106005",
    "Sutton1999": "cs/9905014",
    "agarwal2020": "1907.04543",
    "moerland2022unifying": "2006.15009",

    # IRL / Imitation Learning
    "Abbeel2004": "cs/0409043",  # Available at ML proceedings, not arXiv
    "NgRussell2000": None,  # ICML 2000, no arXiv
    "Ziebart2008": "0710.4922",
    "Ho2016": "1606.03476",
    "Finn2016": "1603.00448",
    "Fu2018": "1710.11248",
    "Kostrikov2018": "1809.02925",
    "Boularias2011": "1506.07023",
    "Baram2017": "1809.02064",
    "Wulfmeier2015": "1507.04888",
    "Ramachandran2007": None,  # IJCAI 2007
    "Ratliff2006": None,  # ICML 2006

    # IRL applied / economics
    "Ermon2015": None,  # AAAI 2015
    "Chan2019": None,  # QJE
    "Bronner2023": None,  # Transportation Research
    "Mai2015": None,  # Journal
    "Yancey2022": None,
    "Zeng2022": None,
    "Sharma2018": "1805.07687",
    "kim2021irl": "2106.03788",
    "cao2021irl": "2106.03788",  # May be different paper
    "cao2021": "2106.03788",
    "rolland2022": "2202.09529",
    "shehab2024": "2401.03608",
    "obriant2024": None,

    # MARL / Games
    "Brown2019": "1811.00164",
    "brown2019": "1811.00164",
    "brown2019_deepcfr": "1811.00164",
    "brown2019deepcfr_icml": "1811.00164",
    "HeinrichSilver2016": "1603.01121",
    "heinrich2016deep": "1603.01121",
    "Lanctot2017": "1711.00832",
    "Mazumdar2019": "1906.01217",
    "Letcher2019": "1905.04926",
    "deepmind2022": "2206.15378",
    "moravcik2017": "1701.01724",
    "bowlingveloso2002": None,  # AIJ 2002
    "littman1994markov": None,  # ML 1994
    "littman1994_minimaxq": None,
    "hu2003nash": None,  # JMLR 2003
    "hu2003_nashq": None,
    "zinkevich2008": "1811.00164",  # Actually NIPS 2007
    "zinkevich2008regret": None,
    "zinkevich2008_cfr": None,
    "tuyls2003": None,
    "tuyls2003selection": None,
    "borgers1997learning": None,  # JET
    "singh2000": None,  # ML journal
    "singh2000_iga_convergence": None,

    # Deep RL / Policy Gradient
    "Silver2016": "1607.01491",  # Actually Nature but also on arXiv
    "Silver2017": "1706.01427",
    "Silver2018": "1712.01815",
    "Schrittwieser2020": "1911.08265",
    "OpenAIFive2019": "1912.06680",
    "Vinyals2019": "1904.02633",

    # Structural estimation + RL
    "adusumillieckardt2022": "2209.15174",
    "AtashbarShi2023": None,  # JEDC
    "atashbar2023": None,  # IMF WP
    "HinterlangTaenzer2024": None,  # JME
    "FernandezVillaverdeNunoPerla2024": None,  # JoE
    "MaliarMaliarWinant2021": "2004.05668",
    "Covarrubias2022": None,  # NBER WP
    "curry2022": "2205.07165",
    "Hollenbeck2019": None,  # QME
    "BreroEtAl2021": "2101.07824",
    "GrafEtAl2024": None,
    "HuYang2025": None,
    "LomysMagnolfi2024": None,
    "RavindranathEtAl2024": None,

    # Bandits
    "auer2002": None,  # ML Journal 2002
    "Auer2002": None,
    "Badanidiyuru2013": "1305.2545",
    "bennettkallus2023": "2306.12351",
    "CesaBianchi2015": None,  # IEEE TIT
    "Flajolet2017": "1712.09966",
    "Goyal2022": None,  # OR
    "Guo2023": None,  # MS
    "Misra2019": "1902.04229",
    "Mueller2019": None,
    "Nuara2018": "1811.09476",
    "Sankararaman2018": None,
    "Xu2021": None,
    "liao2024": None,  # ICLR 2024
    "Akcay2022": None,  # OR

    # RLHF
    "Ouyang2022": "2203.02155",
    "ouyang2022training": "2203.02155",
    "DeepSeekR1_2025": "2501.12948",
    "Bai2022": "2212.08073",
    "Glaese2022": "2209.14375",
    "stiennon2020learning": "2009.01325",
    "christiano2017": "1706.03741",
    "christiano:2017": "1706.03741",
    "ziegler2019fine": "1909.08593",
    "rafailov2023direct": "2305.18290",
    "korbak2022rl": "2210.01241",
    "yuan2024self": "2401.10020",

    # Economics classics (no arXiv)
    "Rust1987": None,
    "HotzMiller1993": None,
    "Miller1984": None,
    "Wolpin1984": None,
    "BerryLevinsohnPakes1995": None,
    "McFadden1974": None,
    "mcfadden:1973": None,
    "Haavelmo1944": None,
    "Heckman1979": None,
    "Bellman1957": None,
    "howard1960": None,
    "HoodKoopmans1953": None,
    "AckerbergCavesFrazer2015": None,
    "AguirregabiriaMira2007": None,
    "BajariBenkardLevin2007": None,
    "DubeFoxSu2012": None,
    "CilibertoTamer2009": None,
    "PesendorferSchmidtDengler2008": None,
    "Petrin2002": None,
    "WeintraubBenkardVanRoy2008": None,
    "FershtmanPakes2012": None,
    "AskerEtAl2020": None,

    # Historical / pre-arXiv
    "Samuel1959": None,
    "Shannon1950": None,
    "Minsky1954": None,
    "Thorndike1913": None,
    "Dickinson1978": None,
    "Skinner1963": None,
    "Jones1924": None,
    "Cross1973": None,
    "Brown1951": None,
    "Robinson1951": None,
    "Shapley1964": None,
    "MondererShapley1996": None,
    "Tesauro1994": None,
    "watkins1992_paper": None,

    # Books (no PDF download)
    "bertsekas1996": None,
    "bertsekastsitsiklis1996": None,
    "sutton2018": None,
    "SuttonBarto1998": None,

    # Misc
    "williams1992": None,  # ML journal
    "williams1992_reinforce": None,
    "Williams1992": None,
    "bartosuttonanderson1983": None,  # IEEE
    "Sutton1988": None,
    "Sutton1990": None,
    "Sutton1999Options": None,
    "brafman2002rmax": None,  # JMLR
    "Brafman2002": None,
    "rummery1994": None,
    "sutton1988": None,
    "tsitsiklis2002": None,
    "sabri2024": None,
    "nomura2025": None,
    "Zhang2023": None,
    "boute2022": None,
    "Bogota2015": None,
    "Dietterich2000": None,
    "Igami2020": None,
    "Igami2020_alphago": None,

    # Applications (institutional reports, not papers)
    "Alibaba2019": None,
    "DiDi2019": "1905.11566",
    "Google2018": None,
    "JPMorgan2018": None,
    "Microsoft2022": None,
    "NPR2025": None,
    "TCJA2018": None,
    "ACA2011": None,
    "FCC2020": None,
    "DLAPiper2020": None,
    "BakerBotts2020": None,
    "Fischer2017": None,
    "FRB2022": None,
    "Constancio2018": None,
    "ArnoldPorter2012": None,
    "Liu2022": None,
}


def load_cache():
    """Load cached Semantic Scholar lookups."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    """Save cache to disk."""
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def query_semantic_scholar(title, cache):
    """Query Semantic Scholar for open-access PDF URL by title."""
    cache_key = re.sub(r"\s+", " ", title.strip().lower())
    if cache_key in cache:
        return cache[cache_key]

    time.sleep(1.0)

    try:
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    try:
        query = urllib.parse.quote(title)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={query}&limit=1&fields=openAccessPdf,externalIds,title"
        )
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "RL-Survey-Paper-Downloader/1.0")
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())

        result = {"pdf_url": None, "arxiv_id": None}
        if data.get("data") and len(data["data"]) > 0:
            paper = data["data"][0]
            if paper.get("openAccessPdf", {}).get("url"):
                result["pdf_url"] = paper["openAccessPdf"]["url"]
            ext_ids = paper.get("externalIds", {})
            if ext_ids.get("ArXiv"):
                result["arxiv_id"] = ext_ids["ArXiv"]

        cache[cache_key] = result
        return result

    except Exception as e:
        print(f"    S2 lookup failed for '{title[:50]}': {e}")
        cache[cache_key] = {"pdf_url": None, "arxiv_id": None}
        return cache[cache_key]


def generate_download_scripts():
    entries = parse_bib(BIB_PATH)
    print(f"Parsed {len(entries)} bib entries")

    cache = load_cache()

    # Map citation keys to chapters
    chapter_citations = {}
    for ch_dir, tex_files in CHAPTERS.items():
        keys = set()
        for tf in tex_files:
            tex_path = ROOT / ch_dir / tf
            keys |= extract_citations_from_tex(tex_path)
        chapter_citations[ch_dir] = keys
        print(f"  {ch_dir}: {len(keys)} citations")

    # Collect all unique cited keys across chapters for batch lookup
    all_cited_keys = set()
    for keys in chapter_citations.values():
        all_cited_keys |= keys

    # Resolve download URLs for all cited entries
    print(f"\nResolving download URLs for {len(all_cited_keys)} unique cited papers...")
    resolved = {}  # key -> {"url": ..., "source": ...}

    needs_s2 = []  # (key, title) pairs that need Semantic Scholar lookup

    for i, key in enumerate(sorted(all_cited_keys, key=str.lower)):
        entry = entries.get(key)
        if not entry:
            for bib_key, bib_entry in entries.items():
                if bib_key.lower() == key.lower():
                    entry = bib_entry
                    break

        # 1. Check known arXiv IDs mapping (works even without bib entry)
        known_id = KNOWN_ARXIV_IDS.get(key)
        if known_id:
            resolved[key] = {
                "url": f"https://arxiv.org/pdf/{known_id}.pdf",
                "source": "known_arxiv",
            }
            continue

        if not entry:
            # Check if known to be unavailable
            if key in KNOWN_ARXIV_IDS:
                resolved[key] = {"url": None, "source": "known_unavailable"}
                continue
            resolved[key] = {"url": None, "source": "missing_from_bib"}
            continue

        title = entry.get("title", "")

        # 2. Check bib entry for arXiv ID
        arxiv_id = extract_arxiv_id(entry)
        if arxiv_id:
            resolved[key] = {
                "url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "source": "bib_arxiv",
            }
            continue

        # 3. Check bib entry for direct PDF URL
        url = extract_url(entry)
        if url and (url.endswith(".pdf") or "pdf" in url.lower()):
            resolved[key] = {"url": url, "source": "bib_url"}
            continue

        # 4. If key is in KNOWN_ARXIV_IDS with None value, skip S2 (known to be unavailable)
        if key in KNOWN_ARXIV_IDS:
            doi = extract_doi(entry)
            resolved[key] = {
                "url": None,
                "source": "known_unavailable",
                "doi": doi if doi else None,
            }
            continue

        # 5. Queue for Semantic Scholar lookup
        needs_s2.append((key, title, entry))

    # Batch S2 lookups for remaining papers
    if needs_s2:
        print(f"\n  {len(needs_s2)} papers not in known mapping, querying Semantic Scholar...")
        for i, (key, title, entry) in enumerate(needs_s2):
            if title:
                print(f"  [{i+1}/{len(needs_s2)}] S2 lookup: {title[:60]}...")
                s2 = query_semantic_scholar(title, cache)
                if s2.get("pdf_url"):
                    resolved[key] = {"url": s2["pdf_url"], "source": "semantic_scholar"}
                    continue
                if s2.get("arxiv_id"):
                    resolved[key] = {
                        "url": f"https://arxiv.org/pdf/{s2['arxiv_id']}.pdf",
                        "source": "s2_arxiv",
                    }
                    continue

            doi = extract_doi(entry)
            resolved[key] = {
                "url": None,
                "source": "not_found",
                "doi": doi if doi else None,
            }

    save_cache(cache)

    # Generate one script per chapter
    scripts_dir = ROOT / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    summary_lines = []

    for ch_dir, cited_keys in chapter_citations.items():
        if not cited_keys:
            summary_lines.append(f"{ch_dir}: no citations found in tex")
            continue

        papers_dir = ROOT / ch_dir / "papers"
        lines = [
            "#!/usr/bin/env bash",
            f"# Download papers for {ch_dir}",
            f"# Auto-generated by generate_download_scripts.py",
            "",
            "set -euo pipefail",
            "",
            f'DEST="{papers_dir}"',
            'mkdir -p "$DEST"',
            "",
        ]

        downloadable = 0
        no_source_keys = []

        for key in sorted(cited_keys, key=str.lower):
            entry = entries.get(key)
            if not entry:
                for bib_key, bib_entry in entries.items():
                    if bib_key.lower() == key.lower():
                        entry = bib_entry
                        break

            title = entry.get("title", "") if entry else ""
            author = entry.get("author", "") if entry else ""
            filename = sanitize_filename(title, key)

            r = resolved.get(key, {"url": None, "source": "unknown"})

            lines.append(f"# {key}: {title[:80]}")
            if author:
                lines.append(f"# Authors: {author[:80]}")

            if r["url"]:
                lines.append(
                    f'[ -f "$DEST/{filename}" ] || '
                    f'curl -sL -o "$DEST/{filename}" '
                    f'"{r["url"]}"'
                )
                lines.append(f"# Source: {r['source']}")
                downloadable += 1
            else:
                doi_note = f" DOI: {r.get('doi')}" if r.get("doi") else ""
                lines.append(f"# NO OPEN-ACCESS SOURCE FOUND.{doi_note} Download manually.")
                no_source_keys.append(key)

            lines.append("")

        # Write the script
        script_name = f"download_{ch_dir}.sh"
        script_path = scripts_dir / script_name
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script_path.chmod(0o755)

        summary_lines.append(
            f"{ch_dir}: {len(cited_keys)} cited | "
            f"{downloadable} downloadable, "
            f"{len(no_source_keys)} manual"
        )
        if no_source_keys:
            for nk in no_source_keys:
                summary_lines.append(f"  - manual: {nk}")

    # Print summary
    print("\n=== DOWNLOAD SCRIPT SUMMARY ===")
    for line in summary_lines:
        print(line)

    total_downloadable = sum(
        1 for r in resolved.values() if r["url"]
    )
    total_manual = sum(1 for r in resolved.values() if not r["url"])
    print(f"\nTotal: {total_downloadable} downloadable, {total_manual} manual")
    print(f"Scripts written to: {scripts_dir}/")
    print("Run individual scripts with: bash scripts/download_chXX_topic.sh")


if __name__ == "__main__":
    generate_download_scripts()
