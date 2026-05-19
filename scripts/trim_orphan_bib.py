#!/usr/bin/env python3
"""Trim orphan entries from docs/refs.bib.

Loads cited-keys set from /tmp/cites_2026-05-20.json (produced by
arxiv-check's extract_cites.py), parses docs/refs.bib via bibtexparser,
writes back only entries whose key is cited.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/pranjal/.claude/skills/arxiv-check/scripts')

import bibtexparser
from bibtexparser.bwriter import BibTexWriter

CITES_JSON = '/tmp/cites_2026-05-20.json'
BIB_PATH = Path('docs/refs.bib')

cited = set(json.load(open(CITES_JSON))['cited_keys'])
print(f'Cited keys: {len(cited)}')

parser = bibtexparser.bparser.BibTexParser(
    common_strings=True, ignore_nonstandard_types=False
)
with open(BIB_PATH) as f:
    db = bibtexparser.load(f, parser=parser)

print(f'Total bib entries: {len(db.entries)}')
kept = [e for e in db.entries if e.get('ID', e.get('id', '')) in cited]
dropped = [e for e in db.entries if e.get('ID', e.get('id', '')) not in cited]
print(f'Keeping: {len(kept)}')
print(f'Dropping: {len(dropped)}')

db.entries = kept

writer = BibTexWriter()
writer.indent = '  '
writer.order_entries_by = None

with open(BIB_PATH, 'w') as f:
    f.write(writer.write(db))

print(f'Wrote {len(db.entries)} entries to {BIB_PATH}')
