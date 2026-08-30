#!/usr/bin/env python3
"""Detect splice/truncation corruption in generated text.

Targets the signature the user reported at 40k context: fluent fragments
with mid-sentence truncations, duplicated phrases/section restarts, broken
list numbering, orphaned fragments. Windowed metrics:
  - frag_rate: sentences not ending in terminal punctuation and under 40
    chars (orphaned fragments)
  - dup5: density of repeated 5-grams within the window
  - splice: occurrences of a lowercase-letter run followed by a digit or
    paren directly (e.g. "drains3.", "queue untilX") — missing-space token
    splices
Flags a window when any metric is an outlier vs. the file's own median.
"""
import sys
import re


def metrics(text):
    sents = re.split(r"(?<=[.!?])\s+|\n", text)
    sents = [s.strip() for s in sents if s.strip()]
    frag = sum(
        1
        for s in sents
        if len(s) < 40 and not re.search(r"[.!?:;`]$", s) and len(s.split()) > 1
    )
    frag_rate = frag / max(1, len(sents))
    words = text.split()
    grams = {}
    for i in range(len(words) - 4):
        g = tuple(words[i : i + 5])
        grams[g] = grams.get(g, 0) + 1
    dup5 = sum(c for c in grams.values() if c > 1) / max(1, len(words) - 4)
    splice = len(re.findall(r"[a-z]{3}[0-9]\.", text)) + len(
        re.findall(r"[a-z]{2}[0-9][a-z]", text)
    )
    return frag_rate, dup5, splice, len(sents)


def main():
    for path in sys.argv[1:]:
        text = open(path, errors="replace").read()
        W = 2000
        rows = []
        for i in range(0, len(text), W):
            rows.append((i, *metrics(text[i : i + W])))
        import statistics as st

        med_frag = st.median([r[1] for r in rows])
        med_dup = st.median([r[2] for r in rows])
        flagged = 0
        for i, frag, dup, sp, ns in rows:
            bad = frag > max(0.35, med_frag * 2) or dup > max(0.06, med_dup * 2.5) or sp >= 3
            if bad:
                flagged += 1
            mark = "  <-- SPLICE-SUSPECT" if bad else ""
            print(f"{path}:{i:>7} frag={frag:.2f} dup5={dup:.3f} splice={sp} sents={ns}{mark}")
        print(f"{path}: {flagged}/{len(rows)} windows flagged")


if __name__ == "__main__":
    main()
