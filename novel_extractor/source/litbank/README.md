# LitBank source texts

This directory is populated on demand and its `*.txt` files are **not committed**.

```bash
python novel_extractor/download_litbank.py            # fetch + verify (100 files, ~58 MB)
python novel_extractor/download_litbank.py --verify   # check what is on disk
```

The download is pinned to LitBank commit `3e50db0` and every file is checked against
`novel_extractor/litbank_checksums.sha256`, so a mismatch means upstream changed and
the files would no longer be the ones used in the paper.

You only need these to re-run `extract.py`. The extracted attributes that the section
6.2 analysis actually consumes are committed at `novel_extractor/extracted/litbank/`.

Texts are public-domain Project Gutenberg novels distributed with LitBank
(Bamman, Popat & Shen, 2019), CC BY 4.0.
