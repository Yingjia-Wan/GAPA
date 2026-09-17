# Attribute extraction

## Source texts

### What is here

`source/sampled_attributes_replaced.csv` — the sampled attribute list used during annotation.

`source/litbank/` — **not vendored.** The 100 public-domain Project Gutenberg novels
distributed with LitBank (Bamman et al., 2019) are fetched on demand:

```bash
python novel_extractor/download_litbank.py            # fetch + verify (~58 MB)
python novel_extractor/download_litbank.py --verify   # check what is on disk
```

The download is pinned to LitBank commit `3e50db0` and every file is verified against
`novel_extractor/litbank_checksums.sha256`, so upstream drift fails loudly instead of
silently changing your inputs. LitBank is CC BY 4.0 — keep its attribution.

You only need these to re-run `extract.py`; the extracted attributes the analysis
consumes are committed at `novel_extractor/extracted/litbank/`.

### What is NOT here

The 21 in-copyright files used for the novel-extracted portion of GAPA have been
**removed**, because redistributing entire copyrighted books is not permissible:

- *Harry Potter* 1–7 (`hp_series/`, and `harrypotter1.txt`)
- *Twilight* 1–5 (`twilight_series/`, and `twilight1.txt`)
- *A Game of Thrones*, *The Fellowship of the Ring*, *The Hunger Games*,
  *The Maze Runner*

**The derived data is kept in full.** `novel_extractor/extracted/*_physattr.csv` retains
every extracted attribute phrase together with the sentence it was found in. Those
excerpts are non-contiguous single sentences amounting to 1.6–3.7% of any individual
work — retained so the extraction can be verified in context.

### Reconstructing the inputs

To re-run extraction end to end, obtain your own copies of the 21 files listed in
`source/checksums.sha256`, convert them to plain text, and place them at the paths recorded
there. Then verify:

```bash
cd novel_extractor/source
sha256sum -c checksums.sha256
```

The manifest is a *verification* aid, not a download list: it will report every file as
missing until you supply it. The extraction pipeline is whitespace- and
encoding-sensitive, so an edition whose checksum differs will not reproduce the
published extraction exactly.

## Annotation

The sampled attributes go out for human annotation, which is how the extraction is checked.

```bash
python novel_extractor/annotation/create_annotation_file.py \
    --input novel_extractor/sampled_attributes_final.csv \
    --output novel_extractor/annotation/annotation.csv
python novel_extractor/annotation/eval_annotation_results.py \
    --original novel_extractor/sampled_attributes_final.csv \
    --annotation_files <filled annotator CSVs>
```

Annotators label each sampled attribute on whether it really is a physical attribute, whether it is contextually well-formed, and the character's gender; `annotation/annotation.csv` is the blank sheet they were given, and `annotation/ANNOTATION_GUIDELINES.md` is what they were told. The eval script reports inter-annotator agreement and checks the extractor's own gender call against theirs.
