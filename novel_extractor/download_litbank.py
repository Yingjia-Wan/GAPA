"""Fetch the LitBank source texts that section 6.2 is extracted from.

The 100 novels are not vendored into this repository. They are public-domain Project
Gutenberg texts distributed with LitBank (Bamman et al., 2019), and they are ~58 MB --
a third of the repo -- while nothing in this codebase reads them except `extract.py`,
which you only need if you are re-running extraction from scratch. The extracted
attributes in `novel_extractor/extracted/litbank/` are committed and are what the
analysis actually consumes.

The download is **pinned to a specific upstream commit** and every file is verified
against `litbank_checksums.sha256`. A branch reference would silently mean "whatever
LitBank looks like today"; a checksum mismatch here means upstream changed and your
inputs would no longer be the ones the paper used.

    python novel_extractor/download_litbank.py            # download + verify
    python novel_extractor/download_litbank.py --verify    # verify what is on disk
    python novel_extractor/download_litbank.py --force     # re-download over existing

LitBank is CC BY 4.0. Cite:
    Bamman, Popat & Shen (2019). An annotated dataset of literary entities. NAACL.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

# Pinned upstream state. Do not change these two together without re-verifying:
# the manifest is the checksum of `original/` AT this commit.
LITBANK_REPO = "dbamman/litbank"
LITBANK_COMMIT = "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8"
TARBALL_URL = f"https://codeload.github.com/{LITBANK_REPO}/tar.gz/{LITBANK_COMMIT}"

SCRIPT_DIR = Path(__file__).resolve().parent
DEST_DIR = SCRIPT_DIR / "source" / "litbank"
MANIFEST = SCRIPT_DIR / "litbank_checksums.sha256"


def load_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        sys.exit(f"Missing manifest: {MANIFEST}")
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(expected: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    ok, bad, missing = [], [], []
    for name, digest in sorted(expected.items()):
        p = DEST_DIR / name
        if not p.exists():
            missing.append(name)
        elif sha256(p) == digest:
            ok.append(name)
        else:
            bad.append(name)
    return ok, bad, missing


def download(expected: dict[str, str]) -> None:
    print(f"Fetching {LITBANK_REPO} @ {LITBANK_COMMIT[:12]} ...")
    with urllib.request.urlopen(TARBALL_URL, timeout=300) as resp:
        blob = resp.read()
    print(f"  {len(blob) / 1e6:.1f} MB downloaded; extracting original/ ...")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(expected)
    written = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            # <repo>-<sha>/original/<file>.txt
            if len(parts) < 3 or parts[1] != "original":
                continue
            name = parts[-1]
            if name not in wanted:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            (DEST_DIR / name).write_bytes(src.read())
            written += 1
    print(f"  wrote {written} file(s) to {DEST_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="Only check what is already on disk; download nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the files are already present and valid.")
    args = ap.parse_args()

    expected = load_manifest()
    ok, bad, missing = verify(expected)

    if args.verify:
        print(f"{len(ok)}/{len(expected)} verified, {len(bad)} corrupt, {len(missing)} missing")
        for n in bad:
            print(f"  CHECKSUM MISMATCH: {n}")
        for n in missing[:10]:
            print(f"  missing: {n}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more missing")
        return 0 if not bad and not missing else 1

    if not bad and not missing and not args.force:
        print(f"All {len(ok)} files already present and verified. Nothing to do.")
        return 0

    download(expected)
    ok, bad, missing = verify(expected)
    print(f"\n{len(ok)}/{len(expected)} verified")
    if bad or missing:
        for n in bad:
            print(f"  CHECKSUM MISMATCH: {n}")
        for n in missing:
            print(f"  MISSING: {n}")
        print("\nUpstream no longer matches the pinned manifest. These would NOT be the\n"
              "inputs the paper used - do not proceed with extraction.")
        return 1
    print("All files match the pinned manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
