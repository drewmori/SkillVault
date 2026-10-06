"""Fetch verified portable Ollama into the ignored journey-test output folder."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "journey-test" / "ollama-runtime"
DEST.mkdir(parents=True, exist_ok=True)
release = json.load(urllib.request.urlopen("https://api.github.com/repos/ollama/ollama/releases/latest", timeout=30))
asset = next(item for item in release["assets"] if item["name"] == "ollama-windows-amd64.zip")
digest = asset["digest"].removeprefix("sha256:")
archive = DEST / asset["name"]

def verified(path):
    if not path.exists() or path.stat().st_size != asset["size"]:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest

if not verified(archive):
    partial = archive.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    print(f"Downloading {release['tag_name']} ({asset['size']/1e9:.2f} GB) from the official Ollama release", flush=True)
    with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response, partial.open("wb") as target:
        total = 0
        next_notice = 200 * 1024 * 1024
        while chunk := response.read(4 * 1024 * 1024):
            target.write(chunk)
            total += len(chunk)
            if total >= next_notice:
                print(f"Downloaded {total/1e9:.2f} GB", flush=True)
                next_notice += 200 * 1024 * 1024
    if not verified(partial):
        raise RuntimeError("Downloaded runtime failed its official size or SHA256 check")
    partial.replace(archive)

unpacked = DEST / "bin"
unpacked.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as package:
    for entry in package.infolist():
        target = (unpacked / entry.filename).resolve()
        if not target.is_relative_to(unpacked.resolve()):
            raise RuntimeError("Unexpected archive path")
    package.extractall(unpacked)
    names = package.namelist()
print(f"Verified runtime SHA256 {digest}. Extracted {len(names)} entries to {unpacked}", flush=True)
