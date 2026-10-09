"""Download and extract MPIIFaceGaze (about 1 GB) into data/."""

import argparse
import urllib.request
import zipfile
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

URL = (
    "https://darus.uni-stuttgart.de/api/access/dataset/:persistentId/"
    "?persistentId=doi:10.18419/DARUS-3240"
)
PARTICIPANTS = 15


def download(url: str, dest: Path) -> None:
    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out, Progress(*columns) as progress:
        total = int(response.headers.get("Content-Length") or 0) or None
        task = progress.add_task("Downloading", total=total)
        while chunk := response.read(1 << 20):
            out.write(chunk)
            progress.update(task, advance=len(chunk))


def extract(archive: Path, dest: Path) -> None:
    """Extract an archive without macOS metadata, then delete it."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        zf.extractall(dest, names)
    archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--force", action="store_true", help="download even if the data exists")
    args = parser.parse_args()

    root = args.out / "MPIIFaceGaze"
    if (root / "Data").exists() and not args.force:
        print(f"Already present at {root / 'Data'}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "download.zip"
    download(URL, archive)
    print("Extracting")
    extract(archive, args.out)
    extract(root / "Data.zip", root)

    found = len(list((root / "Data").glob("p??")))
    if found != PARTICIPANTS:
        raise SystemExit(f"Error: expected {PARTICIPANTS} participant folders, found {found}")
    print(f"Done: {found} participant folders in {root / 'Data'}")


if __name__ == "__main__":
    main()