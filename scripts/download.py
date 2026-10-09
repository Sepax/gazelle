"""Download MPIIFaceGaze (about 1 GB) and the MPIIGaze evaluation sample lists into data/."""

import argparse
import io
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

FACEGAZE_URL = (
    "https://darus.uni-stuttgart.de/api/access/dataset/:persistentId/"
    "?persistentId=doi:10.18419/DARUS-3240"
)
MPIIGAZE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/165887"  # MPIIGaze.zip, 2.3 GB
SAMPLE_DIR = "MPIIGaze/Evaluation Subset/sample list for eye image/"
PARTICIPANTS = 15


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched with HTTP range requests."""

    def __init__(self, url: str):
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request) as response:
            self.url = response.geturl()  # final URL after redirects
            self.size = int(response.headers["Content-Range"].rpartition("/")[2])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self.size - self.pos)
        if want <= 0:
            return 0
        headers = {"Range": f"bytes={self.pos}-{self.pos + want - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers)) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


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


def download_facegaze(out: Path) -> None:
    root = out / "MPIIFaceGaze"
    archive = out / "download.zip"
    download(FACEGAZE_URL, archive)
    print("Extracting")
    extract(archive, out)
    extract(root / "Data.zip", root)

    found = len(list((root / "Data").glob("p??")))
    if found != PARTICIPANTS:
        raise SystemExit(f"Error: expected {PARTICIPANTS} participant folders, found {found}")
    print(f"MPIIFaceGaze: {found} participant folders in {root / 'Data'}")


def download_sample_lists(out: Path) -> None:
    """Fetch only the sample lists (about 150 KB) from the 2.3 GB MPIIGaze archive."""
    print("Fetching MPIIGaze sample lists")
    with zipfile.ZipFile(io.BufferedReader(RemoteFile(MPIIGAZE_URL), 1 << 20)) as zf:
        members = [n for n in zf.namelist() if n.startswith(SAMPLE_DIR) and n.endswith(".txt")]
        if len(members) != PARTICIPANTS:
            raise SystemExit(f"Error: expected {PARTICIPANTS} sample lists, found {len(members)}")
        zf.extractall(out, members)
    print(f"Sample lists: {len(members)} files in {out / SAMPLE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--force", action="store_true", help="download even if the data exists")
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if (out / "MPIIFaceGaze" / "Data").exists() and not args.force:
        print(f"MPIIFaceGaze: already present at {out / 'MPIIFaceGaze' / 'Data'}")
    else:
        download_facegaze(out)

    if len(list((out / SAMPLE_DIR).glob("p??.txt"))) == PARTICIPANTS and not args.force:
        print(f"Sample lists: already present at {out / SAMPLE_DIR}")
    else:
        download_sample_lists(out)


if __name__ == "__main__":
    main()