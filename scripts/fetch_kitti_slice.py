import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

KITTI_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"


# seekable file over http range requests. lets zipfile read one member of huge zip.
class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size or len(buf) == 0:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


# pull named members out of remote zip, flatten into out_dir
def extract_members(url: str, wanted: list[str], out_dir: Path) -> None:
    raw = io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20)
    with zipfile.ZipFile(raw) as zf:
        names = set(zf.namelist())
        for name in wanted:
            if name not in names:
                raise FileNotFoundError(f"{name} not in {url}")
            target = out_dir / Path(name).name
            target.write_bytes(zf.read(name))


# fetch first n velodyne scans + calib + times + poses for one kitti odometry sequence
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="00")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/kitti"))
    args = ap.parse_args()

    seq_dir = args.out / "sequences" / args.seq
    velo_dir = seq_dir / "velodyne"
    velo_dir.mkdir(parents=True, exist_ok=True)

    scans = [f"dataset/sequences/{args.seq}/velodyne/{i:06d}.bin" for i in range(args.start, args.start + args.count)]
    extract_members(f"{KITTI_BASE}/data_odometry_velodyne.zip", scans, velo_dir)
    extract_members(f"{KITTI_BASE}/data_odometry_calib.zip", [f"dataset/sequences/{args.seq}/calib.txt", f"dataset/sequences/{args.seq}/times.txt"], seq_dir)
    extract_members(f"{KITTI_BASE}/data_odometry_poses.zip", [f"dataset/poses/{args.seq}.txt"], seq_dir)
    (seq_dir / f"{args.seq}.txt").rename(seq_dir / "poses.txt")
    print(f"GOT {args.count} SCANS + CALIB + TIMES + POSES -> {seq_dir}")


if __name__ == "__main__":
    main()
