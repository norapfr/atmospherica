# diagnostico.py
from pathlib import Path

for f in sorted(Path("ml/data").glob("era5_sevilla_*.nc")):
    size = f.stat().st_size
    with open(f, "rb") as fp:
        header = fp.read(16)
    print(f"{f.name}: {size/1024:.1f} KB | header hex: {header.hex()} | ascii: {header[:8]}")