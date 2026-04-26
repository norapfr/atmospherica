# descomprimir.py
import zipfile
from pathlib import Path

data_dir = Path("ml/data")

for f in sorted(data_dir.glob("era5_sevilla_20??.nc")):
    year = f.stem.split("_")[-1]
    print(f"Procesando {f.name}...")
    with zipfile.ZipFile(f, "r") as z:
        for member in z.namelist():
            # Determinar sufijo: instant o accum
            if "instant" in member:
                suffix = "instant"
            elif "accum" in member:
                suffix = "accum"
            else:
                suffix = member.replace("/", "_")

            target = data_dir / f"era5_{year}_{suffix}.nc"
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  -> {target.name}")

print("\nArchivos resultantes:")
for f in sorted(data_dir.glob("era5_20??_*.nc")):
    print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")