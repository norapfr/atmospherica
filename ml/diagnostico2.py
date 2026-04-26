# diagnostico2.py
import xarray as xr
from pathlib import Path

path = Path("ml/data")

for stream in ["instant", "accum"]:
    files = sorted(path.glob(f"era5_sevilla_2020_*{stream}*.nc"))
    if files:
        ds = xr.open_dataset(files[0], engine="netcdf4")
        print(f"\n--- {stream} ---")
        print("Variables:", list(ds.data_vars))
        print("Coordenadas:", list(ds.coords))
        print("Dimensiones:", dict(ds.dims))
        ds.close()