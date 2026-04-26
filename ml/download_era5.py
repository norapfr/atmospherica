import cdsapi
import os

def download_era5(output_dir: str = "ml/data"):
    os.makedirs(output_dir, exist_ok=True)
    c = cdsapi.Client()

    print("Descargando ERA5 para Sevilla (2020-2024)...\n")

    for year in range(2020, 2025):
        print(f"Descargando año {year}...")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "surface_pressure",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "total_precipitation",
                    "total_cloud_cover",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time":  ["06:00", "12:00", "18:00"],
                "area":  [38.0, -6.5, 36.5, -5.0],
                "format": "netcdf",
            },
            f"{output_dir}/era5_sevilla_{year}.nc"
        )

    print("\nDescarga completada.")

if __name__ == "__main__":
    download_era5()