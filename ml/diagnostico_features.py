# diagnostico_features.py
import pandas as pd

df = pd.read_csv("ml/data_Sevilla/features15Y.csv", index_col=0, parse_dates=True)

print("=== ESTADISTICAS DE TEMPERATURA ===")
print(df["temp_max"].describe())
print(f"\nDias con temp_max >= 35: {(df['temp_max'] >= 35).sum()}")
print(f"Dias con temp_max >= 38: {(df['temp_max'] >= 38).sum()}")
print(f"Dias con temp_max <= 8:  {(df['temp_max'] <= 8).sum()}")
print(f"Dias con temp_max <= 12: {(df['temp_max'] <= 12).sum()}")

print("\n=== ESTADISTICAS DE PRECIPITACION ===")
print(df["precip_total"].describe())
print(f"Dias con precip >= 10mm: {(df['precip_total'] >= 10).sum()}")
print(f"Dias con precip >= 20mm: {(df['precip_total'] >= 20).sum()}")

print("\n=== ESTADISTICAS DE VIENTO ===")
print(df["wind_max"].describe())
print(f"Dias con wind_max >= 8:  {(df['wind_max'] >= 8).sum()}")
print(f"Dias con wind_max >= 10: {(df['wind_max'] >= 10).sum()}")

print("\n=== EVENTOS ACTUALES ===")
print(f"event_heat:    {df['event_heat'].sum()}")
print(f"event_cold:    {df['event_cold'].sum()}")
print(f"event_rain:    {df['event_rain'].sum()}")
print(f"event_wind:    {df['event_wind'].sum()}")
print(f"event_extreme: {df['event_extreme'].sum()}")