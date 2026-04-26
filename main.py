import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from data.fetcher import get_all_data
from visual.mapper import map_to_visual
from visual.generator import generate_html
import webbrowser

if __name__ == "__main__":
    print("ATMOSPHERICA\n")

    print("Obteniendo datos...")
    data = get_all_data()
    print(f"  {data['city']} — {data['temperature']}C | {data['pressure']} hPa | "
          f"viento {data['wind_speed']} m/s | PM2.5 {data['pm2_5']}")

    print("\nMapeando parametros visuales...")
    visual = map_to_visual(data)

    print("\nGenerando cuadro...")
    path = generate_html(visual)

    print("\nAbriendo en navegador...")
    webbrowser.open(f"file://{os.path.abspath(path)}")
    print("Cuando el cuadro termine de pintarse, pulsa GUARDAR PNG.")