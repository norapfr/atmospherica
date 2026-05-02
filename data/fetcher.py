import requests
from config import API_KEY, CITY, COUNTRY_CODE, UNITS
from config import BASE_URL_WEATHER, BASE_URL_AIR


def get_weather() -> dict:
    params = {
        "q": f"{CITY},{COUNTRY_CODE}",
        "appid": API_KEY,
        "units": UNITS,
    }
    response = requests.get(BASE_URL_WEATHER, params=params)
    response.raise_for_status()
    return response.json()


def get_air_quality(lat: float, lon: float) -> dict:
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
    }
    response = requests.get(BASE_URL_AIR, params=params)
    response.raise_for_status()
    return response.json()


def get_all_data() -> dict:
    weather = get_weather()

    lat = weather["coord"]["lat"]
    lon = weather["coord"]["lon"]
    air = get_air_quality(lat, lon)

    # Precipitación real de la última hora (mm) — campo opcional en la API
    rain_1h = weather.get("rain", {}).get("1h", 0.0)

    # Código del fenómeno meteorológico
    # 2xx tormenta · 3xx llovizna · 5xx lluvia · 6xx nieve
    # 7xx niebla/calima · 800 despejado · 8xx nubes
    weather_id = weather["weather"][0]["id"]

    # Componentes de calidad del aire — todos los que devuelve la API
    air_components = air["list"][0]["components"]

    return {
        # ── Temperatura ──────────────────────────────────────────
        "temperature": weather["main"]["temp"],
        "temp_min":    weather["main"]["temp_min"],
        "temp_max":    weather["main"]["temp_max"],
        "feels_like":  weather["main"].get("feels_like", weather["main"]["temp"]),

        # ── Atmósfera ────────────────────────────────────────────
        "humidity":    weather["main"]["humidity"],
        "pressure":    weather["main"]["pressure"],
        "sea_level":   weather["main"].get("sea_level", weather["main"]["pressure"]),
        "grnd_level":  weather["main"].get("grnd_level", weather["main"]["pressure"]),

        # ── Viento ───────────────────────────────────────────────
        "wind_speed":  weather["wind"]["speed"],
        "wind_deg":    weather["wind"]["deg"],
        "wind_gust":   weather["wind"].get("gust", weather["wind"]["speed"]),

        # ── Nubes y visibilidad ──────────────────────────────────
        "clouds":      weather["clouds"]["all"],
        "visibility":  weather.get("visibility", 10000),

        # ── Precipitación ────────────────────────────────────────
        "rain_1h":     rain_1h,
        "rain_3h":     weather.get("rain", {}).get("3h", 0.0),
        "snow_1h":     weather.get("snow", {}).get("1h", 0.0),

        # ── Fenómeno ─────────────────────────────────────────────
        "weather_id":  weather_id,
        "weather_main": weather["weather"][0]["main"],
        "weather_desc": weather["weather"][0]["description"],

        # ── Sol ──────────────────────────────────────────────────
        "sunrise":     weather["sys"].get("sunrise", 0),
        "sunset":      weather["sys"].get("sunset", 0),

        # ── Ubicación ────────────────────────────────────────────
        "city":        weather["name"],
        "lat":         lat,
        "lon":         lon,

        # ── Calidad del aire ─────────────────────────────────────
        "aqi":         air["list"][0]["main"]["aqi"],  # 1=bueno … 5=muy malo
        "pm2_5":       air_components.get("pm2_5",  0.0),
        "pm10":        air_components.get("pm10",   0.0),
        "no2":         air_components.get("no2",    0.0),
        "o3":          air_components.get("o3",     0.0),
        "co":          air_components.get("co",     0.0),
        "so2":         air_components.get("so2",    0.0),
        "nh3":         air_components.get("nh3",    0.0),
    }