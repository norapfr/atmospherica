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

    return {
        "temperature": weather["main"]["temp"],
        "temp_min": weather["main"]["temp_min"],
        "temp_max": weather["main"]["temp_max"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "wind_deg": weather["wind"]["deg"],
        "clouds": weather["clouds"]["all"],
        "visibility": weather.get("visibility", 10000),
        "weather_id": weather["weather"][0]["id"],
        "city": weather["name"],
        "pm2_5": air["list"][0]["components"]["pm2_5"],
        "no2": air["list"][0]["components"]["no2"],
        "o3": air["list"][0]["components"]["o3"],
    }