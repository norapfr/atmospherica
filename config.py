import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Sevilla"
COUNTRY_CODE = "ES"
UNITS = "metric"

BASE_URL_WEATHER = "https://api.openweathermap.org/data/2.5/weather"
BASE_URL_AIR = "http://api.openweathermap.org/data/2.5/air_pollution"