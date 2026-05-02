import math


def normalize(value: float, min_val: float, max_val: float) -> float:
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def lerp_color(color_a: tuple, color_b: tuple, t: float) -> tuple:
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


# Grupos de weather_id de OpenWeatherMap
def _weather_group(weather_id: int) -> str:
    if 200 <= weather_id < 300: return "storm"
    if 300 <= weather_id < 400: return "drizzle"
    if 500 <= weather_id < 600: return "rain"
    if 600 <= weather_id < 700: return "snow"
    if 700 <= weather_id < 800: return "atmosphere"  # niebla, calima, humo
    if weather_id == 800:       return "clear"
    if weather_id > 800:        return "clouds"
    return "unknown"


def map_to_visual(data: dict) -> dict:
    # --- TEMPERATURA ---
    t_norm = normalize(data["temperature"], 0, 46)

    cold_color    = (28, 58, 110)
    cool_color    = (45, 85, 140)
    neutral_color = (90, 70, 50)
    warm_color    = (160, 90, 30)
    hot_color     = (190, 110, 20)

    if t_norm < 0.25:
        base_color = lerp_color(cold_color, cool_color, t_norm / 0.25)
    elif t_norm < 0.5:
        base_color = lerp_color(cool_color, neutral_color, (t_norm - 0.25) / 0.25)
    elif t_norm < 0.75:
        base_color = lerp_color(neutral_color, warm_color, (t_norm - 0.5) / 0.25)
    else:
        base_color = lerp_color(warm_color, hot_color, (t_norm - 0.75) / 0.25)

    # --- PRESIÓN ---
    pressure_norm = normalize(data["pressure"], 990, 1030)
    density       = 0.3 + (pressure_norm * 0.7)
    num_layers    = int(6 + pressure_norm * 14)

    # --- PM2.5 ---
    pm_norm        = normalize(data["pm2_5"], 0, 75)
    fragmentation  = pm_norm
    fragment_count = int(2 + pm_norm * 40)

    # --- HUMEDAD ---
    humidity_norm = normalize(data["humidity"], 10, 95)
    opacity_base  = int(160 + (1 - humidity_norm) * 80)
    veil_opacity  = int(humidity_norm * 80)

    # --- VIENTO ---
    wind_angle_rad = math.radians(data["wind_deg"])
    wind_dx        = math.sin(wind_angle_rad)
    wind_dy        = -math.cos(wind_angle_rad)
    wind_energy    = normalize(data["wind_speed"], 0, 20)
    stroke_length  = int(20 + wind_energy * 180)
    stroke_width   = max(1, int(wind_energy * 8))

    # --- NUBES ---
    cloud_norm  = normalize(data["clouds"], 0, 100)
    bg_darkness = int(15 + cloud_norm * 35)

    # --- LLUVIA ---
    # rain_1h: mm caídos en la última hora (0 si no llueve)
    # Rango: 0 sin lluvia · 2 lluvia moderada · 10+ lluvia torrencial
    rain_1h      = data.get("rain_1h", 0.0)
    rain_norm    = normalize(rain_1h, 0, 10)   # 0 seco · 1 torrencial
    is_raining   = rain_1h > 0.0
    weather_id   = data.get("weather_id", 800)
    weather_grp  = _weather_group(weather_id)

    return {
        "base_color":        base_color,
        "density":           density,
        "num_layers":        num_layers,
        "fragmentation":     fragmentation,
        "fragment_count":    fragment_count,
        "opacity_base":      opacity_base,
        "veil_opacity":      veil_opacity,
        "wind_dx":           wind_dx,
        "wind_dy":           wind_dy,
        "wind_energy":       wind_energy,
        "stroke_length":     stroke_length,
        "stroke_width":      stroke_width,
        "bg_darkness":       bg_darkness,
        "temperature_norm":  t_norm,
        # lluvia
        "rain_1h":           rain_1h,
        "rain_norm":         round(rain_norm, 3),
        "is_raining":        is_raining,
        "weather_id":        weather_id,
        "weather_group":     weather_grp,
        "raw":               data,
    }