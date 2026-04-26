import math


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Convierte cualquier valor a un rango 0.0 - 1.0"""
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def lerp_color(color_a: tuple, color_b: tuple, t: float) -> tuple:
    """Interpola entre dos colores RGB según t (0.0 = a, 1.0 = b)"""
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def map_to_visual(data: dict) -> dict:
    """
    Gramática visual de ATMOSPHERICA:

    temperatura  → tono base (frío azul ↔ cálido ámbar)
    presión      → densidad de capas (compacto ↔ abierto)
    pm2_5        → fragmentación (formas continuas ↔ rotas)
    humedad      → opacidad general (nítido ↔ velado)
    viento       → ángulo e intensidad de los trazos
    nubes        → oscuridad del fondo
    """

    # --- TEMPERATURA → color base ---
    # Rango Sevilla: 0°C invierno extremo, 46°C verano extremo
    t_norm = normalize(data["temperature"], 0, 46)

    cold_color = (28, 58, 110)      # azul noche profundo
    cool_color = (45, 85, 140)      # azul medio
    neutral_color = (90, 70, 50)    # tierra neutra
    warm_color = (160, 90, 30)      # ámbar oscuro
    hot_color = (190, 110, 20)      # ámbar intenso

    if t_norm < 0.25:
        base_color = lerp_color(cold_color, cool_color, t_norm / 0.25)
    elif t_norm < 0.5:
        base_color = lerp_color(cool_color, neutral_color, (t_norm - 0.25) / 0.25)
    elif t_norm < 0.75:
        base_color = lerp_color(neutral_color, warm_color, (t_norm - 0.5) / 0.25)
    else:
        base_color = lerp_color(warm_color, hot_color, (t_norm - 0.75) / 0.25)

    # --- PRESIÓN → densidad de capas ---
    # Alta presión (>1020): día estable, capas compactas
    # Baja presión (<1005): inestabilidad, capas separadas
    pressure_norm = normalize(data["pressure"], 990, 1030)
    density = 0.3 + (pressure_norm * 0.7)   # 0.3 (abierto) → 1.0 (compacto)
    num_layers = int(6 + pressure_norm * 14)  # 6 capas (tormenta) → 20 (anticiclón)

    # --- PM2.5 → fragmentación ---
    # OMS: <10 bueno, 10-25 moderado, >25 malo, >75 muy malo
    pm_norm = normalize(data["pm2_5"], 0, 75)
    fragmentation = pm_norm              # 0.0 = formas continuas, 1.0 = muy fragmentado
    fragment_count = int(2 + pm_norm * 40)  # número de fragmentos rotos

    # --- HUMEDAD → opacidad ---
    # Baja humedad: colores opacos y definidos (Sevilla en verano)
    # Alta humedad: velos translúcidos superpuestos
    humidity_norm = normalize(data["humidity"], 10, 95)
    opacity_base = int(160 + (1 - humidity_norm) * 80)  # 160-240
    veil_opacity = int(humidity_norm * 80)                # 0-80

    # --- VIENTO → dirección y energía de trazos ---
    wind_angle_rad = math.radians(data["wind_deg"])
    wind_dx = math.sin(wind_angle_rad)   # componente horizontal
    wind_dy = -math.cos(wind_angle_rad)  # componente vertical
    wind_energy = normalize(data["wind_speed"], 0, 20)  # 0 calma → 1 vendaval
    stroke_length = int(20 + wind_energy * 180)          # longitud del trazo
    stroke_width = max(1, int(wind_energy * 8))          # grosor del trazo

    # --- NUBES → oscuridad del fondo ---
    cloud_norm = normalize(data["clouds"], 0, 100)
    bg_darkness = int(15 + cloud_norm * 35)   # fondo: casi negro → gris oscuro

    return {
        "base_color": base_color,
        "density": density,
        "num_layers": num_layers,
        "fragmentation": fragmentation,
        "fragment_count": fragment_count,
        "opacity_base": opacity_base,
        "veil_opacity": veil_opacity,
        "wind_dx": wind_dx,
        "wind_dy": wind_dy,
        "wind_energy": wind_energy,
        "stroke_length": stroke_length,
        "stroke_width": stroke_width,
        "bg_darkness": bg_darkness,
        "temperature_norm": t_norm,
        "raw": data,
    }