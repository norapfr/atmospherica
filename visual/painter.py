import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import math
import random


class AtmosphericPainter:
    """
    ATMOSPHERICA — cada decision visual viene de un dato climatico real.

    TEMPERATURA  -> paleta de color (frio=azul/violeta, calido=ambar/dorado)
    PRESION      -> escala del ruido Perlin (estable=fluido, baja=turbulento)
    PM2.5        -> fragmentacion de trazos (limpio=continuo, contaminado=roto)
    HUMEDAD      -> luminosidad y saturacion (seco=vivo, humedo=velado)
    VIENTO       -> direccion y longitud de trazos
    NUBES        -> valor del fondo (despejado=negro profundo, nublado=gris)
    COMPOSICION  -> determinada por el tipo de dia, no por el artista
    """

    def __init__(self, width=800, height=1000, seed=None):
        self.W = width
        self.H = height
        self.seed = seed or random.randint(0, 99999)
        random.seed(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # RUIDO PERLIN — organicidad base
    # ------------------------------------------------------------------
    def _perlin(self, x, y, scale=0.003, octaves=4, offset_x=0, offset_y=0):
        value, amplitude, frequency, max_val = 0.0, 1.0, scale, 0.0
        for i in range(octaves):
            xi = (x + offset_x) * frequency
            yi = (y + offset_y) * frequency
            value += amplitude * (
                math.sin(xi * 1.7 + math.cos(yi * 1.3 + i)) *
                math.cos(yi * 1.5 + math.sin(xi * 0.9 + i * 0.7))
            )
            max_val += amplitude
            amplitude *= 0.5
            frequency *= 2.1
        return value / max_val  # -1 a 1

    # ------------------------------------------------------------------
    # PALETA — temperatura decide los colores, humedad la saturacion
    # ------------------------------------------------------------------
    def _build_palette(self, visual_params):
        """
        Devuelve (color_dark, color_mid, color_bright, color_accent)
        100% determinado por temperatura y humedad.
        """
        t = visual_params["temperature_norm"]   # 0=frio, 1=calor extremo
        h = visual_params["veil_opacity"] / 80.0  # 0=seco, 1=humedo
        p = visual_params["fragmentation"]        # 0=limpio, 1=contaminado

        # FRIO (t < 0.25): azul profundo / violeta / gris acerado
        if t < 0.25:
            dark   = (8,  15,  35)
            mid    = (25, 45,  95)
            bright = (60, 90, 160)
            accent = (90, 70, 140)

        # FRESCO (0.25-0.45): azul-verde / teal / aguamarina
        elif t < 0.45:
            f = (t - 0.25) / 0.20
            dark   = (8,  20,  30)
            mid    = (int(20+f*15), int(55+f*20), int(80-f*10))
            bright = (int(45+f*30), int(100+f*30), int(120-f*20))
            accent = (int(60+f*40), int(120+f*20), int(100+f*10))

        # TEMPLADO (0.45-0.65): tierra / ocre / verde musgo — primavera
        elif t < 0.65:
            f = (t - 0.45) / 0.20
            dark   = (int(15+f*10), int(12+f*8),  int(8+f*5))
            mid    = (int(65+f*40), int(55+f*30), int(20+f*10))
            bright = (int(120+f*50), int(100+f*40), int(35+f*20))
            accent = (int(150+f*40), int(110+f*30), int(30+f*15))

        # CALIDO (0.65-0.82): ambar / naranja / dorado — verano suave
        elif t < 0.82:
            f = (t - 0.65) / 0.17
            dark   = (int(20+f*15), int(10+f*5),  5)
            mid    = (int(140+f*40), int(80+f*20), int(15+f*5))
            bright = (int(200+f*30), int(130+f*30), int(30+f*10))
            accent = (int(220+f*20), int(150+f*20), int(40+f*10))

        # EXTREMO (0.82-1.0): rojo-naranja intenso — ola de calor
        else:
            f = (t - 0.82) / 0.18
            dark   = (30, 8,  3)
            mid    = (int(180+f*30), int(60+f*20), 10)
            bright = (int(230+f*20), int(100+f*20), int(20+f*10))
            accent = (245, int(80+f*30), 15)

        # Desaturar si hay mucha humedad o contaminacion
        desat = h * 0.35 + p * 0.20
        def desat_color(c):
            grey = int((c[0] + c[1] + c[2]) / 3)
            return tuple(int(c[i] * (1-desat) + grey * desat) for i in range(3))

        return (
            desat_color(dark),
            desat_color(mid),
            desat_color(bright),
            desat_color(accent)
        )

    # ------------------------------------------------------------------
    # COMPOSICION CLIMATICA — que zona del canvas es protagonista
    # ------------------------------------------------------------------
    def _climate_weight(self, x, y, visual_params):
        """
        0.0 = zona en sombra / inactiva
        1.0 = zona protagonista / luminosa

        La forma de esta distribucion la decide el clima:
        - Dia estable calido    -> foco central radial (calor sube al centro)
        - Dia ventoso           -> bandas diagonales en la direccion del viento
        - Dia contaminado       -> manchas dispersas irregulares (la policion fragmenta)
        - Dia frio              -> actividad en bordes, vacio central (frio contrae)
        - Dia nublado humedo    -> gradiente suave de arriba a abajo (lluvia cae)
        """
        cx, cy = self.W / 2, self.H / 2
        max_dist = math.sqrt(cx**2 + cy**2)

        wind_dx = visual_params["wind_dx"]
        wind_dy = visual_params["wind_dy"]
        wind_e  = visual_params["wind_energy"]
        pressure = visual_params["density"]        # 0.3=baja presion, 1.0=alta
        pm       = visual_params["fragmentation"]  # contaminacion
        temp     = visual_params["temperature_norm"]
        humidity = visual_params["veil_opacity"] / 80.0
        clouds   = visual_params["bg_darkness"] / 50.0

        # --- VIENTO FUERTE: bandas diagonales ---
        if wind_e > 0.55:
            proj = ((x - cx) * wind_dx + (y - cy) * wind_dy)
            proj_n = proj / max_dist
            band = math.exp(-proj_n**2 * 1.8)
            perp_x, perp_y = -wind_dy, wind_dx
            perp = ((x - cx) * perp_x + (y - cy) * perp_y) / max_dist
            ripple = 0.5 + 0.5 * math.sin(perp * 8.0 + self._perlin(x, y, 0.008) * 4)
            noise = self._perlin(x, y, 0.005, 3) * 0.25
            return max(0.0, min(1.0, band * 0.55 + ripple * 0.30 + noise + 0.15))

        # --- CONTAMINACION ALTA: manchas irregulares ---
        elif pm > 0.45:
            n1 = self._perlin(x, y, 0.007, 4) * 0.5 + 0.5
            n2 = self._perlin(x, y, 0.003, 3, 500, 300) * 0.5 + 0.5
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            scatter = (n1 * 0.5 + n2 * 0.5) * (1.0 - dist * 0.3)
            return max(0.0, min(1.0, scatter * 0.85 + 0.10))

        # --- FRIO: energia en los bordes ---
        elif temp < 0.28:
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            edge = dist ** 1.8
            noise = self._perlin(x, y, 0.005, 3) * 0.20
            return max(0.0, min(1.0, edge * 0.80 + noise + 0.08))

        # --- NUBLADO Y HUMEDO: gradiente vertical suave (lluvia) ---
        elif clouds > 0.55 and humidity > 0.55:
            grad = 1.0 - (y / self.H) * 0.6   # mas activo arriba
            noise = self._perlin(x, y, 0.004, 3) * 0.25
            dist = math.sqrt((x-cx)**2 + (y-cy)**2) / max_dist
            return max(0.0, min(1.0, grad * 0.65 + noise + (1-dist)*0.20))

        # --- DIA ESTABLE CALIDO (Sevilla tipico): foco central ---
        else:
            # El foco se desplaza ligeramente con el viento
            fx = cx + wind_dx * 100 * wind_e
            fy = cy + wind_dy * 100 * wind_e
            dist = math.sqrt((x-fx)**2 + (y-fy)**2) / max_dist
            # Curva de calor: mas suave que gaussiana, mas pictórica
            radial = max(0.0, 1.0 - dist ** 1.3)
            # Ruido organico para que no sea un circulo perfecto
            noise = self._perlin(x, y, 0.006, 4) * 0.28
            # La presion alta suaviza, la baja crea mas textura
            pressure_blend = pressure * radial + (1-pressure) * (radial * 0.7 + noise * 0.3)
            return max(0.0, min(1.0, pressure_blend + noise * 0.5))

    # ------------------------------------------------------------------
    # FLOW FIELD — campo de fuerzas atmosfericas
    # ------------------------------------------------------------------
    def _build_flow_field(self, visual_params):
        cols = self.W // 6
        rows = self.H // 6
        field = np.zeros((rows, cols, 2))

        wind_dx  = visual_params["wind_dx"]
        wind_dy  = visual_params["wind_dy"]
        wind_e   = visual_params["wind_energy"]
        pressure = visual_params["density"]

        # Baja presion = ruido mas agresivo (turbulencia)
        # Alta presion = ruido suave (flujo laminar)
        noise_scale = 0.0035 + (1.0 - pressure) * 0.009
        noise_octaves = 3 if pressure > 0.6 else 5

        for row in range(rows):
            for col in range(cols):
                x, y = col * 6, row * 6
                n = self._perlin(x, y, scale=noise_scale, octaves=noise_octaves)
                # El angulo base viene del ruido
                angle = n * math.pi * 3.0
                # El viento real lo sesga
                dx = math.cos(angle) + wind_dx * wind_e * 1.2
                dy = math.sin(angle) + wind_dy * wind_e * 1.2
                mag = math.sqrt(dx**2 + dy**2) + 1e-6
                field[row, col] = [dx/mag, dy/mag]

        return field

    # ------------------------------------------------------------------
    # FONDO — lienzo imprimado con el estado atmosferico
    # ------------------------------------------------------------------
    def _paint_background(self, img, visual_params, palette):
        dark, mid, bright, accent = palette
        bg_base = visual_params["bg_darkness"]  # nubes -> oscuridad
        pixels = img.load()

        for y in range(self.H):
            for x in range(self.W):
                w = self._climate_weight(x, y, visual_params)
                n = self._perlin(x, y, scale=0.004, octaves=3) * 0.5 + 0.5  # 0-1

                # El fondo mezcla dark y mid segun la composicion climatica
                # Las zonas protagonistas tienen un fondo levemente mas claro
                blend = w * 0.35 + n * 0.15
                r = int(dark[0] + (mid[0] - dark[0]) * blend)
                g = int(dark[1] + (mid[1] - dark[1]) * blend)
                b = int(dark[2] + (mid[2] - dark[2]) * blend)

                # Ajuste por nubes: dias nublados fondo mas gris
                grey_blend = bg_base / 50.0 * 0.3
                grey = int((r+g+b)/3)
                r = int(r*(1-grey_blend) + grey*grey_blend)
                g = int(g*(1-grey_blend) + grey*grey_blend)
                b = int(b*(1-grey_blend) + grey*grey_blend)

                pixels[x, y] = (
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b)),
                    255
                )
        return img

    # ------------------------------------------------------------------
    # CAPA DE PARTICULAS — trazos que siguen el flow field
    # ------------------------------------------------------------------
    def _paint_layer(self, field, visual_params, palette, layer_index, num_layers):
        dark, mid, bright, accent = palette
        depth = layer_index / max(num_layers - 1, 1)  # 0=profundo, 1=superficie

        wind_e  = visual_params["wind_energy"]
        pm      = visual_params["fragmentation"]
        opacity = visual_params["opacity_base"]
        temp    = visual_params["temperature_norm"]

        cols = self.W // 6
        rows = self.H // 6

        layer_img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer_img)

        # Capas profundas: pocas particulas, trazos largos, color oscuro
        # Capas superficiales: muchas particulas, trazos variados, color brillante
        curve = math.sin(depth * math.pi)  # pico en capas medias
        n_particles = int(150 + curve * 1800 + depth * 600)
        use_long_strokes = depth > 0.4  # capas superficiales tienen trazos largos
        n_long = int(n_particles * 0.15) if use_long_strokes else 0
        n_short = n_particles - n_long
        for _ in range(n_particles):
            # Posicion de nacimiento sesgada por composicion climatica
            # Las particulas prefieren nacer donde el clima pone el foco
            best_w = 0
            px, py = 0, 0
            for attempt in range(5):
                tx = random.uniform(0, self.W)
                ty = random.uniform(0, self.H)
                tw = self._climate_weight(tx, ty, visual_params)
                # Aceptar con probabilidad proporcional al weight
                # Siempre aceptar en el ultimo intento para llenar el canvas
                if tw > best_w or attempt == 4:
                    if random.random() < tw or attempt == 4:
                        px, py = tx, ty
                        best_w = tw
                        break

            w = self._climate_weight(px, py, visual_params)

            # COLOR segun profundidad y posicion climatica
            # Capas profundas: color oscuro/medio
            # Capas superficiales en zonas calidas: color brillante/acento
            if depth < 0.35:
                base_c = dark
                target_c = mid
                t_blend = depth / 0.35
            elif depth < 0.70:
                base_c = mid
                target_c = bright
                t_blend = (depth - 0.35) / 0.35
            else:
                base_c = bright
                target_c = accent
                t_blend = (depth - 0.70) / 0.30

            # Las zonas con mas weight (foco climatico) tiran hacia el acento
            focus_blend = w * 0.55 + t_blend * 0.45
            r = int(base_c[0] + (target_c[0] - base_c[0]) * focus_blend)
            g = int(base_c[1] + (target_c[1] - base_c[1]) * focus_blend)
            b = int(base_c[2] + (target_c[2] - base_c[2]) * focus_blend)

            # OPACIDAD: alta en zonas protagonistas, capas superficiales mas visibles
            alpha_base = int(opacity * (0.12 + w * 0.50 + depth * 0.28))
            # Variacion aleatoria natural
            alpha = int(alpha_base * random.uniform(0.6, 1.3))
            alpha = max(12, min(235, alpha))

            # Trazos largos estructurales (15% de particulas superficiales)
            if use_long_strokes and _ < n_long:
                steps = int(80 + wind_e * 200 + w * 80 + random.gauss(0, 20))
            else:
                steps = int(8 + wind_e * 40 + w * 25 + depth * 20 + random.gauss(0, 8))
            steps = max(4, min(400, steps))

            # VELOCIDAD: el viento y el foco aceleran las particulas
            speed_base = 2.5 + wind_e * 5.0 + w * 2.5

            points = [(px, py)]

            for step in range(steps):
                col_idx = int(px / 6) % cols
                row_idx = int(py / 6) % rows
                dx, dy = field[row_idx, col_idx]

                # Micro-turbulencia organica (Perlin fino)
                micro = self._perlin(px + step*0.3, py + step*0.3, scale=0.022) * 0.6

                # El weight local puede redirigir ligeramente la particula
                # hacia el centro del foco climatico
                cx, cy = self.W/2, self.H/2
                to_cx = (cx - px) / (self.W + 1e-6)
                to_cy = (cy - py) / (self.H + 1e-6)
                pull = w * 0.08  # atraccion suave hacia el foco

                px += dx * speed_base + micro + to_cx * pull
                py += dy * speed_base + micro + to_cy * pull

                # FRAGMENTACION por PM2.5: contaminacion rompe los trazos
                # En zonas de foco climatico la fragmentacion se reduce
                frag_prob = pm * 0.14 * (1.0 - w * 0.6)
                if random.random() < frag_prob:
                    break

                if not (0 <= px < self.W and 0 <= py < self.H):
                    break

                points.append((px, py))

            if len(points) > 3:
                # GROSOR: capas superficiales mas gruesas, viento engrosa, foco engrosa
                stroke_w = max(1, int(
                    0.8 +
                    depth * 3.5 +
                    wind_e * 2.0 +
                    w * 2.5 +
                    random.uniform(0, 1.0)
                ))
                draw.line(points, fill=(r, g, b, alpha), width=stroke_w)

        return layer_img

    # ------------------------------------------------------------------
    # VELO ATMOSFERICO — humedad suaviza, aire seco deja nitido
    # ------------------------------------------------------------------
    def _apply_veil(self, img, visual_params):
        veil = visual_params["veil_opacity"]
        if veil < 8:
            return img
        radius = veil / 10.0
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return Image.blend(img, blurred, alpha=min(0.45, veil / 120.0))

    # ------------------------------------------------------------------
    # POST-PROCESO — contraste y saturacion finales segun el dia
    # ------------------------------------------------------------------
    def _post_process(self, img, visual_params):
        temp = visual_params["temperature_norm"]
        humidity = visual_params["veil_opacity"] / 80.0

        # Dias calidos y secos: mas contraste y saturacion
        # Dias frios y humedos: menos contraste, mas plano
        contrast_val = 1.10 + temp * 0.35 - humidity * 0.20
        saturation_val = 1.15 + temp * 0.40 - humidity * 0.25

        img = ImageEnhance.Contrast(img).enhance(contrast_val)
        img = ImageEnhance.Color(img).enhance(saturation_val)
        return img

    # ------------------------------------------------------------------
    # DIAGNOSTICO — que tipo de dia es y por que
    # ------------------------------------------------------------------
    def _diagnose(self, visual_params):
        t    = visual_params["temperature_norm"]
        wind = visual_params["wind_energy"]
        pm   = visual_params["fragmentation"]
        pres = visual_params["density"]
        hum  = visual_params["veil_opacity"] / 80.0
        cld  = visual_params["bg_darkness"] / 50.0

        if wind > 0.55:
            tipo = "VENTOSO - bandas diagonales en direccion del viento"
        elif pm > 0.45:
            tipo = "CONTAMINADO - manchas dispersas irregulares"
        elif t < 0.28:
            tipo = "FRIO - energia en bordes, vacio central"
        elif cld > 0.55 and hum > 0.55:
            tipo = "NUBLADO/LLUVIA - gradiente vertical suave"
        else:
            tipo = "ESTABLE - foco central radial (anticiclon)"

        temp_c = visual_params["raw"]["temperature"]
        print(f"\n  Diagnostico climatico: {tipo}")
        print(f"  Temperatura: {temp_c:.1f}C  (norm: {t:.2f})")
        print(f"  Presion: {visual_params['raw']['pressure']} hPa  (densidad: {pres:.2f})")
        print(f"  Viento: {visual_params['raw']['wind_speed']} m/s  (energia: {wind:.2f})")
        print(f"  PM2.5: {visual_params['raw']['pm2_5']}  (fragmentacion: {pm:.2f})")
        print(f"  Humedad: {visual_params['raw']['humidity']}%  (velo: {visual_params['veil_opacity']})")
        print(f"  Nubes: {visual_params['raw']['clouds']}%")
        print(f"  Seed: {self.seed}\n")

    # ------------------------------------------------------------------
    # COMPOSITOR PRINCIPAL
    # ------------------------------------------------------------------
    def paint(self, visual_params) -> Image.Image:
        self._diagnose(visual_params)

        # Paleta determinada por temperatura y humedad
        palette = self._build_palette(visual_params)
        print(f"  Paleta -> dark:{palette[0]} mid:{palette[1]} bright:{palette[2]}")

        # Canvas base
        img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 255))

        # 1. Fondo
        print("\n  [1/4] Fondo climatico...")
        img = self._paint_background(img, visual_params, palette)

        # 2. Flow field
        print("  [2/4] Campo de fuerzas...")
        field = self._build_flow_field(visual_params)

        # 3. Capas de particulas
        num_layers = visual_params["num_layers"]
        print(f"  [3/4] Pintando {num_layers} capas de trazos...")
        for i in range(num_layers):
            layer = self._paint_layer(field, visual_params, palette, i, num_layers)
            img = Image.alpha_composite(img, layer)
            pct = int((i+1)/num_layers*100)
            print(f"         [{pct:3d}%] capa {i+1}/{num_layers}")

        # 4. Velo de humedad
        print("  [4/4] Velo atmosferico y post-proceso...")
        img = self._apply_veil(img, visual_params)

        # 5. Post-proceso: contraste y saturacion segun el dia
        img_rgb = img.convert("RGB")
        img_rgb = self._post_process(img_rgb, visual_params)

        return img_rgb