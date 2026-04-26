import json
import os
import math
from datetime import datetime


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ATMOSPHERICA</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#06060a; display:flex; flex-direction:column;
          align-items:center; justify-content:center; min-height:100vh;
          font-family:'Courier New', monospace; }}
  #topbar {{ width:900px; max-width:calc(100vw - 32px); display:flex;
             align-items:flex-start; justify-content:space-between;
             margin-bottom:12px; gap:18px; }}
  #topinfo {{ display:flex; flex-direction:column; gap:4px; flex:1; }}
  #title {{ font-size:11px; color:#5a5a5a; letter-spacing:0.18em; }}
  #meta  {{ font-size:9.5px; color:#383838; letter-spacing:0.1em; line-height:1.8; }}
  #progress {{ font-size:9px; color:#222; letter-spacing:0.1em; }}
  #wrap {{ position:relative; }}
  canvas {{ display:block; }}
  #legend {{ margin-top:14px; display:flex; flex-direction:column;
             gap:6px; max-width:900px; width:100%; }}
  .leg {{ display:grid; grid-template-columns:9px 1fr; column-gap:9px;
          align-items:start; font-size:9.5px; color:#444; letter-spacing:0.04em; }}
  .dot {{ width:9px; height:9px; border-radius:2px; margin-top:2px; }}
  .leg strong {{ display:block; color:#686868; font-weight:normal; letter-spacing:0.08em; }}
  .leg small {{ display:block; color:#383838; font-size:8.5px; line-height:1.45; margin-top:1px; }}
  #save {{ padding:7px 20px; background:transparent; border:1px solid #181818;
           color:#383838; font-family:monospace; font-size:9px; cursor:pointer;
           letter-spacing:0.14em; transition:all 0.2s; white-space:nowrap; }}
  #save:hover {{ border-color:#555; color:#999; }}
</style>
</head>
<body>
<div id="topbar">
  <div id="topinfo">
    <div id="title"></div>
    <div id="meta"></div>
    <div id="progress">preparando...</div>
  </div>
  <button id="save" onclick="saveImg()">GUARDAR PNG</button>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="legend"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
const C = {climate_json};
const W = 900, H = 1100;

function saveImg() {{
  const cnv = document.getElementById('c');
  const tmp = document.createElement('canvas');
  tmp.width = cnv.width; tmp.height = cnv.height;
  const ctx = tmp.getContext('2d');
  ctx.fillStyle = '#06060a';
  ctx.fillRect(0, 0, tmp.width, tmp.height);
  ctx.drawImage(cnv, 0, 0);
  document.getElementById('progress').textContent = 'guardando...';
  tmp.toBlob(blob => {{
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.download = 'atmospherica_' + C.city + '_' + C.date + '.png';
    a.href = url;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    document.getElementById('progress').textContent = 'guardado';
  }}, 'image/png');
}}

new p5(sk => {{

  let passes = [], passIdx = 0, done = false;

  // ═══════════════════════════════════════════════════════════════
  // PALETA — temperatura + hora mandan en el color
  // El fondo SIEMPRE es oscuro (max brillo 18).
  // Las formas tienen brillo moderado para destacar sobre negro.
  // ═══════════════════════════════════════════════════════════════
  function buildPalette() {{
    const t   = C.temp_norm;
    const h   = C.hour;
    const cld = C.cloud_norm;

    // Hue base segun hora — solo afecta al tinte, no al brillo del fondo
    let timeHue, timeSat;
    if      (h < 5)  {{ timeHue = 230; timeSat = 50; }}  // madrugada: azul
    else if (h < 8)  {{ timeHue = 280; timeSat = 45; }}  // alba: malva
    else if (h < 11) {{ timeHue = 40;  timeSat = 40; }}  // manana: ocre
    else if (h < 15) {{ timeHue = 48;  timeSat = 35; }}  // mediodia: ambar
    else if (h < 19) {{ timeHue = 28;  timeSat = 55; }}  // tarde: naranja
    else if (h < 22) {{ timeHue = 18;  timeSat = 50; }}  // atardecer
    else             {{ timeHue = 220; timeSat = 45; }}  // noche

    // Temperatura desplaza el hue: frio→azul, calor→rojo
    const tempShift = (t - 0.5) * 65;
    const fHue      = (timeHue + tempShift + 360) % 360;
    const fSat      = Math.min(100, timeSat + t * 28);
    const cloudD    = cld * 0.32;

    // Brillo de las formas: oscuro incluso en mediodia, luminoso en calor
    // Rango seguro: 8–55 para que el fondo negro domine siempre
    const formBri   = Math.max(8, Math.min(55, 12 + t * 42 - cld * 12));

    return {{
      // Color principal de las formas (temperatura + hora)
      primary:    [fHue,          fSat*(1-cloudD),        formBri],
      // Complementario — ~150° de diferencia
      complement: [(fHue+150+t*28)%360, fSat*0.68*(1-cloudD), formBri*0.78],
      // Acento — alta saturacion para destellos
      accent:     [(fHue+218+t*18)%360, Math.min(100,fSat*1.28), Math.min(62,formBri*1.18)],
      // Oscuro — capas profundas de las formas
      dark:       [fHue,          fSat*0.38*(1-cloudD),   Math.max(4, formBri*0.16)],
      // Fondo — SIEMPRE muy oscuro, solo un tinte sutil de la hora
      bg:         [timeHue,       timeSat*0.10,            Math.max(3, 4 + h*0.25)],
      // Neutro — presion y nubes
      neutral:    [(fHue+28)%360, 16,                     Math.min(50, formBri*0.62)],
    }};
  }}

  // ═══════════════════════════════════════════════════════════════
  // SETUP
  // ═══════════════════════════════════════════════════════════════
  sk.setup = function() {{
    let cnv = sk.createCanvas(W, H);
    cnv.elt.id = 'c';
    cnv.parent('wrap');
    cnv.elt.style.position = 'absolute';
    cnv.elt.style.top = '0'; cnv.elt.style.left = '0';
    document.getElementById('wrap').style.width  = W + 'px';
    document.getElementById('wrap').style.height = H + 'px';
    sk.colorMode(sk.HSB, 360, 100, 100, 100);
    sk.noiseDetail(3, 0.5);

    const pal = buildPalette();
    paintBackground(pal);
    buildPasses(pal);
    renderMeta();
    renderLegend(pal);
    updateProgress();
  }};

  // ═══════════════════════════════════════════════════════════════
  // LOOP
  // ═══════════════════════════════════════════════════════════════
  sk.draw = function() {{
    if (done) {{ sk.noLoop(); return; }}
    const pass  = passes[passIdx];
    if (!pass)  {{ finish(); return; }}
    const batch = Math.ceil(pass.total / 50);
    const end   = Math.min(pass.drawn + batch, pass.total);
    for (let i = pass.drawn; i < end; i++) pass.fn(i / Math.max(pass.total-1, 1));
    pass.drawn = end;
    if (pass.drawn >= pass.total) {{
      passIdx++;
      if (passIdx >= passes.length) finish();
      else updateProgress();
    }}
  }};

  function finish() {{
    done = true;
    document.getElementById('progress').textContent = 'listo — pulsa guardar';
  }}

  // ═══════════════════════════════════════════════════════════════
  // FONDO — SIEMPRE oscuro, gradiente muy contenido
  // Solo un tinte sutil del momento del dia emerge desde un punto
  // Madrugada/noche: casi negro puro
  // Mediodia: negro con un halo tenue ambar en el centro
  // ═══════════════════════════════════════════════════════════════
  function paintBackground(pal) {{
    const [bh, bs, bb] = pal.bg;
    const [ph, ps, pb] = pal.primary;
    const cld  = C.cloud_norm;
    const hour = C.hour;

    // Base: negro casi puro con tinte minimo
    sk.noStroke();
    sk.background(bh, bs, bb);

    // Halo de luz — muy contenido, maximo brillo 14
    let gx, gy;
    if      (hour >= 6  && hour < 12) {{ gx = W*0.28; gy = H*0.72; }}
    else if (hour >= 12 && hour < 17) {{ gx = W*0.5;  gy = H*0.08; }}
    else if (hour >= 17 && hour < 21) {{ gx = W*0.82; gy = H*0.62; }}
    else                              {{ gx = W*0.5;  gy = H*0.5;  }}

    const maxR = Math.sqrt(W*W + H*H) * 0.75;
    for (let r = maxR; r > 0; r -= 9) {{
      const prog = 1 - r/maxR;
      // Brillo maximo del halo: 14 — nunca supera eso
      const haloBri = Math.min(14, pb * (0.08 + prog * 0.22) * (1 - cld*0.4));
      const haloSat = ps * (0.06 + prog * 0.18);
      sk.fill(ph, haloSat, haloBri, 100);
      sk.ellipse(gx, gy, r*1.55, r*1.18);
    }}

    // Grano de lienzo — textura imprimada muy sutil
    for (let i = 0; i < 90; i++) {{
      const x = sk.random(W), y = sk.random(H);
      const n = sk.noise(x*0.032, y*0.032);
      sk.fill(ph, ps*0.15, pb*(0.08+n*0.12), sk.random(0.2, 1.2));
      sk.ellipse(x, y, sk.random(2,11), sk.random(1,7));
    }}
  }}

  // ═══════════════════════════════════════════════════════════════
  // DISTRIBUCION — zonas que evitan el centro exacto
  // ═══════════════════════════════════════════════════════════════
  function zone() {{
    const z = Math.floor(sk.random(6));
    switch(z) {{
      case 0: return [sk.random(W*0.02,W*0.38), sk.random(H*0.03,H*0.42)];
      case 1: return [sk.random(W*0.58,W*0.97), sk.random(H*0.03,H*0.40)];
      case 2: return [sk.random(W*0.02,W*0.40), sk.random(H*0.58,H*0.97)];
      case 3: return [sk.random(W*0.56,W*0.97), sk.random(H*0.56,H*0.97)];
      case 4: return [sk.random(W*0.08,W*0.92), sk.random(H*0.02,H*0.18)];
      default:return [sk.random(W*0.08,W*0.92), sk.random(H*0.82,H*0.98)];
    }}
  }}

  function edge() {{
    const m = sk.random(40,130), s = Math.floor(sk.random(4));
    if (s===0) return [sk.random(W), m];
    if (s===1) return [W-m, sk.random(H)];
    if (s===2) return [sk.random(W), H-m];
    return [m, sk.random(H)];
  }}

  function free() {{
    return [sk.random(W*0.05,W*0.95), sk.random(H*0.05,H*0.95)];
  }}

  // ═══════════════════════════════════════════════════════════════
  // PRIMITIVAS — 3 capas internas, alpha bajo para capas sutiles
  // Alpha total de las formas: 8–28 (translucidas sobre negro)
  // ═══════════════════════════════════════════════════════════════

  // Triangulo — tension, filo, frio
  function triangle(x, y, size, rot, h, s, b, alpha) {{
    sk.push(); sk.translate(x,y); sk.rotate(rot);
    for (let i=0; i<3; i++) {{
      const f = 1 - i*0.2;
      sk.fill(h+i*5, Math.min(100,s*(0.9-i*0.1)), Math.min(100,b*(1+i*0.15)), alpha*(0.55-i*0.13));
      sk.noStroke();
      sk.triangle(0,-size*f, -size*0.85*f,size*0.55*f, size*0.85*f,size*0.55*f);
      if (i===0) {{
        sk.noFill();
        sk.stroke(h, s, Math.min(100,b*1.4), alpha*0.3);
        sk.strokeWeight(0.6);
        sk.triangle(0,-size, -size*0.85,size*0.55, size*0.85,size*0.55);
      }}
    }}
    sk.pop();
  }}

  // Rectangulo — estabilidad, orden, presion alta
  function rectangle(x, y, w2, h2, rot, h, s, b, alpha) {{
    sk.push(); sk.translate(x,y); sk.rotate(rot);
    for (let i=0; i<3; i++) {{
      const f = 1 - i*0.18;
      sk.fill(h+i*3, Math.min(100,s*(0.9-i*0.12)), Math.min(100,b*(1+i*0.14)), alpha*(0.55-i*0.13));
      sk.noStroke();
      sk.rect(-w2*0.5*f, -h2*0.5*f, w2*f, h2*f);
      if (i===0) {{
        sk.noFill();
        sk.stroke(h, s, Math.min(100,b*1.45), alpha*0.25);
        sk.strokeWeight(0.5);
        sk.rect(-w2*0.5, -h2*0.5, w2, h2);
      }}
    }}
    sk.pop();
  }}

  // Rombo — dinamismo, transicion
  function rhombus(x, y, size, rot, h, s, b, alpha) {{
    sk.push(); sk.translate(x,y); sk.rotate(rot);
    for (let i=0; i<3; i++) {{
      const f = 1 - i*0.2;
      sk.fill(h+i*6, Math.min(100,s*(0.88-i*0.1)), Math.min(100,b*(1+i*0.14)), alpha*(0.55-i*0.13));
      sk.noStroke();
      sk.quad(0,-size*f, size*0.65*f,0, 0,size*f, -size*0.65*f,0);
      if (i===0) {{
        sk.noFill();
        sk.stroke(h, s, Math.min(100,b*1.4), alpha*0.28);
        sk.strokeWeight(0.55);
        sk.quad(0,-size, size*0.65,0, 0,size, -size*0.65,0);
      }}
    }}
    sk.pop();
  }}

  // Circulo/Elipse — calor, expansion, irradiacion
  function circle(x, y, rx, ry, h, s, b, alpha) {{
    for (let i=0; i<4; i++) {{
      const f = 1 + i*0.30;
      sk.noStroke();
      sk.fill(h+i*3, Math.min(100,s*(0.88-i*0.12)), Math.min(100,b*(1+i*0.08)), alpha*(0.52-i*0.11));
      sk.ellipse(x+sk.random(-3,3), y+sk.random(-2,2), rx*2*f, ry*2*f);
    }}
    // Nucleo brillante
    sk.fill(h, Math.min(100,s*1.1), Math.min(100,b*1.3), alpha*0.65);
    sk.ellipse(x, y, rx*0.38, ry*0.38);
  }}

  // ═══════════════════════════════════════════════════════════════
  // PASES DE PINTURA
  // Orden: temperatura → presion → viento → PM2.5 → humedad → nubes → gesto
  // ═══════════════════════════════════════════════════════════════
  function buildPasses(pal) {{
    const t     = C.temp_norm;
    const we    = C.wind_energy;
    const pm    = C.pm_norm;
    const hum   = C.humidity_norm;
    const pnorm = C.pressure_norm;
    const cld   = C.cloud_norm;

    passes = [

      // ── PASE 1: TEMPERATURA ─────────────────────────────────────
      // Forma segun temperatura:
      //   frio  (<0.30) → triangulos agudos — tension, filo
      //   medio (<0.55) → rectangulos y rombos — equilibrio
      //   calor (>=0.55)→ circulos — expansion, calor irradia
      // Alpha bajo (8–22) para acumulacion sutil sobre negro
      {{ name:'temperatura', total:Math.floor(180+t*220), drawn:0, fn:(prog) => {{
        const [h,s,b] = prog<0.38 ? pal.dark : prog<0.72 ? pal.primary : pal.accent;
        const [x,y]   = zone();
        const rot     = sk.noise(x*0.004, y*0.004) * sk.TWO_PI;
        const size    = 14 + t*95 + sk.random(32);
        // Alpha: muy bajo para que se acumulen como veladuras
        const alpha   = 8 + t*14 + sk.random(8);
        const sat     = Math.min(100, s*(0.75+t*0.4));
        const bri     = Math.min(100, b*(0.9+prog*0.15));
        if      (t < 0.30) triangle(x, y, size, rot, h, sat, bri, alpha);
        else if (t < 0.55) sk.random()<0.5
          ? rectangle(x, y, size, size*sk.random(0.4,1.8), rot, h, sat, bri, alpha)
          : rhombus(x, y, size, rot, h, sat, bri, alpha);
        else               circle(x, y, size*sk.random(0.6,1.4), size*sk.random(0.5,1.2), h, sat, bri, alpha);
      }}}},

      // ── PASE 2: PRESION ──────────────────────────────────────────
      // Alta presion: lineas largas y paralelas (isobaras)
      // Baja presion: arcos cortos en los bordes
      {{ name:'presion', total:Math.floor(80+pnorm*120), drawn:0, fn:(prog) => {{
        const [h,s,b] = pal.neutral;
        const alpha   = 8 + pnorm*22;
        const sat     = Math.min(100, s*(0.5+pnorm*0.7));
        const bri     = Math.min(100, b*(0.6+pnorm*0.6));
        if (pnorm > 0.55) {{
          const [x,y] = zone();
          const len   = 80 + pnorm*280;
          const angle = sk.noise(x*0.003, y*0.003, 5)*sk.PI*0.6 - sk.PI*0.3;
          sk.push(); sk.translate(x,y); sk.rotate(angle);
          for (let i=0; i<3; i++) {{
            const off = (i-1)*(6+pnorm*8);
            sk.noFill();
            sk.stroke(h, sat*(0.9-i*0.15), Math.min(100,bri*(1+i*0.12)), alpha*(0.7-i*0.16));
            sk.strokeWeight(1.1+pnorm*1.4-i*0.28);
            sk.line(-len*0.5, off, len*0.5, off);
          }}
          sk.pop();
        }} else {{
          const [x,y] = edge();
          const r     = sk.random(40,140);
          const start = sk.random(sk.TWO_PI);
          sk.noFill(); sk.stroke(h, sat, bri, alpha); sk.strokeWeight(sk.random(0.7,2.3));
          sk.arc(x, y, r*2.2, r*1.4, start, start+sk.random(sk.PI*0.4,sk.PI*1.4));
        }}
      }}}},

      // ── PASE 3: VIENTO ───────────────────────────────────────────
      // Trazos en la direccion exacta del viento con fade progresivo
      {{ name:'viento', total:Math.floor(60+we*180), drawn:0, fn:(prog) => {{
        const [h,s,b] = pal.complement;
        const windAng = Math.atan2(C.wind_dy, C.wind_dx);
        const [x,y]   = free();
        const spread  = sk.PI*(we<0.3 ? 0.85 : 0.18);
        const angle   = windAng + sk.random(-spread, spread);
        const len     = we<0.15 ? sk.random(10,50) : sk.random(60,350*we);
        const alpha   = 6 + we*52;
        const sw      = we<0.1 ? sk.random(0.3,0.9) : sk.random(0.7,2.2+we*2.5);
        const sat     = Math.min(100, s*(0.4+we*0.9));
        const bri     = Math.min(100, b*(0.8+we*0.3));
        const steps   = Math.max(3, Math.floor(len/8));
        const spd     = len/steps;
        let cx=x, cy=y;
        for (let i=0; i<steps; i++) {{
          const nx=cx+Math.cos(angle)*spd, ny=cy+Math.sin(angle)*spd;
          sk.stroke(h, sat, bri, alpha*(1-i/steps*0.82));
          sk.strokeWeight(sw*(0.4+(1-i/steps)*0.65));
          sk.noFill(); sk.line(cx,cy,nx,ny);
          cx=nx; cy=ny;
        }}
      }}}},

      // ── PASE 4: PM2.5 ────────────────────────────────────────────
      // Grano de particulas — erosiona la oscuridad con polvo
      {{ name:'pm2.5', total:Math.floor(40+pm*300), drawn:0, fn:(prog) => {{
        const [h,s,b] = pal.neutral;
        if (pm<0.15 && sk.random()>pm*4) return;
        const [x,y] = free();
        const grain  = Math.floor(2+pm*20);
        const alpha  = 4 + pm*20;
        for (let i=0; i<grain; i++) {{
          const gx=x+sk.random(-60*pm,60*pm), gy=y+sk.random(-60*pm,60*pm);
          const gr=sk.random(0.5, 2+pm*4.5);
          sk.noStroke();
          sk.fill(h+sk.random(-25,25), Math.min(100,s*(0.4+pm*0.6)), b*(0.5+pm*0.5), alpha*sk.random(0.5,1.5));
          sk.random()<0.4 ? sk.rect(gx-gr,gy-gr,gr*2,gr*2) : sk.ellipse(gx,gy,gr*2,gr*1.5);
        }}
      }}}},

      // ── PASE 5: HUMEDAD ──────────────────────────────────────────
      // Velos translucidos — >65%: lluvia vertical / <65%: niebla
      {{ name:'humedad', total:Math.floor(40+hum*160), drawn:0, fn:(prog) => {{
        const [h,s,b] = pal.complement;
        if (hum<0.38 && sk.random()>hum*2.5) return;
        const [x,y] = sk.random()<0.5 ? zone() : edge();
        const alpha  = 3 + hum*14;
        const sat    = Math.min(100, s*(0.2+hum*0.5));
        const bri    = Math.min(100, b*(0.7+hum*0.35));
        if (hum>0.65) {{
          const len=15+hum*50;
          sk.stroke(h,sat,bri,alpha); sk.strokeWeight(0.4+hum*0.7); sk.noFill();
          sk.line(x, y-len*0.5, x+sk.random(-2,2), y+len*0.5);
        }} else {{
          const nv=Math.floor(2+hum*4);
          for (let i=0;i<nv;i++) {{
            const r=sk.random(30,150+hum*150);
            sk.noStroke();
            sk.fill(h+sk.random(-20,20), sat*0.6, Math.min(100,bri+6), alpha*(0.35-i*0.07));
            sk.ellipse(x+sk.random(-15,15), y+sk.random(-10,10), r*(1+i*0.4), r*0.7*(1+i*0.4));
          }}
        }}
      }}}},

      // ── PASE 6: NUBES ────────────────────────────────────────────
      // Masas redondeadas translucidas — despejado=nada, cubierto=dominante
      {{ name:'nubes', total:Math.floor(20+cld*120), drawn:0, fn:(prog) => {{
        if (cld<0.15 && sk.random()>cld*4) return;
        const [h,s,b] = pal.neutral;
        const [x,y]   = sk.random()<0.6 ? zone() : edge();
        const w2=60+cld*200+sk.random(60), h2=20+cld*80+sk.random(30);
        const alpha=4+cld*18;
        sk.noStroke();
        for (let i=0;i<4;i++) {{
          const f=1+i*0.3;
          sk.fill(h+sk.random(-10,10), s*(0.2+cld*0.3), Math.min(100,b*(0.8+cld*0.22)), alpha*(0.3-i*0.06));
          sk.ellipse(x+sk.random(-12,12), y+sk.random(-8,8), w2*f, h2*f);
        }}
      }}}},

      // ── PASE 7: GESTO FINAL ──────────────────────────────────────
      // Firma artistica del dia — formas pequenas en los bordes
      {{ name:'gesto', total:80, drawn:0, fn:(prog) => {{
        const [h,s,b] = sk.random()<0.45 ? pal.accent : pal.primary;
        const [x,y]   = edge();
        const size    = sk.random(5,24), rot=sk.random(sk.TWO_PI);
        const alpha   = sk.random(10,30);
        const sat=Math.min(100,s*1.1), bri=Math.min(100,b*1.05);
        const st=Math.floor(sk.random(5));
        if      (st===0) triangle(x,y,size,rot,h,sat,bri,alpha);
        else if (st===1) rectangle(x,y,size,size*sk.random(0.5,2),rot,h,sat,bri,alpha);
        else if (st===2) rhombus(x,y,size,rot,h,sat,bri,alpha);
        else if (st===3) circle(x,y,size,size*sk.random(0.6,1.2),h,sat,bri,alpha);
        else {{
          sk.stroke(h,sat,bri,alpha); sk.strokeWeight(sk.random(0.4,1.8)); sk.noFill();
          const len=sk.random(12,50);
          sk.line(x-Math.cos(rot)*len,y-Math.sin(rot)*len,x+Math.cos(rot)*len,y+Math.sin(rot)*len);
        }}
      }}}},
    ];
  }}

  // ═══════════════════════════════════════════════════════════════
  // UI
  // ═══════════════════════════════════════════════════════════════
  function renderMeta() {{
    document.getElementById('title').innerHTML =
      'ATMOSPHERICA &nbsp;·&nbsp; ' + C.city.toUpperCase() + ' &nbsp;·&nbsp; ' + C.date;
    document.getElementById('meta').innerHTML =
      C.temp_c.toFixed(1) + '°C &nbsp;·&nbsp; ' +
      C.pressure + ' hPa &nbsp;·&nbsp; ' +
      C.wind_speed.toFixed(1) + ' m/s ' + C.wind_dir_label +
      ' &nbsp;·&nbsp; PM2.5 ' + C.pm25.toFixed(1) +
      ' &nbsp;·&nbsp; HR ' + C.humidity + '% &nbsp;·&nbsp; ' +
      C.hour + 'h';
  }}

  function renderLegend(pal) {{
    const items = [
      {{ pal:pal.primary,    label:'temperatura ' + C.temp_c.toFixed(0) + '°C + hora ' + C.hour + 'h',
         text:'color y tipo de forma — frio=triangulos, templado=rectangulos, calor=circulos. Hora define la luminosidad.' }},
      {{ pal:pal.neutral,    label:'presion ' + C.pressure + ' hPa',
         text:'alta presion: lineas largas y paralelas. baja presion: arcos tensos en los bordes.' }},
      {{ pal:pal.complement, label:'viento ' + C.wind_speed.toFixed(1) + ' m/s ' + C.wind_dir_label,
         text:'trazos en la direccion real del viento — longitud proporcional a la velocidad.' }},
      {{ pal:pal.neutral,    label:'PM2.5 ' + C.pm25.toFixed(1),
         text:'grano disperso — puntos y cuadraditos. mas contaminacion, mas erosion visible.' }},
      {{ pal:pal.complement, label:'humedad ' + C.humidity + '%',
         text:'velos translucidos. alta humedad: lluvia vertical. baja: dia seco y nitido.' }},
      {{ pal:pal.neutral,    label:'nubes ' + Math.round(C.cloud_norm*100) + '%',
         text:'masas redondeadas que velan la luz segun la cobertura nubosa.' }},
    ];
    document.getElementById('legend').innerHTML = items.map(it => {{
      const hex = hsbHex(it.pal);
      return `<div class="leg">
        <div class="dot" style="background:${{hex}}"></div>
        <div><strong>${{it.label}}</strong><small>${{it.text}}</small></div>
      </div>`;
    }}).join('');
  }}

  function hsbHex([h,s,b]) {{
    const [hh,ss,bb]=[h/360,s/100,b/100];
    const i=Math.floor(hh*6),f=hh*6-i;
    const q=bb*(1-ss*f),t=bb*(1-ss*(1-f)),p=bb*(1-ss);
    let r,g,bv;
    switch(i%6){{case 0:r=bb;g=t;bv=p;break;case 1:r=q;g=bb;bv=p;break;
      case 2:r=p;g=bb;bv=t;break;case 3:r=p;g=q;bv=bb;break;
      case 4:r=t;g=p;bv=bb;break;default:r=bb;g=p;bv=q;}}
    return '#'+[r,g,bv].map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('');
  }}

  function updateProgress() {{
    if (passIdx < passes.length)
      document.getElementById('progress').textContent =
        'pintando ' + passes[passIdx].name + '... (' + (passIdx+1) + '/' + passes.length + ')';
  }}

}});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────

def _wind_label(deg: float) -> str:
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSO','SO','OSO','O','ONO','NO','NNO']
    return dirs[int((deg + 11.25) / 22.5) % 16]


def _day_type(we, pm, t, cld, hum):
    if we  > 0.55:              return "windy"
    if pm  > 0.45:              return "polluted"
    if t   < 0.28:              return "cold"
    if cld > 0.55 and hum>0.55: return "rainy"
    return "stable"


# ─────────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def generate_html(visual_params: dict, output_dir: str = "output") -> str:
    """
    Genera un fichero HTML con la visualizacion ATMOSPHERICA.

    Parametros esperados en visual_params:
      raw              : dict con los datos crudos de la API meteorologica
        raw.city         : str   nombre de la ciudad
        raw.temperature  : float temperatura en Celsius
        raw.pressure     : float presion en hPa
        raw.wind_speed   : float velocidad del viento m/s
        raw.wind_deg     : float direccion del viento en grados
        raw.pm2_5        : float particulas PM2.5
        raw.humidity     : int   humedad relativa %
        raw.clouds       : int   cobertura nubosa % (opcional, default 15)
      temperature_norm : float temperatura normalizada 0-1
      wind_energy      : float energia del viento normalizada 0-1
      wind_dx          : float componente X del viento (cos)
      wind_dy          : float componente Y del viento (sin)
      fragmentation    : float PM2.5 normalizado 0-1
      veil_opacity     : float humedad * 80 (se divide entre 80 internamente)
      density          : float presion normalizada 0-1

    Retorna la ruta del fichero HTML generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw   = visual_params["raw"]
    now   = datetime.now()
    date  = now.strftime("%Y-%m-%d")
    hour  = now.hour
    city  = raw["city"]

    t     = visual_params["temperature_norm"]
    we    = visual_params["wind_energy"]
    pm    = visual_params["fragmentation"]
    hum   = visual_params["veil_opacity"] / 80.0
    pnorm = visual_params["density"]
    cld   = raw.get("clouds", 15) / 100.0

    wind_dx = visual_params.get("wind_dx", math.cos(math.radians(raw.get("wind_deg", 0))))
    wind_dy = visual_params.get("wind_dy", math.sin(math.radians(raw.get("wind_deg", 0))))

    day_type = _day_type(we, pm, t, cld, hum)

    climate_data = {
        "city":           city,
        "date":           date,
        "hour":           hour,
        "temp_c":         raw["temperature"],
        "pressure":       raw["pressure"],
        "wind_speed":     raw["wind_speed"],
        "wind_dir_label": _wind_label(raw.get("wind_deg", 0)),
        "pm25":           raw["pm2_5"],
        "humidity":       raw["humidity"],
        "clouds":         raw.get("clouds", 15),
        "temp_norm":      round(t,       3),
        "pressure_norm":  round(pnorm,   3),
        "wind_energy":    round(we,      3),
        "wind_dx":        round(wind_dx, 3),
        "wind_dy":        round(wind_dy, 3),
        "pm_norm":        round(pm,      3),
        "humidity_norm":  round(hum,     3),
        "cloud_norm":     round(cld,     3),
        "day_type":       day_type,
    }

    print(f"\n  Tipo de dia  : {day_type}")
    print(f"  Temperatura  : {raw['temperature']:.1f}C  (norm {t:.2f})")
    print(f"  Hora         : {hour}h")
    print(f"  Presion      : {raw['pressure']} hPa  (norm {pnorm:.2f})")
    print(f"  Viento       : {raw['wind_speed']:.1f} m/s {_wind_label(raw.get('wind_deg',0))}")
    print(f"  PM2.5        : {raw['pm2_5']:.1f}")
    print(f"  Humedad      : {raw['humidity']}%")
    print(f"  Nubes        : {raw.get('clouds',15)}%")

    html = HTML_TEMPLATE.format(
        climate_json=json.dumps(climate_data, indent=2),
    )

    fname = f"atmospherica_{city.replace(' ','_')}_{date}_{hour:02d}h.html"
    path  = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  HTML         : {path}")
    return path