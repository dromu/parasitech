import hashlib
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

from contorCircle import countourCircle
from prediction import ModelUnavailableError, predictParasite

BASE_DIR = Path(__file__).resolve().parent
CANVAS_SIZE = 750       # Tamaño del lienzo; las regiones se recortan en esta escala
PROCESS_SIZE = 2100     # Resolución a la que se detecta el campo del microscopio
MIN_BOX = 8             # Lado mínimo (px) para considerar válida una región
NOT_IDENTIFIED = "No identificado"
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40),
]
TOOLS = {"Dibujar recuadros": "rect", "Mover / ajustar": "transform"}


st.set_page_config(
    page_title="ParasiTech · Detección de parásitos con IA",
    page_icon="🔬",
    layout="wide",
    menu_items={"About": "ParasiTech: apoyo al diagnóstico de parasitosis intestinales con IA."},
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1200px;}
        [data-testid="stSidebar"] img {margin-bottom: -1rem;}
        .pt-hero h1 {margin-bottom: 0; color: #4A1F4E;}
        .pt-hero p {color: #555; margin-top: .25rem; font-size: 1.05rem;}
        .pt-step {
            background: #F6F1F7; border-left: 4px solid #6B2C6F; border-radius: 6px;
            padding: .6rem .9rem; margin-bottom: .5rem; font-size: .95rem;
        }
        .pt-legend span {
            display: inline-block; padding: .15rem .6rem; margin: 0 .4rem .4rem 0;
            border-radius: 999px; color: #fff; font-size: .85rem; font-weight: 600;
        }
        .pt-footer {
            margin-top: 3rem; padding: 1rem; border-top: 1px solid #ddd;
            text-align: center; font-size: .85rem; color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Estado de la sesión
# --------------------------------------------------------------------------
defaults = {
    "accepted": False,
    "dialog_shown": False,
    "user": {},
    "image_key": None,      # Identifica la imagen + umbral ya procesados
    "work_image": None,     # Imagen 750x750 que ve el médico y usa el modelo
    "full_image": None,     # Imagen procesada en alta resolución (para descargar)
    "results": None,        # Resultados de la última identificación
    "canvas_version": 0,    # Cambiarlo reinicia el lienzo (borra los recuadros)
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


@st.dialog("Bienvenido a ParasiTech", width="large")
def terms_dialog():
    st.write(
        "Al utilizar esta aplicación, aceptas que la información y los datos que suministres "
        "podrán ser utilizados para mejorar la funcionalidad y el rendimiento del sistema. "
        "Nos comprometemos a proteger tu privacidad y a utilizar tus datos de manera ética y "
        "responsable, de acuerdo con las leyes de protección de datos vigentes."
    )
    st.info(
        "Estamos en la etapa de validación y agradeceríamos tu colaboración "
        "proporcionando la siguiente información:"
    )
    with st.form("terms_form", border=False):
        name = st.text_input("Nombre")
        c1, c2 = st.columns(2)
        profession = c1.text_input("Profesión")
        institution = c2.text_input("Institución")
        acepto = st.checkbox("He leído y acepto los términos")
        submitted = st.form_submit_button("Comenzar", type="primary", use_container_width=True)

    if submitted:
        if not (name.strip() and profession.strip() and institution.strip()):
            st.error("Por favor, completa todos los campos.")
        elif not acepto:
            st.error("Debes aceptar los términos para continuar.")
        else:
            st.session_state.accepted = True
            st.session_state.user = {
                "name": name.strip(), "profession": profession.strip(),
                "institution": institution.strip(),
            }
            st.rerun()


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def load_uploaded_image(uploaded, threshold):
    """Procesa la imagen solo cuando cambia el archivo o el umbral.

    Antes se redimensionaba y recortaba la imagen en cada interacción con el
    lienzo, lo que saturaba CPU y memoria del servidor.
    """
    data = uploaded.getvalue()
    key = (hashlib.md5(data).hexdigest(), threshold)
    if st.session_state.image_key == key:
        return

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        st.session_state.image_key = None
        st.session_state.work_image = None
        st.session_state.full_image = None
        raise

    img = img.resize((PROCESS_SIZE, PROCESS_SIZE), Image.LANCZOS)
    full = countourCircle(img, threshold)
    del img

    # Si cambió la imagen (no solo el umbral) se borran recuadros y resultados.
    same_file = st.session_state.image_key and st.session_state.image_key[0] == key[0]
    if not same_file:
        st.session_state.canvas_version += 1
    st.session_state.results = None
    st.session_state.image_key = key
    st.session_state.full_image = full
    st.session_state.work_image = full.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)


def boxes_from_canvas(json_data):
    """Extrae los recuadros dibujados como (x1, y1, x2, y2) en píxeles del lienzo.

    Tiene en cuenta el escalado aplicado con la herramienta "Mover / ajustar".
    """
    boxes = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") != "rect":
            continue
        w = obj["width"] * obj.get("scaleX", 1)
        h = obj["height"] * obj.get("scaleY", 1)
        x1 = max(0, min(CANVAS_SIZE, obj["left"]))
        y1 = max(0, min(CANVAS_SIZE, obj["top"]))
        x2 = max(0, min(CANVAS_SIZE, obj["left"] + w))
        y2 = max(0, min(CANVAS_SIZE, obj["top"] + h))
        if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
            boxes.append(tuple(round(v) for v in (x1, y1, x2, y2)))
    return boxes


def get_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def label_of(top, threshold):
    cls, prob = top[0]
    return cls if prob * 100 >= threshold else NOT_IDENTIFIED


def color_map(labels):
    colors = {}
    for label in labels:
        if label == NOT_IDENTIFIED:
            colors[label] = (120, 120, 120)
        elif label not in colors:
            colors[label] = PALETTE[len([c for c in colors if c != NOT_IDENTIFIED]) % len(PALETTE)]
    return colors


def annotate(image, boxes, labels, probs, colors):
    """Dibuja recuadros numerados con etiqueta legible sobre una copia de `image`."""
    img = image.copy()
    scale = img.width / CANVAS_SIZE
    draw = ImageDraw.Draw(img)
    font = get_font(max(12, int(13 * scale)))
    line = max(2, int(3 * scale))

    for n, (box, label, prob) in enumerate(zip(boxes, labels, probs), start=1):
        x1, y1, x2, y2 = (v * (img.width if i % 2 == 0 else img.height) / CANVAS_SIZE
                          for i, v in enumerate(box))
        color = colors[label]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)

        text = f"{n}. {label}" + (f" ({prob:.0f}%)" if label != NOT_IDENTIFIED else "")
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = int(3 * scale)
        ty = y2 + pad if y2 + th + 3 * pad < img.height else max(0, y1 - th - 3 * pad)
        tx = min(max(0, x1), img.width - tw - 2 * pad)
        draw.rectangle((tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad), fill=color)
        draw.text((tx + pad, ty + pad - tb[1]), text, fill="white", font=font)
    return img


def to_jpeg(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Aceptación de términos
# --------------------------------------------------------------------------
st.sidebar.image(str(BASE_DIR / "parasitechLogo.png"), use_column_width=True)

if not st.session_state.accepted:
    st.markdown(
        '<div class="pt-hero"><h1>🔬 ParasiTech</h1>'
        "<p>Identificación asistida por inteligencia artificial de protozoos y helmintos "
        "intestinales en imágenes de microscopía.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.info("Para usar la herramienta, completa el registro y acepta los términos de uso.")
    if not st.session_state.dialog_shown:
        st.session_state.dialog_shown = True
        terms_dialog()
    elif st.button("Registrarme y comenzar", type="primary"):
        terms_dialog()
    st.stop()


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.user.get("name"):
        st.caption(f"👤 {st.session_state.user['name']} · {st.session_state.user['institution']}")

    st.header("Configuración")
    parasite = st.radio(
        "Tipo de parásito a identificar",
        ("Protozoo", "Helminto"),
        horizontal=True,
        help="Se usa un modelo de IA distinto para cada grupo.",
    )
    tool = st.radio(
        "Herramienta del lienzo",
        list(TOOLS),
        help="Dibuja un recuadro alrededor de cada estructura sospechosa. "
             "Con «Mover / ajustar» puedes corregir su posición y tamaño.",
    )
    confidence = st.slider(
        "Confianza mínima (%)", 60, 100, 90,
        help="Las regiones con una probabilidad menor se marcan como «No identificado».",
    )
    with st.expander("Ajustes avanzados"):
        contour = st.slider(
            "Umbral de recorte del campo", 0, 255, 90,
            help="Controla la detección del campo circular del microscopio. "
                 "Ajústalo si la imagen aparece recortada de forma incorrecta.",
        )

    st.divider()
    st.markdown("**Cómo usar**")
    st.markdown(
        "1. Sube una imagen de microscopía.\n"
        "2. Selecciona protozoo o helminto.\n"
        "3. Dibuja un recuadro sobre cada estructura.\n"
        "4. Pulsa **Identificar**.\n"
        "5. Revisa y descarga el informe."
    )


# --------------------------------------------------------------------------
# Contenido principal
# --------------------------------------------------------------------------
st.markdown(
    '<div class="pt-hero"><h1>🔬 ParasiTech</h1>'
    "<p>Identificación asistida por IA de parásitos intestinales en microscopía.</p></div>",
    unsafe_allow_html=True,
)
st.caption(
    "⚕️ Herramienta de apoyo al diagnóstico. Los resultados deben ser confirmados por "
    "un profesional de laboratorio clínico y no sustituyen el criterio médico."
)

uploaded = st.file_uploader(
    "Imagen de microscopía", type=["png", "jpg", "jpeg"],
    help="Fotografía del campo del microscopio (PNG o JPG).",
)

if uploaded is None:
    st.session_state.image_key = None
    st.session_state.work_image = None
    st.session_state.full_image = None
    st.session_state.results = None
    st.markdown(
        '<div class="pt-step">📤 <b>Paso 1.</b> Sube una imagen para comenzar.</div>'
        '<div class="pt-step">🖱️ <b>Paso 2.</b> Dibuja recuadros sobre las estructuras '
        "sospechosas.</div>"
        '<div class="pt-step">🧠 <b>Paso 3.</b> Pulsa <b>Identificar</b> y revisa los '
        "resultados.</div>",
        unsafe_allow_html=True,
    )
else:
    try:
        with st.spinner("Preparando imagen…"):
            load_uploaded_image(uploaded, contour)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        st.error("No se pudo leer la imagen. Verifica que sea un archivo PNG o JPG válido.")
        st.stop()

    actions = st.container()

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.25)",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#EEEEEE",
        background_image=st.session_state.work_image,
        update_streamlit=True,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode=TOOLS[tool],
        key=f"canvas_{st.session_state.canvas_version}",
        display_toolbar=True,
    )
    boxes = boxes_from_canvas(canvas_result.json_data)

    with actions:
        c1, c2, c3 = st.columns([2, 1, 3])
        run = c1.button(
            f"🔍 Identificar ({len(boxes)} {'región' if len(boxes) == 1 else 'regiones'})",
            type="primary", disabled=not boxes, use_container_width=True,
        )
        if c2.button("🗑️ Limpiar", use_container_width=True, disabled=not boxes):
            st.session_state.canvas_version += 1
            st.session_state.results = None
            st.rerun()
        if not boxes:
            c3.caption("Dibuja al menos un recuadro sobre la imagen para habilitar la identificación.")
        else:
            c3.caption("Usa la herramienta «Mover / ajustar» para corregir un recuadro.")

    if run:
        try:
            with st.spinner(f"Analizando {len(boxes)} región(es) con el modelo de {parasite.lower()}s…"):
                preds = predictParasite(parasite, st.session_state.work_image, boxes)
            st.session_state.results = {
                "parasite": parasite,
                "boxes": boxes,
                "preds": preds,
                "file": uploaded.name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        except ModelUnavailableError as e:
            st.error(str(e))
        except Exception:
            st.error("Ocurrió un error al analizar la imagen. Intenta de nuevo.")

    res = st.session_state.results
    if res:
        if res["boxes"] != boxes or res["parasite"] != parasite:
            st.warning("La selección o el tipo de parásito cambió. Pulsa **Identificar** "
                       "para actualizar los resultados.")

        labels = [label_of(p, confidence) for p in res["preds"]]
        top_probs = [p[0][1] * 100 for p in res["preds"]]
        colors = color_map(labels)

        st.subheader("Resultados")
        identified = sum(lbl != NOT_IDENTIFIED for lbl in labels)
        m1, m2, m3 = st.columns(3)
        m1.metric("Regiones analizadas", len(labels))
        m2.metric("Identificadas", identified)
        m3.metric("Sin identificar", len(labels) - identified,
                  help=f"Probabilidad por debajo del {confidence}%.")

        legend = "".join(
            f'<span style="background: rgb{colors[lbl]}">{lbl}: {labels.count(lbl)}</span>'
            for lbl in dict.fromkeys(sorted(labels, key=lambda x: x == NOT_IDENTIFIED))
        )
        st.markdown(f'<div class="pt-legend">{legend}</div>', unsafe_allow_html=True)

        rows = []
        for n, (box, lbl, pred) in enumerate(zip(res["boxes"], labels, res["preds"]), start=1):
            rows.append({
                "N.º": n,
                "Predicción": lbl,
                "Probabilidad (%)": round(pred[0][1] * 100, 2),
                "Clase más probable": pred[0][0],
                "Alternativas": ", ".join(f"{c} ({p * 100:.1f}%)" for c, p in pred[1:]),
                "Coordenadas (x1, y1, x2, y2)": str(box),
            })
        df = pd.DataFrame(rows)

        # Las imágenes anotadas solo se regeneran si cambian las etiquetas, no en
        # cada interacción con el lienzo.
        if res.get("rendered_for") != labels:
            res["preview"] = to_jpeg(annotate(st.session_state.work_image, res["boxes"],
                                              labels, top_probs, colors))
            res["download"] = to_jpeg(annotate(st.session_state.full_image, res["boxes"],
                                               labels, top_probs, colors), quality=92)
            res["rendered_for"] = labels

        tab_img, tab_table = st.tabs(["🖼️ Imagen anotada", "📋 Detalle por región"])
        with tab_img:
            st.image(res["preview"], caption="Reconocimiento de parásitos", width=CANVAS_SIZE)
        with tab_table:
            st.dataframe(
                df, hide_index=True, use_container_width=True,
                column_config={
                    "Probabilidad (%)": st.column_config.ProgressColumn(
                        "Probabilidad (%)", min_value=0, max_value=100, format="%.1f%%"),
                },
            )

        base = f"parasitech_{res['time'].replace(' ', '_').replace(':', '')}"
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button(
            "⬇️ Descargar imagen",
            data=res["download"],
            file_name=f"{base}.jpg", mime="image/jpeg", use_container_width=True,
        )
        report = df.assign(Archivo=res["file"], Modelo=res["parasite"], Fecha=res["time"],
                           **{"Confianza mínima (%)": confidence})
        d2.download_button(
            "⬇️ Descargar tabla (CSV)",
            data=report.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{base}.csv", mime="text/csv", use_container_width=True,
        )


st.markdown(
    """
    <div class="pt-footer">
        © 2024 ParasiTech - Todos los derechos reservados.<br>
        Desarrollado por
        <a href="https://orcid.org/0009-0001-9428-4192" target="_blank">Jader Muñoz</a>,
        <a href="https://orcid.org/0000-0003-3538-6313" target="_blank">Reinel Vasquez</a> y
        <a href="https://orcid.org/0000-0003-1548-942X" target="_blank">Rubiel Vargas</a><br>
        Financiado por la "Implementación del Proyecto Jóvenes Investigadores e Innovadores en el
        Departamento del Cauca" BPIN 2020000100043
    </div>
    """,
    unsafe_allow_html=True,
)
