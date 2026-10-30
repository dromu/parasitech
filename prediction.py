import os
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "Protozoo": BASE_DIR / "efficiennetB5_90perce.pkl",
    "Helminto": BASE_DIR / "vit_small_patch16_224_Helmintos.pkl",
}


class ModelUnavailableError(RuntimeError):
    pass


@st.cache_resource(show_spinner="Cargando modelo de IA (solo la primera vez)…", max_entries=len(MODELS))
def load_model(parasite):
    """Carga el modelo una sola vez por servidor y lo comparte entre usuarios.

    Los modelos se cargan bajo demanda: si nadie usa el de helmintos no ocupa
    memoria.
    """
    if parasite not in MODELS:
        raise ValueError("Tipo de parásito no reconocido. Usa 'Protozoo' o 'Helminto'.")

    path = MODELS[parasite]
    # Un archivo de pocos bytes es el puntero de Git LFS, no el modelo real.
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        raise ModelUnavailableError(
            f"No se encontró el modelo '{path.name}'. Descárgalo con `git lfs pull`."
        )

    import torch
    from fastai.vision.all import load_learner

    # Evita que torch abra un hilo por núcleo en servidores con pocos recursos.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    learn = load_learner(path, cpu=True)
    learn.model.eval()
    return learn


def predictParasite(parasite, image, boxes, top_k=3):
    """Clasifica cada región de `image` delimitada por `boxes` en un solo lote.

    `boxes` es una lista de (x1, y1, x2, y2) en píxeles de `image`.
    Devuelve, por región, una lista con las `top_k` clases más probables como
    tuplas (clase, probabilidad 0-1) ordenadas de mayor a menor.
    """
    if not boxes:
        return []

    learn = load_model(parasite)
    crops = [image.crop(box) for box in boxes]

    dl = learn.dls.test_dl(crops, num_workers=0)
    with learn.no_bar(), learn.no_logging():
        probs, _ = learn.get_preds(dl=dl)

    vocab = list(learn.dls.vocab)
    k = min(top_k, len(vocab))
    values, indices = probs.topk(k, dim=1)

    return [
        [(str(vocab[i]), float(v)) for v, i in zip(row_v.tolist(), row_i.tolist())]
        for row_v, row_i in zip(values, indices)
    ]
