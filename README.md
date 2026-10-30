# ParasiTech

Herramienta web de apoyo al diagnóstico que identifica, con inteligencia artificial, protozoos y
helmintos intestinales en imágenes de microscopía. El profesional marca con recuadros las
estructuras sospechosas y el modelo clasifica cada una.

> Los resultados son orientativos y deben ser confirmados por un profesional de laboratorio clínico.

## Uso

1. Sube una imagen del campo del microscopio (PNG o JPG).
2. Elige el tipo de parásito: **Protozoo** (EfficientNet-B5) o **Helminto** (ViT small).
3. Dibuja un recuadro sobre cada estructura. Con «Mover / ajustar» puedes corregirlos.
4. Pulsa **Identificar**. La confianza mínima se puede cambiar después sin volver a analizar.
5. Descarga la imagen anotada y la tabla de resultados (CSV).

## Ejecución local

Los modelos (`*.pkl`) se guardan con Git LFS:

```bash
git lfs pull
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run parasiTech.py
```

Se recomienda Python 3.11 o 3.12. `requirements.txt` instala PyTorch solo para CPU.

## Estructura

| Archivo | Descripción |
| --- | --- |
| `parasiTech.py` | Aplicación Streamlit (interfaz). |
| `prediction.py` | Carga de los modelos (en caché, bajo demanda) y clasificación por lotes. |
| `contorCircle.py` | Detección y recorte del campo circular del microscopio. |
| `.streamlit/config.toml` | Tema y ajustes del servidor. |
