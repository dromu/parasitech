from fastai.vision.all import *
from PIL import Image
from pathlib import Path
import cv2
import time

# Cargar el modelo globalmente para evitar cargarlo repetidamente
pathModel = Path("efficiennetB5_90perce.pkl")
learn = load_learner(pathModel)


pathModelH = Path("vit_small_patch16_224_Helmintos.pkl")
learnHel = load_learner(pathModelH)

def predictParasite(parasite, image, coordinates): 
    # Abrir la imagen
    # image = Image.open(imagePath)

    # Descomponer las coordenadas
    left, top, width, height = coordinates
    x1, y1 = left, top
    x2, y2 = left + width, top + height

    # Redimensionar la imagen
    resized_image = image.resize((750, 750), Image.LANCZOS)

    # Recortar la región de interés
    img = resized_image.crop((x1, y1, x2, y2))


    # Realizar la predicción según el tipo de parásito
    if parasite == "Protozoo": 
        pred_class, pred_idx, outputs = learn.predict(img)
        return pred_class, pred_idx, outputs  # Devolver el resultado de la predicción

    elif parasite == "Helminto":
        pred_class, pred_idx, outputs = learnHel.predict(img)
        return pred_class, pred_idx, outputs  # Devolver el resultado de la predicción

    else:
        raise ValueError("Tipo de parásito no reconocido. Usa 'protozoo' o 'helminto'.")


