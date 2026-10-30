import cv2
import numpy as np
from PIL import Image

# Si el contorno más grande ocupa menos de esta fracción de la imagen se
# considera ruido y se conserva la imagen completa.
MIN_AREA_RATIO = 0.05


def countourCircle(imagenPil, umbralMen):
    """Recorta el campo circular del microscopio y enmascara el fondo.

    Si no se encuentra un contorno útil devuelve la imagen original en RGB,
    en lugar de fallar.
    """
    imagenPil = imagenPil.convert("RGB")
    image = np.asarray(imagenPil)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un umbral para crear una imagen binaria
    _, thresholded = cv2.threshold(blurred, umbralMen, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return imagenPil

    # Seleccionar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_AREA_RATIO * gray.size:
        return imagenPil

    # Crear una máscara con el contorno más grande y aplicarla a la imagen
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Recortar el rectángulo delimitador del contorno más grande
    x, y, w, h = cv2.boundingRect(largest_contour)
    result = masked_image[y:y + h, x:x + w]

    return Image.fromarray(result)
