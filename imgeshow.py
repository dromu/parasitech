from PIL import Image, ImageDraw, ImageFont

# Cargar una imagen desde el archivo
img = Image.open("IMG_20240826_091306342.jpg")

# Redimensionar la imagen (opcional)
# img = img.resize((500, 500))  # Cambia el tamaño si es necesario

# Crear un objeto para dibujar en la imagen
draw = ImageDraw.Draw(img)

# Definir la fuente y tamaño del texto
font_size = 100
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

# Definir el texto y las coordenadas donde se mostrará
texto = "Hola, Mundo!"
x, y = 1000,1000 # Coordenadas para el texto

# Dibujar el texto en la imagen
draw.text((x, y), texto, font=font, fill="black")

# Guardar la imagen modificada
img.save("imagen_con_texto3.jpg")

# Mostrar la imagen
img.show()
