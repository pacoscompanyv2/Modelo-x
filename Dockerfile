FROM python:3.9-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y datos del proyecto
COPY . .

# Ejecutar el entrenamiento dentro del contenedor para generar el .joblib
RUN python model/train.py

# Exponer el puerto 80 y arrancar la API con Uvicorn
EXPOSE 90
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "90"]
