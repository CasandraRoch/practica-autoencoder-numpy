# Autoencoder desde Cero con NumPy

Implementación manual de un autoencoder de 3 capas para comprimir 
y reconstruir imágenes RGB de 8×8 píxeles, incluyendo prueba de 
eliminación de ruido (denoising).

## ¿Qué hace este proyecto?

1. Genera una imagen RGB aleatoria de 8×8 píxeles
2. Entrena un autoencoder con backpropagation manual hasta que el 
   Error Cuadrático Medio (ECM) sea menor a 0.001
3. Reconstruye la imagen desde el espacio latente comprimido
4. Aplica ruido gaussiano a la imagen original y prueba si el 
   autoencoder logra reconstruirla limpia

## Arquitectura

- Función de activación: Sigmoide
- Optimización: Gradiente descendente + Momentum (α=0.7, δ=0.7)
- Criterio de parada: ECM < 0.001 o 20,000 épocas

## Tecnologías

- Python 3
- NumPy — álgebra lineal y entrenamiento manual
- Matplotlib — visualización de imágenes y curva de aprendizaje

## Aprendizajes

- Implementación de backpropagation sin frameworks de deep learning
- Uso de momentum para acelerar la convergencia
- Normalización de datos de imagen para redes neuronales
- Concepto de espacio latente y compresión de información
