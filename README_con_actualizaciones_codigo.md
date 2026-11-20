## 4. Adaptaciones realizadas en Windows

El proyecto original se adaptó para ejecutarse en **Windows, Python 3.12 y JupyterLab**, conservando los nombres y la estructura general de los archivos. Las instrucciones originales para macOS se mantienen en las secciones anteriores.

### 4.1 `encode_faces.py`

El código original no fue modificado. Se actualizaron las imágenes del directorio `dataset` y se ejecutó nuevamente el entrenamiento, obteniendo **270 codificaciones faciales**. El archivo actualizado se guardó como `encodings.pickle`.

### 4.2 `recognize_faces_image.py`

- **Compatibilidad con NumPy:** se incorporó una clase basada en `pickle.Unpickler` para adaptar las referencias internas de NumPy al cargar codificaciones generadas con otra versión de la biblioteca.
- **Comparación facial:** se calcularon las distancias entre el rostro detectado y las codificaciones almacenadas. Después, se agruparon las distancias por persona, se seleccionaron las tres menores de cada grupo y se calculó su promedio. La identidad con el promedio más bajo se consideró la mejor candidata.
- **Validación de identidad:** se asignó el nombre únicamente cuando el promedio fue menor o igual a `0.48` y la diferencia respecto al segundo mejor candidato fue de al menos `0.055`. En caso contrario, se utilizó `Unknown`.
- **Conversión de colores:** se conservó la transformación original de BGR a RGB mediante `cv2.cvtColor()`.
- **Visualización:** se cambió la ventana a `WINDOW_NORMAL` y se ajustó su tamaño para visualizar fotografías grandes. También se diferenciaron los rostros reconocidos y `Unknown` mediante recuadros verdes y rojos.

### 4.3 `recognize_faces_video.py`

- **Compatibilidad con NumPy:** se incorporó la misma adaptación de carga de `encodings.pickle` utilizada en el programa de imágenes.
- **Cambio en la clasificación:** se reemplazó la votación basada en la cantidad de coincidencias por una comparación de distancias. Para cada persona se tomaron sus tres codificaciones más cercanas al rostro analizado y se calculó el promedio de las distancias. Se seleccionó la identidad con el menor promedio, siempre que cumpliera el umbral `0.48` y el margen `0.055`.
- **Procesamiento de fotogramas:** se mantuvo el redimensionamiento a 750 píxeles de ancho y la conversión posterior de BGR a RGB.
- **Visualización:** se conservaron la captura mediante `VideoStream` y las opciones de salida. Se añadieron colores diferentes para los recuadros de identidades reconocidas y `Unknown`.

### 4.4 `recognize_faces_video_file.py`

- **Compatibilidad con NumPy:** se incorporó una clase de carga compatible para leer el archivo de codificaciones entre versiones de NumPy.
- **Corrección de BGR a RGB:** el código original redimensionaba el fotograma, pero convertía a RGB la imagen sin redimensionar. Se corrigió para que la conversión se realice sobre el mismo fotograma redimensionado que se utiliza en la detección facial.
- **Resolución de análisis:** se incorporó el parámetro `--width`, con un valor predeterminado de 900 píxeles, evitando ampliar los videos que ya tengan un ancho menor.
- **Comparación facial:** se agruparon las distancias por persona y se calculó el promedio de las tres menores. Se asignó la identidad con el mejor promedio cuando cumplió el umbral `0.48` y el margen mínimo `0.055`.
- **Seguimiento de rostros:** se añadieron asociaciones entre fotogramas utilizando la posición del rostro y la distancia entre sus codificaciones faciales. Esto permite relacionar detecciones consecutivas con un mismo rostro.
- **Continuidad temporal:** se incorporó una retención de identidad de hasta `0.35` segundos, condicionada a que la comparación facial siga siendo compatible con la identidad previamente reconocida.
- **Generación de videos:** se utilizó el códec `mp4v` para archivos MP4 y se obtuvo la cantidad de fotogramas por segundo (FPS) del video de entrada para conservarla en el archivo generado.
- **Visualización y guardado:** se mantuvieron los parámetros originales `--input`, `--output` y `--display`, permitiendo generar el video sin mostrar la ventana de procesamiento.

### 4.5 Archivos conservados

- `search_bing_api.py`: sin modificaciones.
- `requirements.txt`: sin modificaciones en los códigos adaptados.
- Los comentarios originales en inglés se conservaron cuando seguían siendo correctos y se actualizaron únicamente los correspondientes a las partes modificadas.

### 4.6 Comandos de ejecución en JupyterLab

**Reconocimiento en imágenes:**

```python
%run recognize_faces_image.py --encodings encodings.pickle --image examples/Karla01.JPG
```

**Reconocimiento con cámara web:**

```python
%run recognize_faces_video.py --encodings encodings.pickle
```

**Reconocimiento en videos grabados:**

```python
%run recognize_faces_video_file.py --encodings encodings.pickle --input videos/Karla.mp4 --output output/Karla_nuevo_entrenamiento.mp4 --display 0
```

Para procesar otros videos, únicamente se cambian las rutas de entrada (`--input`) y salida (`--output`).