# Reconocimiento facial con OpenCV y Deep Learning

Guía de instalación, ejecución y evaluación del proyecto `face_recognition`.

Reconocimiento facial con `face_recognition` (dlib) y OpenCV sobre imágenes, videos grabados y cámara web. Permite identificar personas registradas y mostrar la etiqueta `Desconocido` cuando un rostro detectado no cumple los criterios de aceptación.

Configuraciones documentadas: macOS (Apple Silicon, Python 3.14) y Windows (Python 3.12, JupyterLab).
## Tutorial de referencia
[PyImageSearch — Face Recognition](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
---

## 1. Instalación de librerías

### 1.1 Activar el entorno virtual

```bash
cd "{path}/face-recognition-opencv"
source .envproject/bin/activate
```

Si no existe el entorno, se crea con `python3 -m venv .envproject`.
O también con `virtualenv .envproject`.
### 1.2 Instalar las dependencias

```bash
pip install -r requirements.txt
pip install --no-deps face_recognition==1.3.0 face_recognition_models==0.3.0
```

### 1.3 Por qué son dos instalaciones (explicación; no requiere ejecutar comandos)

El segundo comando va aparte y con `--no-deps` por tres razones:

| Motivo | Explicación |
|---|---|
| `dlib-bin` en vez de `dlib` | PyPI no publica wheels de `dlib`, solo código fuente: instalarlo implica compilar con cmake (varios minutos). `dlib-bin` es el mismo dlib 20.0.1 ya compilado. |
| `--no-deps` | `face_recognition` pide `dlib` **por nombre**. Como lo instalado se llama `dlib-bin`, pip no lo da por satisfecho y se baja `dlib-20.0.1.tar.gz` para compilarlo. El flag se lo impide. |
| `--no-deps` no cabe en el archivo | Es una opción del comando `pip install`, no de una línea de `requirements.txt` (pip responde `Invalid requirement: --no-deps`). Por eso es un comando separado. |

`face_recognition_models` se nombra explícito porque, al cortar la resolución con `--no-deps`, pip ya no lo instalaría solo y el import fallaría con
`Please install 'face_recognition_models'...`.

### 1.4 Paquetes que quedan instalados

| Paquete | Para qué |
|---|---|
| `opencv-python` | cv2: imágenes, video, ventanas, dibujo |
| `face_recognition` | detección y encoding de rostros |
| `face_recognition_models` | modelos preentrenados de dlib |
| `dlib-bin` | motor de dlib, precompilado |
| `imutils` | `paths`, `resize`, `VideoStream` |
| `numpy`, `pillow` | operaciones con los encodings e imágenes |
| `setuptools<81` | aporta `pkg_resources` (eliminado en setuptools 81) |
| `requests` | solo para `search_bing_api.py` |
| `pandas`, `scikit-learn`, `matplotlib` | métricas y gráficos de `evaluar.ipynb` |

### 1.5 Verificar que quedó bien

```bash
python -c "import cv2, dlib, face_recognition, imutils; print('OK')"
```

---

## 2. Las tres maneras de correrlo

Todas parten del entorno activado y usan el archivo `encodings.pickle` que ya viene en el proyecto o generado con `encode_faces.py`.

Los cuatro scripts tienen **valores por defecto**, así que se pueden correr sin ningún argumento.
Los valores predeterminados se aplican cuando no se indican argumentos explícitos.

| Script | `-e` | `-i` | `-d` |
|---|---|---|---|
| `recognize_faces_image.py` | `encodings.pickle` | `examples/example_01.png` | `hog` |
| `recognize_faces_video_file.py` | `encodings.pickle` | `videos/lunch_scene.mp4` | `hog` |
| `recognize_faces_video.py` | `encodings.pickle` |  (webcam - default) | `hog` |
| `encode_faces.py` | `encodings.pickle` (salida de vectores 128) | `dataset` | `hog` |

> **Sobre `-d`:** el proyecto utiliza `hog` por defecto para reducir el costo computacional.
> El detector `cnn` puede utilizarse como alternativa, pero suele resultar más exigente sin aceleración GPU;
> su velocidad depende del hardware, del tamaño de la imagen y de la configuración.

### 2.1 Sobre una imagen

```bash
python recognize_faces_image.py                              # usa los defaults
python recognize_faces_image.py -i examples/example_03.png   # otra imagen
```

Abre una ventana con las caras marcadas. Para cerrarla: **haz clic sobre la ventana** (para darle
el foco del teclado) y presiona **cualquier tecla**.

Las imágenes de prueba se encuentran en: `examples/example_01.png`, `example_02.png`, `example_03.png`.

| Parámetro | Significado |
|---|---|
| `-e` | archivo de encodings a usar |
| `-i` | imagen de entrada |
| `-d` | método de detección: `hog` (rápido) o `cnn` (preciso, lento) |

### 2.2 Sobre un archivo de video

```bash
python recognize_faces_video_file.py                         # usa los defaults

# o especificando entrada y salida:
python recognize_faces_video_file.py \
  -i videos/lunch_scene.mp4 \
  -o output/lunch_scene_salida.avi -y 1
```

| Parámetro | Significado |
|---|---|
| `-i` | video de entrada |
| `-o` | video anotado de salida (opcional) |
| `-y 1` | mostrar ventana mientras procesa (`-y 0` para ocultarla) |


### 2.3 En vivo con la webcam

```bash
python recognize_faces_video.py                              # solo mostrar
python recognize_faces_video.py -o output/webcam.avi         # además grabar
```
---

## 3. Generar encodings de nuevas imágenes

### 3.1 Cómo funciona

`encode_faces.py` recorre las subcarpetas de `dataset/`, convierte cada rostro en un vector de
128 dimensiones y guarda todos los vectores en un `.pickle` junto al nombre de la persona.
**El nombre de la carpeta es la etiqueta**: lo que aparece sobre la cara al reconocer.

```
dataset/
├── alan_grant/      → se etiqueta "alan_grant"
├── claire_dearing/
├── ellie_sattler/
├── ian_malcolm/
├── john_hammond/
├── Karla_Abarca/
├── Marco_Guapi/
└── owen_grady/
```

### 3.2 Agregar una persona nueva

1. Crear la carpeta con el nombre que quieres que aparezca:

2. Copiar ahí sus fotos. 
Requisitos:
   - Entre 10 y 50 fotos por persona;
   - Una sola cara visible por foto, de frente y con buena luz;
   - Variedad de ángulos, expresiones y fondos mejora la precisión.

### 3.3 Regenerar el archivo de encodings

```bash
python encode_faces.py #Default
python encode_faces.py -i dataset -e encodings.pickle -d hog #Con parametros
```

| Parámetro | Significado |
|---|---|
| `-i` | carpeta raíz del dataset |
| `-e` | archivo `.pickle` de salida |
| `-d` | `hog` (rápido) o `cnn` (más preciso, mucho más lento) |

---

## 4. Evaluar los encodings

La evaluación está en el notebook `evaluar.ipynb`. Genera los embeddings directamente desde `dataset/`
(no lee `encodings.pickle` para calcular las métricas de esa evaluación) y calcula métricas, matrices de confusión y gráficos.

### 4.1 Qué hace el notebook

| Parte | Pregunta | Método |
|---|---|---|
| **Parte 1 — Embeddings** | ¿los vectores de 128 dimensiones separan bien a las personas? | vecino más cercano (k-NN, k = 1) con validación cruzada estratificada de 5 partes |
| **Parte 2 — Reconocimiento** | con las reglas de `recognize_faces_*.py`, ¿el sistema acierta? | promedio de las `NEAREST` distancias más cercanas por persona + `THRESHOLD` y `MIN_MARGIN`, *leave-one-out* |

### 4.2 Preparar el kernel

`ipykernel` ya viene en `requirements.txt`. Con el entorno activado, registra el entorno como kernel
de Jupyter (solo la primera vez):

```bash
source .envproject/bin/activate
python -m ipykernel install --user --name envproject --display-name "Python 3 (.envproject)"
```

### 4.3 Abrir y correr el notebook

**Opción A — VS Code (recomendada):**

1. Abre `evaluar.ipynb`.
2. Arriba a la derecha, en **Select Kernel**, elige `.envproject` (o el kernel registrado en 4.2).
3. Pulsa **Run All**.

**Opción B — JupyterLab:**

```bash
pip install jupyterlab        # no viene en requirements.txt
jupyter lab evaluar.ipynb
```

En el menú **Kernel → Change Kernel** elige `Python 3 (.envproject)` y luego **Run → Run All Cells**.

> El notebook debe ejecutarse desde la carpeta raíz del proyecto, porque busca `dataset/` con
> ruta relativa. La celda 2.1 es la más lenta: detecta y codifica cada rostro del dataset.

### 4.4 Parámetros que se pueden cambiar

Se editan en sus celdas y se vuelve a correr todo (**Run All**).

| Celda | Parámetro | Valor | Efecto |
|---|---|---|---|
| 2 | `DATASET` | `dataset` | carpeta con una subcarpeta por persona |
| 2 | `DETECTOR` | `hog` | `hog` (rápido) o `cnn` (preciso, lento sin GPU) |
| 2 | `NUM_JITTERS` | `1` | variaciones del recorte que se promedian |
| 2 | `MODELO` | `small` | `small` (5 puntos) o `large` (68 puntos) |
| 5 | `THRESHOLD` | `0.48` | distancia promedio máxima para aceptar a alguien |
| 5 | `NEAREST` | `3` | vecinos más cercanos promediados por persona |
| 5 | `MIN_MARGIN` | `0.055` | ventaja mínima sobre la segunda persona |

### 4.5 Qué muestra

| Sección | Resultado |
|---|---|
| 3 | `classification_report` de los embeddings (precision, recall, F1 por persona) |
| 4 | matriz de confusión de los embeddings |
| 6 | conteo de aciertos, desconocidos y confusiones del reconocimiento |
| 7 | `classification_report` del reconocimiento (`micro avg` en lugar de `accuracy`) |
| 8 | matriz de confusión del reconocimiento, con columna extra `Desconocido` |
| 9 | curva de aciertos / desconocidos / confusiones según `THRESHOLD` |
| 10 | F1 por persona: embeddings frente a reconocimiento |

Un rostro que queda como *Desconocido* baja el **recall** de su persona pero no la **precision** de
nadie: el sistema no se equivocó de persona, solo no decidió.

---


## 5. Adaptaciones realizadas en Windows

El proyecto original se adaptó para ejecutarse en **Windows, Python 3.12 y JupyterLab**, conservando los nombres y la estructura general de los archivos. Las instrucciones originales para macOS se mantienen en las secciones anteriores.

### 5.1 `encode_faces.py`

El código original no fue modificado. Se actualizaron las imágenes del directorio `dataset` y se ejecutó nuevamente el entrenamiento, obteniendo **270 codificaciones faciales**. El archivo actualizado se guardó como `encodings.pickle`.

### 5.2 `recognize_faces_image.py`

- **Compatibilidad con NumPy:** se incorporó una clase basada en `pickle.Unpickler` para adaptar las referencias internas de NumPy al cargar codificaciones generadas con otra versión de la biblioteca.
- **Comparación facial:** se calcularon las distancias entre el rostro detectado y las codificaciones almacenadas. Después, se agruparon las distancias por persona, se seleccionaron las tres menores de cada grupo y se calculó su promedio. La identidad con el promedio más bajo se consideró la mejor candidata.
- **Validación de identidad:** se asignó el nombre únicamente cuando el promedio fue menor o igual a `0.48` y la diferencia respecto al segundo mejor candidato fue de al menos `0.055`. En caso contrario, se utilizó la etiqueta `Desconocido`.
- **Conversión de colores:** se conservó la transformación original de BGR a RGB mediante `cv2.cvtColor()`.
- **Visualización:** se cambió la ventana a `WINDOW_NORMAL` y se ajustó su tamaño para visualizar fotografías grandes. También se diferenciaron los rostros reconocidos y `Desconocido` mediante recuadros verdes y rojos.

### 5.3 `recognize_faces_video.py`

- **Compatibilidad con NumPy:** se incorporó la misma adaptación de carga de `encodings.pickle` utilizada en el programa de imágenes.
- **Cambio en la clasificación:** se reemplazó la votación basada en la cantidad de coincidencias por una comparación de distancias. Para cada persona se tomaron sus tres codificaciones más cercanas al rostro analizado y se calculó el promedio de las distancias. Se seleccionó la identidad con el menor promedio, siempre que cumpliera el umbral `0.48` y el margen `0.055`.
- **Procesamiento de fotogramas:** se mantuvo el redimensionamiento a 750 píxeles de ancho y la conversión posterior de BGR a RGB.
- **Visualización:** se conservaron la captura mediante `VideoStream` y las opciones de salida. Se añadieron colores diferentes para los recuadros de identidades reconocidas y `Desconocido`.

### 5.4 `recognize_faces_video_file.py`

- **Compatibilidad con NumPy:** se incorporó una clase de carga compatible para leer el archivo de codificaciones entre versiones de NumPy.
- **Corrección de BGR a RGB:** el código original redimensionaba el fotograma, pero convertía a RGB la imagen sin redimensionar. Se corrigió para que la conversión se realice sobre el mismo fotograma redimensionado que se utiliza en la detección facial.
- **Resolución de análisis:** se incorporó el parámetro `--width`, con un valor predeterminado de 900 píxeles, evitando ampliar los videos que ya tengan un ancho menor.
- **Comparación facial:** se agruparon las distancias por persona y se calculó el promedio de las tres menores. Se asignó la identidad con el mejor promedio cuando cumplió el umbral `0.48` y el margen mínimo `0.055`.
- **Seguimiento de rostros:** se añadieron asociaciones entre fotogramas utilizando la posición del rostro y la distancia entre sus codificaciones faciales. Esto permite relacionar detecciones consecutivas con un mismo rostro.
- **Continuidad temporal:** se incorporó una retención de identidad de hasta `0.35` segundos, condicionada a que la comparación facial siga siendo compatible con la identidad previamente reconocida.
- **Generación de videos:** se utilizó el códec `mp4v` para archivos MP4 y se obtuvo la cantidad de fotogramas por segundo (FPS) del video de entrada para conservarla en el archivo generado.
- **Visualización y guardado:** se mantuvieron los parámetros originales `--input`, `--output` y `--display`, permitiendo generar el video sin mostrar la ventana de procesamiento.

### 5.5 Archivos conservados

- `search_bing_api.py`: sin modificaciones.
- `requirements.txt`: sin modificaciones en los códigos adaptados.
- Los comentarios originales en inglés se conservaron cuando seguían siendo correctos y se actualizaron únicamente los correspondientes a las partes modificadas.

### 5.6 Comandos de ejecución en JupyterLab

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

---

## 6. Evaluación adicional de personas no registradas

Esta prueba complementa la evaluación de las ocho identidades registradas descrita en `evaluar.ipynb`. Su finalidad es comprobar cómo se clasifican rostros de personas **no incluidas** en la galería `encodings.pickle`.

### 6.1 Archivos de la prueba

| Ruta | Contenido |
|---|---|
| `Unknown/` | Diez fotografías utilizadas como conjunto independiente de prueba; **no deben incorporarse a `dataset/`** para esta evaluación. |
| `evaluacion_desconocidos.ipynb` | Notebook de JupyterLab que procesa las fotografías, presenta las predicciones y calcula el resumen. |
| `evaluacion_desconocidos.csv` | Resultados por rostro detectado, con fotografía, predicción, distancia y margen. |

> La carpeta se denomina `Unknown` como nombre de directorio, pero la etiqueta mostrada por el clasificador es **`Desconocido`**.

### 6.2 Procedimiento para reproducir la evaluación

1. Abrir JupyterLab desde la **raíz del repositorio** con el entorno Python configurado.
2. Comprobar que estén disponibles `Unknown/` y `encodings.pickle`.
3. Abrir `evaluacion_desconocidos.ipynb` y ejecutar **Run → Run All Cells**.
4. Revisar las predicciones y el resumen; el notebook guarda el archivo `evaluacion_desconocidos.csv`.

Se conservaron los mismos criterios de clasificación que utiliza el sistema principal:

| Parámetro | Valor |
|---|---:|
| Detector de rostros | `hog` |
| `THRESHOLD` | `0.48` |
| `NEAREST` | `3` |
| `MIN_MARGIN` | `0.055` |

### 6.3 Resultados de las pruebas realizadas

| Indicador | Resultado |
|---|---:|
| Fotografías procesadas | 10 |
| Rostros detectados | 27 |
| Clasificados como `Desconocido` | 27 |
| Asignaciones a identidades registradas | 0 |
| Tasa de rechazo correcto entre los rostros detectados | 100 % |

La tasa se calculó como `27 / 27 × 100 = 100 %`, considerando que los rostros evaluados pertenecen a personas no registradas. Este porcentaje **solo describe las detecciones de este conjunto de prueba**: no garantiza el mismo resultado con otras personas, imágenes o condiciones. El CSV tampoco mide por sí solo los rostros visibles que el detector pudiera omitir.

Esta evaluación se realizó **sin modificar ni regenerar `encodings.pickle`**. Los resultados se documentan en el apartado 4.5 del informe académico.

### 6.4 Privacidad y publicación de imágenes

Las fotografías de prueba contienen rostros de personas. Antes de publicarlas en un repositorio público, se debe comprobar que se cuenta con autorización para su difusión. Si no existe autorización, mantener las imágenes fuera del repositorio público y publicar únicamente resultados anonimizados y las instrucciones de reproducción con datos autorizados.
