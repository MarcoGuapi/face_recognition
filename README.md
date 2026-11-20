# Guía de uso — face-recognition-opencv

Reconocimiento facial con `face_recognition` (dlib) + OpenCV sobre imágenes, video y webcam.
Verificado en macOS (Apple Silicon) con Python 3.14
## Tutorial oficial del código
[Pyimage Search - Face Recongnition](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
---

## 1. Instalación de librerías

### 1.1 Activar el entorno virtual

```bash
cd "{path}/face-recognition-opencv"
source .envproject/bin/activate
```

Si no existe el entorno, se crea con `python3 -m venv .envproject`.
0 con `virtualenv .envproject`
### 1.2 Instalar las dependencias

```bash
pip install -r requirements.txt
pip install --no-deps face_recognition==1.3.0 face_recognition_models==0.3.0
```

### 1.3 Por qué son dos instalaciones(Solo explicación no requiere realizar tareas)

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

### 1.5 Verificar que quedó bien

```bash
python -c "import cv2, dlib, face_recognition, imutils; print('OK')"
```

---

## 2. Las tres maneras de correrlo

Todas parten del entorno activado y usan el archivo `encodings.pickle` que ya viene en el proyecto o generado con `encode.faces.py`.

Los cuatro scripts tienen **valores por defecto**, así que se pueden correr sin ningún argumento.
Cada parámetro funcionando igual si lo pasas explícitamente, el default solo aplica cuando lo omites.

| Script | `-e` | `-i` | `-d` |
|---|---|---|---|
| `recognize_faces_image.py` | `encodings.pickle` | `examples/example_01.png` | `hog` |
| `recognize_faces_video_file.py` | `encodings.pickle` | `videos/lunch_scene.mp4` | `hog` |
| `recognize_faces_video.py` | `encodings.pickle` |  (webcam - default) | `hog` |
| `encode_faces.py` | `encodings.pickle` (salida de vectores 128) | `dataset` | `hog` |

> **Sobre `-d`:** el default original del proyecto era `cnn`, que es más preciso pero sin GPU
> resulta muy lento el procesamientp. Se cambió a `hog` se puede utilizar `cnn` si la pc tiene `GPU`,
> que corre en menos de un segundo por imagen. Usa `-d cnn` solo para comparar precisión en una
> imagen suelta.

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
python recognize_faces_video.py -o output/webcam.avi         # ademas grabar
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