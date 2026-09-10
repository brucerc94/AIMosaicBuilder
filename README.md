# AI Mosaic Builder

AI Mosaic Builder es una aplicación de escritorio local para analizar grandes colecciones de fotografías, encontrar imágenes con personas, puntuarlas, reducir repeticiones y preparar una selección para un mosaico.

La aplicación está construida con Python + PySide6 y utiliza un modelo multimodal local mediante `llama-cpp-python` para el análisis visual.

## ¿Qué hace?

El flujo principal es:

```text
Carpeta de fotografías
        ↓
Descubrimiento de imágenes
        ↓
Prefiltrado rápido
        ↓
SHA-256 / cache local
        ↓
Modelo multimodal
(Gemma-4 u otro modelo compatible)
        ↓
Análisis visual
        ↓
Detección de personas
        ↓
Ranking
        ↓
Eliminación de imágenes similares
        ↓
Selección de las mejores N
        ↓
Crop dinámico definido por coordenadas
        ↓
mosaic.json
```

El objetivo no es simplemente encontrar todas las fotos que contienen una persona, sino seleccionar las fotografías más útiles para un mosaico: buena calidad, buena composición, buena visibilidad del sujeto y suficiente diversidad.

## Arquitectura

El proyecto está dividido por responsabilidades:

```text
AI-Mosaic-Builder/
│
├── main.py
├── requirements.txt
│
├── engine/
│   ├── vision_llm.py       # Motor multimodal local
│   ├── llama_features.py   # Detección de capacidades llama.cpp
│   ├── image_analyzer.py   # Descubrimiento y pipeline de análisis
│   ├── ranking.py          # Puntuación y selección
│   ├── cache.py            # Cache por SHA-256
│   ├── models.py           # Dataclasses y modelos de datos
│   ├── session.py          # Persistencia de sesiones
│   └── storage.py          # Settings
│
├── vision/
│   ├── person_detector.py  # Detección de personas
│   ├── cropper.py          # Cálculo del crop
│   └── similarity.py       # Similitud / diversidad
│
├── ui/
│   ├── main.py
│   ├── settings.py
│   ├── image_grid.py
│   ├── image_detail.py
│   ├── workers.py
│   └── styles.py
│
└── engine/project_export.py # Exportación del proyecto
```

## Análisis con llama.cpp

El motor utiliza `llama-cpp-python` para cargar el modelo GGUF localmente.

Para un modelo multimodal se utilizan dos archivos:

```text
Modelo principal:
Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf

Vision projector:
mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

La aplicación detecta la familia del modelo y evita seleccionar un handler multimodal incorrecto. Para Gemma-4 utiliza `Gemma4ChatHandler` cuando está disponible en la instalación.

La integración intenta adaptarse a las capacidades reales de la versión instalada de `llama-cpp-python`, en lugar de asumir que todos los parámetros multimodales existen.

## Qué analiza el modelo

Cada fotografía puede producir un análisis estructurado con información como:

- existencia de personas;
- cantidad de personas;
- persona principal;
- visibilidad del sujeto;
- rostro visible;
- cuerpo visible;
- oclusión;
- desenfoque;
- composición;
- calidad técnica;
- calidad del sujeto;
- valor para el mosaico;
- motivo de descarte.

Los resultados se convierten en una estructura validable antes de continuar con las siguientes etapas.

## Detección de personas

El análisis del LLM y la detección física de la persona son capas separadas.

El LLM decide si la fotografía es relevante y ayuda a evaluar el sujeto. El detector de personas proporciona bounding boxes reales para que posteriormente pueda calcularse el crop.

La aplicación utiliza un backend de detección desacoplado para que pueda sustituirse por otro detector en el futuro sin modificar todo el pipeline.

## Ranking y diversidad

Después del análisis se calcula un `final_score` entre 0 y 100 combinando diferentes factores de calidad.

La selección final no depende solamente del ranking. También existe una etapa de diversidad para evitar seleccionar muchas fotografías prácticamente iguales.

Conceptualmente:

```text
calidad
  +
diversidad
  =
selección final
```

El número de imágenes puede configurarse, por ejemplo:

```text
4 / 6 / 9 / 12 / 16 / 20 / Automatic
```

## Cache por carpeta

El cache pertenece a la carpeta de fotografías que se está analizando.

Ejemplo:

```text
G:/fotos/
├── foto01.jpg
├── foto02.jpg
├── foto03.jpg
└── .aimosaic/
    └── analysis_cache.json
```

La clave principal del cache es el SHA-256 del archivo.

Esto permite que, si el usuario vuelve a analizar la misma carpeta, las fotografías que no han cambiado no vuelvan a pasar por el modelo.

Ejemplo:

```text
Primera ejecución:
10 fotos
→ 10 análisis

Después se agregan 3 fotos:
13 fotos detectadas
→ 10 cache hits
→ 3 análisis nuevos
```

Cambiar parámetros de selección, como `Target Images`, no obliga a repetir el análisis de las fotografías ya almacenadas en cache.

El cache también permite separar proyectos: cada carpeta tiene su propio cache.

## Sesiones

Además del cache de análisis, la aplicación guarda el estado de la sesión para poder recuperar el trabajo después de una interrupción.

Esto permite reanudar un procesamiento largo sin comenzar nuevamente desde cero.

## Settings

Los ajustes de la aplicación se guardan en JSON y se cargan al iniciar.

Entre ellos están:

- última carpeta de origen;
- modelo GGUF;
- mmproj;
- contexto;
- capas GPU;
- threads;
- número objetivo de imágenes;
- dimensiones del canvas;
- padding;
- threshold de detección;
- parámetros de ranking;
- modelo del detector de personas;
- directorio de cache.

## Exportación

La exportación está diseñada para ser compatible con el flujo de `ImageMosaicView`.

**AI Mosaic Builder no crea imágenes recortadas para el proyecto final.**

No copia las fotos originales ni genera archivos de crop como parte del export.

Solo guarda el proyecto JSON con:

- `type`;
- `filename` de la imagen original;
- `coords` relativas del crop;
- `zoom`;
- `canvas_size`.

Ejemplo:

```json
[
  {
    "type": "body",
    "filename": "evento/juan.jpg",
    "coords": [0.15, 0.08, 0.61, 0.94],
    "zoom": 0.5,
    "canvas_size": [3840, 2160]
  }
]
```

Las coordenadas se calculan sobre la imagen original y se almacenan en el rango `0..1`.

El archivo se guarda directamente en la carpeta Source Folder:

```text
G:/fotos/mosaic.json
```

Esto permite que `ImageMosaicView` utilice esa misma carpeta como directorio del proyecto y reconstruya el crop dinámicamente al cargar el JSON.

## Flujo completo con ImageMosaicView

Actualmente los proyectos están separados.

```text
AI Mosaic Builder
        ↓
analiza fotos
        ↓
selecciona las mejores
        ↓
genera mosaic.json
        ↓
ImageMosaicView
        ↓
lee las fotos originales
        ↓
reconstruye los crops dinámicamente
        ↓
renderiza el mosaico
```

La integración final se mantiene deliberadamente separada para que AI Mosaic Builder pueda desarrollarse y probarse de forma independiente.

## Instalación

Las dependencias principales están en `requirements.txt`:

```text
PySide6
llama-cpp-python
Pillow
numpy
ultralytics
opencv-python
imagehash
```

Instalación:

```cmd
python -m pip install -r requirements.txt
```

## Uso básico

1. Abrir AI Mosaic Builder.
2. Seleccionar `Source Folder`.
3. Seleccionar el modelo GGUF multimodal.
4. Seleccionar el `mmproj` correspondiente.
5. Configurar `Target Images` y el resto de parámetros.
6. Pulsar `Analyze`.
7. Revisar las imágenes, scores y detecciones.
8. Pulsar `Generate Mosaic`.
9. Se generará `mosaic.json` directamente dentro de la carpeta de origen.
10. Abrir esa carpeta/proyecto con `ImageMosaicView`.

## Prueba de visión

El repositorio incluye una prueba para validar primero la ruta multimodal con una sola imagen antes de procesar grandes cantidades.

La prueba debe confirmar que:

```text
imagen
  ↓
llama-cpp-python
  ↓
Gemma4ChatHandler
  ↓
encoder visual
  ↓
respuesta
  ↓
JSON
```

No se considera válida una configuración que simplemente simule la entrada de imagen.

## Estado actual

El proyecto se encuentra en desarrollo activo. La arquitectura principal, el análisis multimodal, el cache por carpeta, el ranking, la detección de personas y la exportación JSON están separados en módulos para permitir seguir mejorando cada etapa sin acoplarlas entre sí.

## Principios del proyecto

- Todo el análisis es local.
- Las fotografías no necesitan enviarse a servicios cloud.
- El cache evita inferencias repetidas.
- La UI no debe bloquearse durante el procesamiento.
- El modelo multimodal no debe inventar bounding boxes.
- Los crops del proyecto final son dinámicos y se reconstruyen desde las coordenadas guardadas.
- El formato de exportación debe respetar el contrato real del visualizador que lo consume.
