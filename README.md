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
Aplicación de requisitos del usuario
        ↓
Diversidad / eliminación de similares
        ↓
Selección + optimización de layout
        ↓
Crop dinámico definido por coordenadas + zoom
        ↓
mosaic.json
```

El objetivo no es simplemente encontrar todas las fotos que contienen una persona, sino seleccionar las fotografías más útiles para un mosaico: buena calidad, buena composición, buena visibilidad del sujeto, variedad y cumplimiento de la receta solicitada por el usuario.

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
│   ├── ranking.py          # Puntuación, requisitos y selección
│   ├── layout_optimizer.py # Cantidad AUTO, zoom y packing
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

Además, el análisis multimodal clasifica atributos visuales usados por la receta del mosaico:

```text
framing:
  face_only | head_shoulders | upper_body | half_body | full_body | unknown

orientation:
  front | three_quarter_front | side | three_quarter_back | back | unknown

gender_presentation:
  male | female | unknown

content_rating:
  safe | suggestive | explicit | unknown

pose:
  standing | seated | lying | walking | other | unknown

looking_at_camera:
  true | false
```

Cuando un atributo no puede determinarse de forma razonable se utiliza `unknown`.

Los resultados se convierten en una estructura validable antes de continuar con las siguientes etapas.

## Mosaic Requirements

El usuario puede definir una receta para indicar cómo quiere que quede compuesto el mosaico.

Los requisitos obligatorios se aplican durante la selección, no son solamente controles visuales de la interfaz.

### Cantidades mínimas

Se pueden pedir cantidades mínimas de:

- `Face only`;
- `Full body`;
- `Front`;
- `Side`;
- `Back`;
- `Male`;
- `Female`;
- `Face visible`;
- `Body visible`.

Ejemplo:

```text
Face only:     1
Full body:     2
Back:          1
Side:          1
Female:        2
Male:          2
```

El selector intenta reservar primero las plazas necesarias para satisfacer estos mínimos y luego completa las plazas restantes con las mejores fotografías disponibles.

Una fotografía puede satisfacer varias categorías a la vez, pero solo ocupa una plaza del mosaico.

### Contenido

Existe una política de contenido:

```text
Don't care
Safe only
NSFW only
```

`Safe only` evita seleccionar imágenes clasificadas como `suggestive`, `explicit` o `unknown`.

### Calidad mínima

También pueden configurarse:

- calidad mínima de la imagen;
- visibilidad mínima de la persona;
- excluir fotografías borrosas;
- excluir fotografías muy ocluidas.

### Preferencias suaves

Además de los mínimos obligatorios existen preferencias que añaden un pequeño bonus al ranking:

- preferir face-only;
- preferir full-body;
- preferir front;
- preferir side;
- preferir back;
- preferir una sola persona;
- preferir face visible;
- preferir body visible.

La diferencia es importante:

```text
REQUIRED
→ debe intentar cumplirse

PREFERRED
→ mejora la selección, pero no bloquea el mosaico
```

### Requisitos imposibles

Si una receta solicita, por ejemplo, 3 fotografías de espalda y solo existen 1 o 2 candidatas válidas, el selector no inventa fotografías. Registra el requisito no satisfecho en el log y utiliza las mejores candidatas restantes.

## Optimización automática del mosaico

`Target Images = Automatic` es el modo pensado para que el usuario no tenga que decidir cuántas fotos caben.

En este modo, AI Mosaic Builder no usa un número fijo como 12 o 20. Para cada conjunto candidato simula el comportamiento de empaquetado de `ImageMosaicView` y calcula:

- tamaño real del crop de cada persona;
- tamaño del sujeto dentro del crop;
- zoom individual recomendado;
- espacio disponible del canvas;
- solapamiento con imágenes ya colocadas;
- cantidad máxima de imágenes que puede incluir manteniendo un tamaño legible;
- uso total del canvas.

`ImageMosaicView` busca posiciones con un barrido de 10 px y, cuando un elemento no cabe, reduce su zoom en pasos de `0.9`. El optimizador reproduce esa regla para que el JSON exportado tenga un resultado predecible al abrirse en el viewer.

La diferencia importante es que AUTO no empieza todas las imágenes en `zoom = 0.5`. Primero calcula un zoom basado en el tamaño del sujeto y después busca el mayor nivel de zoom que todavía permite colocar el conjunto completo.

Esto permite comportamientos como:

```text
Face crop pequeño
→ zoom mayor
→ rostro visible y aprovechado

Full body grande
→ zoom menor
→ cuerpo completo sin ocupar todo el canvas
```

El optimizador también ordena los elementos por huella esperada para reducir fragmentación del espacio. La posición no se guarda en el JSON: el viewer la vuelve a calcular dinámicamente.

### Ejemplo conceptual

```text
Canvas: 1080 × 960

300 fotos encontradas
        ↓
120 candidatas después de requisitos/ranking
        ↓
AUTO prueba diferentes cantidades
        ↓
para cada cantidad:
    calcula crop
    calcula zoom
    simula packing
    mide espacio utilizado
    comprueba tamaño del sujeto
        ↓
elige el layout más útil
        ↓
por ejemplo: 14 imágenes
con zooms diferentes
```

La cantidad final no debe interpretarse como un límite rígido universal: depende de las dimensiones del canvas, las proporciones de los crops, los sujetos detectados, los requisitos y la calidad de las candidatas.

### Modo manual

Si el usuario selecciona un número concreto, por ejemplo `12`, ese número se trata como una cantidad exacta solicitada. El sistema optimiza el zoom y el packing para esas 12 fotos, pero no cambia silenciosamente la cantidad a otra.

## Detección de personas

El análisis del LLM y la detección física de la persona son capas separadas.

El LLM decide si la fotografía es relevante y ayuda a evaluar el sujeto. El detector de personas proporciona bounding boxes reales para que posteriormente pueda calcularse el crop.

La aplicación utiliza un backend de detección desacoplado para que pueda sustituirse por otro detector en el futuro sin modificar todo el pipeline.

## Ranking y diversidad

Después del análisis se calcula un `final_score` entre 0 y 100 combinando diferentes factores de calidad.

La selección final sigue este orden conceptual:

```text
calidad
  ↓
requisitos obligatorios
  ↓
preferencias
  ↓
diversidad
  ↓
optimización de layout
  ↓
mejores imágenes para el canvas
```

La etapa de diversidad evita seleccionar muchas fotografías prácticamente iguales.

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

Cambiar parámetros de selección, como `Target Images` o la receta del mosaico, no obliga a repetir el análisis de las fotografías ya almacenadas en cache.

Cuando cambia el esquema del análisis, la versión de cache se incrementa para evitar reutilizar resultados antiguos que no contienen los nuevos atributos visuales.

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
- número objetivo de imágenes o `Automatic`;
- dimensiones del canvas;
- padding;
- threshold de detección;
- parámetros de ranking;
- modelo del detector de personas;
- requisitos y preferencias del mosaico;
- directorio de cache.

## Exportación

La exportación está diseñada para ser compatible con el flujo de `ImageMosaicView`.

**AI Mosaic Builder no crea imágenes recortadas para el proyecto final.**

No copia las fotos originales ni genera archivos de crop como parte del export.

Solo guarda el proyecto JSON con:

- `type`;
- `filename` de la imagen original;
- `coords` relativas del crop;
- `zoom` calculado para ese elemento;
- `canvas_size`.

Ejemplo:

```json
[
  {
    "type": "body",
    "filename": "evento/juan.jpg",
    "coords": [0.15, 0.08, 0.61, 0.94],
    "zoom": 0.73,
    "canvas_size": [1080, 960]
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
selecciona las mejores según la receta
        ↓
optimiza cantidad + zoom
        ↓
genera mosaic.json
        ↓
ImageMosaicView
        ↓
lee las fotos originales
        ↓
reconstruye los crops dinámicamente
        ↓
recalcula las posiciones del packing
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
5. Seleccionar `Automatic` para que el programa determine la cantidad de imágenes, o indicar una cantidad fija.
6. Definir los requisitos y preferencias del mosaico.
7. Pulsar `Analyze`.
8. Revisar las imágenes, scores, detecciones y atributos visuales.
9. Pulsar `Generate Mosaic`.
10. Se generará `mosaic.json` directamente dentro de la carpeta de origen.
11. Abrir esa carpeta/proyecto con `ImageMosaicView`.

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

El proyecto se encuentra en desarrollo activo. La arquitectura principal, el análisis multimodal, el cache por carpeta, el ranking, la detección de personas, la receta de selección, la optimización de layout y la exportación JSON están separados en módulos para permitir seguir mejorando cada etapa sin acoplarlas entre sí.

## Principios del proyecto

- Todo el análisis es local.
- Las fotografías no necesitan enviarse a servicios cloud.
- El cache evita inferencias repetidas.
- La UI no debe bloquearse durante el procesamiento.
- El modelo multimodal no debe inventar bounding boxes.
- Los crops del proyecto final son dinámicos y se reconstruyen desde las coordenadas guardadas.
- El formato de exportación debe respetar el contrato real del visualizador que lo consume.
- Los requisitos obligatorios tienen prioridad sobre el ranking simple.
- Los atributos inciertos se marcan como `unknown` en vez de inventarse.
- AUTO debe optimizar el aprovechamiento del canvas y el tamaño visible del sujeto, no usar un zoom fijo.
