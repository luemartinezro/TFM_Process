import librosa
import soundfile as sf
import opensmile
import pandas as pd
import numpy as np
from pathlib import Path
import os
import uuid
import json
import spacy
import re
from collections import Counter
import whisper
import ffmpeg
import warnings



warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU"
)

whisper_model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm")
print("Modelo cargado correctamente")


directorio_origen='TAILBANK/'

# python -m spacy download en_core_web_sm


def keyword_repetitions(doc):
    """
    Calcula la proporción de repeticiones de palabras clave en un texto.
    Se consideran palabras de contenido (sustantivos, verbos y adjetivos),
    excluyendo stopwords y signos no alfabéticos.
    """

    # Extrae lemas normalizados (minúsculas) de palabras relevantes:
    # - Solo tokens alfabéticos
    # - Excluye stopwords
    # - Incluye solo sustantivos, verbos y adjetivos
    words = [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop and t.pos_ in {"NOUN", "VERB", "ADJ"}
    ]

    # Si no hay palabras válidas, devuelve 0 para evitar divisiones por cero
    if not words:
        return 0.0

    # Cuenta cuántas veces aparece cada palabra
    counts = Counter(words)

    # Calcula el número total de repeticiones:
    # para cada palabra que aparece más de una vez,
    # suma las repeticiones adicionales (c - 1)
    repeated = sum(c - 1 for c in counts.values() if c > 1)

    # Devuelve la proporción de repeticiones respecto al total de palabras
    return repeated / len(words)



def identificar_genero_pitch(audio_path):
    """
    Estima el género del hablante a partir del pitch medio de la voz.
    La clasificación se basa en un umbral simple de frecuencia fundamental.
    """

    # Carga el audio (frecuencia de muestreo por defecto, señal mono)
    y, sr = librosa.load(audio_path)

    # Calcula el pitch (frecuencia fundamental estimada) y su magnitud
    # pitches: matriz de frecuencias
    # magnitudes: energía asociada a cada pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Selecciona solo los valores de pitch con magnitud significativa
    # (por encima de la mediana) para reducir ruido
    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    # Calcula el pitch medio del hablante
    pitch_mean = np.mean(pitch_values)

    # Clasificación simple basada en umbral:
    # voces graves -> "male", voces agudas -> "female"
    if pitch_mean < 165:
        return "male"
    else:
        return "female"


def normalizar_nombre_audio(nombre_archivo):
    """
    Normaliza el nombre de un archivo de audio para obtener
    un identificador limpio y consistente del hablante.
    """

    # Obtiene el nombre del archivo sin la extensión
    nombre = nombre_archivo.stem  

    # Elimina el prefijo 'N_' usado para indicar audio normalizado
    nombre = re.sub(r"^N_", "", nombre)     

    # Elimina sufijos numéricos finales (ej. _1, _23)
    nombre = re.sub(r"_\d+$", "", nombre)  

    # Inserta un espacio entre letras minúsculas y mayúsculas consecutivas
    # Ejemplo: "johnDoe" -> "john Doe"
    nombre = re.sub(r"([a-z])([A-Z])", r"\1 \2", nombre)

    # Convierte todo a minúsculas y elimina espacios sobrantes
    return nombre.lower().strip()


def audio_quality_score(y, sr):
    score = 100
    detalles = {}

    # 1️⃣ RMS (volumen promedio de la señal)
    # Calcula la energía media del audio.
    # Un RMS muy bajo indica audio débil o silencioso.
    # Un RMS muy alto indica audio saturado o mal normalizado.
    rms = np.sqrt(np.mean(y**2))
    detalles["rms"] = rms

    # Penaliza si el volumen está fuera del rango saludable para voz
    # Rango recomendado: 0.05 – 0.15
    if rms < 0.05 or rms > 0.15:
        score -= 30


    # 2️⃣ Clipping (saturación de la señal)
    # Detecta si la señal alcanza o supera ±1.0,
    # lo que indica recorte digital (distorsión).
    clipping = np.any(np.abs(y) >= 1.0)
    detalles["clipping"] = clipping

    # El clipping es muy dañino para jitter, shimmer y HNR,
    # por eso se penaliza fuertemente.
    if clipping:
        score -= 40

    # 3️⃣ Duración útil del audio
    # Calcula la duración total en segundos.
    # Audios muy cortos no permiten estimar bien
    # características acústicas estables.
    duration = len(y) / sr
    detalles["duracion"] = duration

    # Penaliza audios demasiado cortos (< 2 segundos)
    # porque generan medidas inestables en openSMILE.
    if duration < 2.0:
        score -= 20

    # 4️⃣ Pico robusto (percentil 95)
    # Mide la amplitud típica alta ignorando picos extremos.
    # Es más robusto que usar el máximo absoluto.
    peak95 = np.percentile(np.abs(y), 95)
    detalles["peak95"] = peak95

    # Si el pico típico supera 1.0, la señal está
    # mal escalada o muy cerca del clipping.
    if peak95 > 1.0:
        score -= 10

    # Asegurar que el score final esté entre 0 y 100
    # Evita valores negativos tras aplicar penalizaciones.
    score = max(0, score)

    # Etiqueta final
    if score >= 80:
        calidad = "Excelente"
    elif score >= 60:
        calidad = "Usable"
    else:
        calidad = "Mala"

    return score, calidad


def normalizacion_audio(audio_path, directorio_origen, directorio_destino):
    """
    Carga un archivo de audio, lo normaliza en energía (RMS),
    evalúa su calidad y lo guarda manteniendo la estructura
    original de directorios.
    """

    # Convierte las rutas a objetos Path para facilitar el manejo de archivos
    audio_path = Path(audio_path)
    directorio_origen = Path(directorio_origen)
    directorio_destino = Path(directorio_destino)

    # =========================
    # MANTENER ESTRUCTURA DE CARPETAS
    # =========================
    # Obtiene la ruta relativa del audio respecto al directorio origen
    rel_path = audio_path.parent.relative_to(directorio_origen)

    # Construye la carpeta destino respetando la estructura original
    carpeta_destino = directorio_destino / rel_path

    # Crea las carpetas necesarias si no existen
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    # =========================
    # CARGA DEL AUDIO
    # =========================
    # Carga el audio:
    # - frecuencia de muestreo fija a 16 kHz
    # - señal mono
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # =========================
    # NORMALIZACIÓN RMS
    # =========================
    # Calcula la energía RMS de la señal
    rms = np.sqrt(np.mean(y**2))

    # Normaliza la señal para que tenga una RMS objetivo (~0.1)
    # Evita división por cero en audios silenciosos
    if rms > 0:
        y = y / rms * 0.1

    # =========================
    # EVALUACIÓN DE CALIDAD
    # =========================
    # Calcula un score de calidad del audio y una etiqueta cualitativa
    score, calidad = audio_quality_score(y, sr)

    # =========================
    # NOMBRE Y RUTA DE SALIDA
    # =========================
    # Prefijo "N_" para indicar que el audio fue normalizado
    nuevo_nombre = f"N_{audio_path.name}"

    # Ruta final del archivo normalizado
    path_dest = carpeta_destino / nuevo_nombre

    # =========================
    # GUARDADO DEL AUDIO
    # =========================
    # Guarda el audio normalizado en disco
    sf.write(path_dest, y, sr)

    # =========================
    # SALIDA
    # =========================
    # Devuelve información relevante para el pipeline posterior
    return {
        "score": score,                      # Score numérico de calidad
        "calidad": calidad,                  # Etiqueta de calidad (ej. buena/mala)
        "audio_normalizado": path_dest,      # Ruta al audio normalizado
        "nombre_audio": nuevo_nombre         # Nombre del archivo generado
    }



def opensmile_parameters(salida_normalizada):
    """
    Extrae características acústicas del audio utilizando OpenSMILE
    con el conjunto eGeMAPSv02 a nivel de funcionales.
    """

    # Inicializa el objeto Smile de OpenSMILE
    # eGeMAPSv02 es un conjunto compacto y optimizado de features,
    # diseñado para tareas de análisis paralingüístico y clínico
    # 88 caracteristicas
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,      # Set de features eGeMAPS v02
        feature_level=opensmile.FeatureLevel.Functionals  # Estadísticos globales del audio
    )

    # Procesa el archivo de audio normalizado y devuelve
    # un DataFrame con las características acústicas extraídas
    return smile.process_file(salida_normalizada)



def opensmile_parameters_Compare_2016(salida_normalizada):
    """
    Extrae características acústicas del audio utilizando OpenSMILE
    con el conjunto de features ComParE 2016 a nivel de funcionales.
    """

    # Inicializa el objeto Smile de OpenSMILE
    # ComParE_2016 es un conjunto estándar de características
    # muy utilizado en tareas paralingüísticas (emoción, patología del habla, etc.)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,   # Set de features ComParE 2016
        feature_level=opensmile.FeatureLevel.Functionals # Estadísticos globales del audio
    )

    # Procesa el archivo de audio normalizado y devuelve
    # un DataFrame con todas las características extraídas
    return smile.process_file(salida_normalizada)



def extract_whisper_spacy_features(audio_path):
    """
    Transcribe un archivo de audio con Whisper y extrae características
    lingüísticas y cognitivas usando spaCy.
    """

    # Diccionario donde se almacenarán todas las features extraídas
    features = {}

    # =========================
    # ASEGURAR FFMPEG (WINDOWS)
    # =========================
    # Ruta donde está instalado ffmpeg (necesario para Whisper)
    ffmpeg_path = r"C:\ffmpeg\bin"

    # Añade ffmpeg al PATH si no está ya incluido
    if ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_path

    # =========================
    # RUTA DE AUDIO
    # =========================
    # Convierte la ruta a objeto Path
    audio_path = Path(audio_path)

    # Verifica que el archivo de audio exista
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    # =========================
    # WHISPER: TRANSCRIPCIÓN
    # =========================
    # Transcribe el audio usando Whisper en inglés
    result = whisper_model.transcribe(str(audio_path), language="en")

    # Texto transcrito limpio de espacios iniciales/finales
    text = result["text"].strip()

    # Limpieza básica del texto: elimina espacios múltiples
    text_clean = re.sub(r"\s+", " ", text)

    # =========================
    # MÉTRICAS BÁSICAS DE TEXTO
    # =========================
    # Número total de caracteres
    features["n_chars"] = len(text_clean)

    # Procesa el texto con spaCy
    doc = nlp(text_clean)

    # Tokens sin signos de puntuación ni espacios
    tokens = [t for t in doc if not t.is_punct and not t.is_space]

    # Palabras alfabéticas (excluye números y símbolos)
    words = [t for t in tokens if t.is_alpha]

    # Número de tokens, palabras y frases
    features["n_tokens"] = len(tokens)
    features["n_words"] = len(words)
    features["n_sents"] = len(list(doc.sents))

    # Media de palabras por frase
    features["mean_words_per_sent"] = (
        features["n_words"] / features["n_sents"]
        if features["n_sents"] > 0 else 0.0
    )

    # =========================
    # DIVERSIDAD LÉXICA
    # =========================
    # Formas léxicas normalizadas (minúsculas)
    word_forms = [t.text.lower() for t in words]

    # Conjunto de palabras únicas
    unique_words = set(word_forms)

    # Type-Token Ratio (TTR)
    features["ttr"] = (
        len(unique_words) / len(word_forms)
        if len(word_forms) > 0 else 0.0
    )

    # MATTR (Moving-Average TTR) con ventana de 50 palabras
    window = 50
    if len(word_forms) >= window:
        ttrs = []
        for i in range(len(word_forms) - window + 1):
            w = word_forms[i:i + window]
            ttrs.append(len(set(w)) / window)
        features["mattr_50"] = float(np.mean(ttrs))
    else:
        # Si el texto es corto, se usa el TTR estándar
        features["mattr_50"] = features["ttr"]

    # =========================
    # VARIABLES COGNITIVAS
    # =========================
    # Número de repeticiones de palabras clave (indicador de posible deterioro)
    features["keyword_repetitions"] = keyword_repetitions(doc)

    # =========================
    # POS TAGGING
    # =========================
    # Conteo de categorías gramaticales (POS)
    pos_counts = Counter(t.pos_ for t in words)
    total_words = len(words)

    # Función auxiliar para calcular ratios POS
    def ratio(pos):
        return pos_counts.get(pos, 0) / total_words if total_words > 0 else 0.0

    # Ratios de categorías gramaticales principales
    features["noun_ratio"] = ratio("NOUN")
    features["verb_ratio"] = ratio("VERB")
    features["adj_ratio"] = ratio("ADJ")
    features["adv_ratio"] = ratio("ADV")
    features["pron_ratio"] = ratio("PRON")
    features["propn_ratio"] = ratio("PROPN")

    # Palabras de contenido: sustantivos, verbos, adjetivos y adverbios
    content_words = sum(
        pos_counts.get(p, 0) for p in ["NOUN", "VERB", "ADJ", "ADV"]
    )

    # Ratio de palabras de contenido sobre el total
    features["content_ratio"] = (
        content_words / total_words if total_words > 0 else 0.0
    )

    # Ratio sustantivos / pronombres
    # Valores altos pueden indicar discurso más informativo
    features["noun_pron_ratio"] = (
        pos_counts.get("NOUN", 0) / pos_counts.get("PRON", 1)
        if pos_counts.get("PRON", 0) > 0 else float("inf")
    )

    # Devuelve el diccionario con todas las features extraídas
    return features

def construir_json_desde_directorio(ruta_base):
    """
    Recorre un directorio de forma recursiva, procesa archivos de audio (.wav),
    extrae características acústicas y lingüísticas, y genera un archivo JSON
    con toda la información.
    """

    # Directorio donde se guardarán los audios normalizados
    directorio_destino = 'dementibank_normalizado/'

    # Convierte la ruta base en un objeto Path para manejo de archivos
    ruta = Path(ruta_base)

    # Lista donde se almacenará la información final de cada audio
    resultados = []

    # Extensiones de audio permitidas
    extensiones_audio = {".wav"}

    # Recorre todos los archivos y subdirectorios de forma recursiva
    for archivo in ruta.rglob("*"):

        # Filtra solo archivos .wav
        if archivo.is_file() and archivo.suffix.lower() in extensiones_audio:
            print("archivo:", archivo)

            # Normaliza el audio y calcula métricas de calidad
            quality = normalizacion_audio(
                archivo,
                directorio_origen,
                directorio_destino
            )

            # Extrae parámetros acústicos usando OpenSMILE
            features = opensmile_parameters(quality["audio_normalizado"])

            # Extrae parámetros del set Compare 2016 de OpenSMILE
            features2 = opensmile_parameters_Compare_2016(
                quality["audio_normalizado"]
            )

            # Obtiene el nombre del archivo sin extensión
            nombre_audio = archivo.stem.strip()

            # Diccionario principal con metadatos y estructuras vacías
            data = {
                # Identificador único del registro
                "uuid": str(uuid.uuid4()),

                # Nombre del audio normalizado
                "audio": quality["nombre_audio"],

                # Nombre del hablante normalizado
                "name": normalizar_nombre_audio(Path(nombre_audio)),

                # Etiqueta de demencia (a completar posteriormente)
                "dementia": "",

                # Género estimado a partir del pitch
                "gender": identificar_genero_pitch(archivo),

                # Etnicidad (placeholder)
                "ethnicity": "",

                # Score global de calidad del audio
                "score": quality["score"],

                # Métrica de calidad del audio
                "calidad": quality["calidad"],

                # Diccionarios para almacenar los distintos parámetros
                "parametros_librosa": {},
                "parametros_opensmile": {},
                "parametros_opensmile_compare": {},
                "parametros_whisperSpacy": {}
            }

            # Convierte y guarda los parámetros OpenSMILE en formato JSON-serializable
            data["parametros_opensmile"].update({
                k: float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                for k, v in features.items()
            })

            # Convierte y guarda los parámetros OpenSMILE Compare 2016
            data["parametros_opensmile_compare"].update({
                k: float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                for k, v in features2.items()
            })

            # Extrae y guarda características acústicas con librosa
            data["parametros_librosa"].update(
                extract_all_librosa_features(quality["audio_normalizado"])
            )

            # Extrae características lingüísticas con Whisper + spaCy
            data["parametros_whisperSpacy"].update(
                extract_whisper_spacy_features(quality["audio_normalizado"])
            )

            # Añade el registro completo a la lista final
            resultados.append(data)

    # Guarda todos los resultados en un archivo JSON
    with open("ADReSSo21.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)


# Llamada a la función usando el directorio de origen
construir_json_desde_directorio(directorio_origen)
