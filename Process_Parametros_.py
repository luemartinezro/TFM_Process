# =========================
# IMPORTS
# =========================
# Procesamiento de audio
import librosa
import soundfile as sf
import opensmile

# Manipulación de datos
import pandas as pd
import numpy as np

# Manejo de rutas y sistema
from pathlib import Path
import os
import uuid
import json
import warnings

# NLP
import spacy
import re
from collections import Counter

# ASR
import whisper
import ffmpeg


# =========================
# CONFIGURACIÓN GLOBAL
# =========================

# Ignorar warnings de Whisper cuando no hay GPU
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU"
)

# Carga del modelo Whisper (ASR)
whisper_model = whisper.load_model("base")

# Carga del modelo spaCy (NLP)
nlp = spacy.load("en_core_web_sm")

print("Modelo cargado correctamente")

# Directorio base con los audios originales
directorio_origen = 'TAILBANK/'


# =========================
# FUNCIONES NLP
# =========================

def keyword_repetitions(doc):
    """
    Calcula la proporción de repeticiones léxicas
    sobre palabras de contenido (NOUN, VERB, ADJ).
    """
    words = [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop and t.pos_ in {"NOUN", "VERB", "ADJ"}
    ]

    if not words:
        return 0.0

    counts = Counter(words)

    # Número de repeticiones (ocurrencias - 1)
    repeated = sum(c - 1 for c in counts.values() if c > 1)

    return repeated / len(words)


# =========================
# ANÁLISIS DE VOZ
# =========================

def identificar_genero_pitch(audio_path):
    """
    Estima el género usando la media del pitch.
    Umbral aproximado basado en F0 promedio.
    """
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values)

    return "male" if pitch_mean < 165 else "female"


def normalizar_nombre_audio(nombre_archivo):
    """
    Limpia y normaliza el nombre del archivo
    para usarlo como identificador del hablante.
    """
    nombre = nombre_archivo.stem
    nombre = re.sub(r"^N_", "", nombre)
    nombre = re.sub(r"_\d+$", "", nombre)
    nombre = re.sub(r"([a-z])([A-Z])", r"\1 \2", nombre)
    return nombre.lower().strip()


# =========================
# CALIDAD DE AUDIO
# =========================

def audio_quality_score(y, sr):
    """
    Evalúa la calidad del audio usando:
    - RMS
    - Clipping
    - Duración
    - Pico robusto (percentil 95)
    """
    score = 100
    detalles = {}

    # RMS (energía media)
    rms = np.sqrt(np.mean(y**2))
    detalles["rms"] = rms

    if rms < 0.05 or rms > 0.15:
        score -= 30

    # Clipping digital
    clipping = np.any(np.abs(y) >= 1.0)
    detalles["clipping"] = clipping

    if clipping:
        score -= 40

    # Duración
    duration = len(y) / sr
    detalles["duracion"] = duration

    if duration < 2.0:
        score -= 20

    # Pico robusto
    peak95 = np.percentile(np.abs(y), 95)
    detalles["peak95"] = peak95

    if peak95 > 1.0:
        score -= 10

    score = max(0, score)

    # Etiqueta final
    if score >= 80:
        calidad = "Excelente"
    elif score >= 60:
        calidad = "Usable"
    else:
        calidad = "Mala"

    return score, calidad


# =========================
# NORMALIZACIÓN DE AUDIO
# =========================

def normalizacion_audio(audio_path, directorio_origen, directorio_destino):
    """
    Normaliza el audio a RMS fijo y mantiene
    la estructura de carpetas original.
    """
    audio_path = Path(audio_path)
    directorio_origen = Path(directorio_origen)
    directorio_destino = Path(directorio_destino)

    # Mantener estructura de carpetas
    rel_path = audio_path.parent.relative_to(directorio_origen)
    carpeta_destino = directorio_destino / rel_path
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    # Cargar audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Normalización RMS
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Evaluar calidad
    score, calidad = audio_quality_score(y, sr)

    # Nuevo nombre
    nuevo_nombre = f"N_{audio_path.name}"
    path_dest = carpeta_destino / nuevo_nombre

    # Guardar audio normalizado
    sf.write(path_dest, y, sr)

    return {
        "score": score,
        "calidad": calidad,
        "audio_normalizado": path_dest,
        "nombre_audio": nuevo_nombre
    }


# =========================
# OPENSMILE
# =========================

def opensmile_parameters(salida_normalizada):
    """
    Extrae eGeMAPS v02 (functionals).
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    return smile.process_file(salida_normalizada)


def opensmile_parameters_Compare_2016(salida_normalizada):
    """
    Extrae ComParE 2016 (functionals).
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    return smile.process_file(salida_normalizada)


# =========================
# LIBROSA FEATURES
# =========================

def extract_all_librosa_features(audio_path):
    """
    Extrae un set completo de features acústicas
    con Librosa (RMS, pitch, silencios, MFCC, etc.).
    """
    features = {}

    # Carga de audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    features["duracion_s"] = float(duration)
    features["sr"] = int(sr)

    if len(y) == 0:
        return features

    # ---------- RMS y silencios ----------
    rms = librosa.feature.rms(y=y)[0]

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_p25"] = float(np.percentile(rms, 25))
    features["rms_p50"] = float(np.percentile(rms, 50))
    features["rms_p75"] = float(np.percentile(rms, 75))

    rms_threshold = np.percentile(rms, 25)
    silent = rms < rms_threshold

    features["silence_ratio_rms"] = float(np.mean(silent))

    # Segmentos de silencio / voz
    silence_lengths, voiced_lengths = [], []
    current = silent[0]
    count = 0

    for s in silent:
        if s == current:
            count += 1
        else:
            (silence_lengths if current else voiced_lengths).append(count)
            current = s
            count = 1

    (silence_lengths if current else voiced_lengths).append(count)

    hop_length = 512
    frame_duration = hop_length / sr

    silence_dur = np.array(silence_lengths) * frame_duration
    voiced_dur = np.array(voiced_lengths) * frame_duration

    features["silence_segments_n"] = int(len(silence_dur))
    features["voiced_segments_n"] = int(len(voiced_dur))
    features["voiced_ratio"] = float(np.mean(~silent))

    # ---------- Pitch ----------
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0 = f0[~np.isnan(f0)]

    if len(f0):
        features["f0_mean"] = float(np.mean(f0))
        features["f0_std"] = float(np.std(f0))
        features["f0_range"] = float(np.max(f0) - np.min(f0))
    else:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_range"] = 0.0

    # ---------- ZCR ----------
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ---------- MFCC + deltas ----------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    return features


# =========================
# WHISPER + SPACY
# =========================

def extract_whisper_spacy_features(audio_path):
    """
    Transcribe audio con Whisper y extrae
    features lingüísticas con spaCy.
    """
    features = {}

    # Asegurar ffmpeg en Windows
    ffmpeg_path = r"C:\ffmpeg\bin"
    if ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_path

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    # Transcripción
    result = whisper_model.transcribe(str(audio_path), language="en")
    text = re.sub(r"\s+", " ", result["text"].strip())

    doc = nlp(text)

    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    words = [t for t in tokens if t.is_alpha]

    features["n_tokens"] = len(tokens)
    features["n_words"] = len(words)
    features["n_sents"] = len(list(doc.sents))

    # Diversidad léxica
    word_forms = [t.text.lower() for t in words]
    features["ttr"] = len(set(word_forms)) / len(word_forms) if word_forms else 0.0

    # Repeticiones
    features["keyword_repetitions"] = keyword_repetitions(doc)

    return features


# =========================
# PIPELINE PRINCIPAL
# =========================

def construir_json_desde_directorio(ruta_base):
    """
    Recorre un directorio de audios, extrae
    todas las features y genera un JSON final.
    """
    directorio_destino = 'dementibank_normalizado/'
    ruta = Path(ruta_base)
    resultados = []

    for archivo in ruta.rglob("*.wav"):
        print("archivo:", archivo)

        quality = normalizacion_audio(
            archivo,
            directorio_origen,
            directorio_destino
        )

        data = {
            "uuid": str(uuid.uuid4()),
            "audio": quality["nombre_audio"],
            "name": normalizar_nombre_audio(archivo),
            "gender": identificar_genero_pitch(archivo),
            "score": quality["score"],
            "calidad": quality["calidad"],
            "parametros_librosa": extract_all_librosa_features(
                quality["audio_normalizado"]
            ),
            "parametros_whisperSpacy": extract_whisper_spacy_features(
                quality["audio_normalizado"]
            )
        }

        resultados.append(data)

    with open("ADReSSo21.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)


# =========================
# EJECUCIÓN
# =========================

construir_json_desde_directorio(directorio_origen)
