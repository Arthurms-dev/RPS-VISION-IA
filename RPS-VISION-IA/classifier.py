import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_mlp.joblib")
TASK_PATH = "hand_landmarker.task"
CLASSES = ["Pedra", "Papel", "Tesoura"]

modelo = None
landmarker = None


def extrair_vetor_landmarks(hand_landmarks):
    base_x, base_y = hand_landmarks[0].x, hand_landmarks[0].y
    vetor = []
    for lm in hand_landmarks:
        vetor.extend([lm.x - base_x, lm.y - base_y])

    max_valor = max(abs(v) for v in vetor) or 1.0
    vetor = [v / max_valor for v in vetor]
    return vetor


def carregar_modelo():
    global modelo, landmarker
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Arquivo {MODEL_PATH} não encontrado. Rode treinar_modelo.py primeiro.")
    if not os.path.exists(TASK_PATH):
        raise FileNotFoundError(f"Arquivo {TASK_PATH} não encontrado na pasta do projeto.")

    modelo = joblib.load(MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return modelo


def classificar_landmarks(hand_landmarks):
    if modelo is None:
        raise RuntimeError("O modelo ainda não foi carregado.")

    vetor = extrair_vetor_landmarks(hand_landmarks)
    X_input = np.array([vetor], dtype=np.float32)

    probs = modelo.predict_proba(X_input)[0]
    classe_idx = int(np.argmax(probs))
    confianca = float(np.max(probs))

    return CLASSES[classe_idx], confianca


def reconhecer_gesto(imagem):
    if modelo is None or landmarker is None:
        raise RuntimeError("O modelo ou MediaPipe ainda não foi carregado.")

    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return "Nenhuma mão", 0.0

    return classificar_landmarks(result.hand_landmarks[0])