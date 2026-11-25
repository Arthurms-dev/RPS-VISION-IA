import os
import sqlite3
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CLASSES = {0: "pedra", 1: "papel", 2: "tesoura"}
DB_PATH = "dataset_treino.db"


def inicializar_dataset_db():
    with sqlite3.connect(DB_PATH) as conn:
        colunas = ", ".join([f"f{i} REAL" for i in range(42)])
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS dataset_treino (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {colunas},
                classe INTEGER NOT NULL
            );
        """)


def extrair_vetor_landmarks(hand_landmarks):
    base_x, base_y = hand_landmarks[0].x, hand_landmarks[0].y
    vetor = []
    for lm in hand_landmarks:
        vetor.extend([lm.x - base_x, lm.y - base_y])

    # Normalização de escala: torna o gesto imune à distância da câmera
    max_valor = max(abs(v) for v in vetor) or 1.0
    vetor = [v / max_valor for v in vetor]
    return vetor


inicializar_dataset_db()
classe_atual = int(input("Digite a classe para coletar (0=Pedra, 1=Papel, 2=Tesoura): "))
if classe_atual not in CLASSES:
    raise ValueError("Classe inválida! Digite 0, 1 ou 2.")

nome_classe = CLASSES[classe_atual]

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

cap = cv2.VideoCapture(0)
coletadas = 0
meta = 100
gravando = False

print(f"Coletando {meta} vetores de landmarks para '{nome_classe.upper()}'. Pressione 'S' para iniciar.")

with vision.HandLandmarker.create_from_options(options) as landmarker, sqlite3.connect(DB_PATH) as conn:
    placeholders = ", ".join(["?"] * 43)
    while cap.isOpened() and coletadas < meta:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            for lm in result.hand_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            if gravando:
                vetor = extrair_vetor_landmarks(result.hand_landmarks[0])
                conn.execute(f"INSERT INTO dataset_treino VALUES (NULL, {placeholders})", (*vetor, classe_atual))
                coletadas += 1
                cv2.waitKey(50)

        status = f"GRAVANDO ({coletadas}/{meta})" if gravando else "Pressione 'S' para gravar"
        cv2.putText(frame, f"{nome_classe.upper()} - {status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Coleta Landmarks MediaPipe", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("s"):
            gravando = True
        elif tecla == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Concluído! {coletadas} amostras salvas em '{DB_PATH}'.")