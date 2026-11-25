import os
import time
import numpy as np
import serial
import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dotenv import load_dotenv

from database import (
    inicializar_banco,
    registrar_partida,
    consultar_ultimas_partidas
)

from agent import escolher_jogada_agente

from classifier import (
    carregar_modelo,
    reconhecer_gesto,
    classificar_landmarks
)


# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

load_dotenv()

ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM4")
BAUD_RATE = 9600
TASK_PATH = "hand_landmarker.task"

st.set_page_config(
    page_title="RPS Vision",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RPS Vision")
st.subheader("Pedra, Papel e Tesoura com Inteligência Artificial")


# ==========================================
# 2. CONEXÃO COM O ARDUINO
# ==========================================

@st.cache_resource
def conectar_arduino():
    try:
        conexao = serial.Serial(
            ARDUINO_PORT,
            BAUD_RATE,
            timeout=1,
            write_timeout=1
        )
        time.sleep(2)
        return conexao
    except Exception as erro:
        print(f"Arduino não conectado: {erro}")
        return None


arduino = conectar_arduino()


def enviar_comando_serial(comando_bytes):
    global arduino
    if arduino is None:
        return

    try:
        if not arduino.is_open:
            arduino.open()
        arduino.reset_output_buffer()
        arduino.write(comando_bytes)
        arduino.flush()
    except Exception as erro:
        print(f"Falha na escrita serial ({erro}). Tentando reconectar...")
        try:
            arduino.close()
        except Exception:
            pass
        st.cache_resource.clear()
        arduino = conectar_arduino()
        if arduino is not None:
            try:
                arduino.write(comando_bytes)
                arduino.flush()
            except Exception as erro_retry:
                st.warning(f"Arduino não respondeu ao comando: {erro_retry}")


def movimentar_arduino(jogada):
    comandos = {
        "pedra": b"1",
        "papel": b"2",
        "tesoura": b"3"
    }
    comando = comandos.get(jogada.lower())
    if comando:
        enviar_comando_serial(comando)


def posicionar_em_papel():
    enviar_comando_serial(b"2")
    time.sleep(1.0)


# ==========================================
# 3. CONEXÃO COM O BANCO DE DADOS
# ==========================================

try:
    inicializar_banco()
except Exception as erro:
    st.error(f"Erro ao conectar ao banco de dados: {erro}")
    st.stop()


# ==========================================
# 4. CARREGAMENTO DO MODELO E MEDIAPIPE
# ==========================================

@st.cache_resource
def init_recursos_ia():
    carregar_modelo()
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE
    )
    detector_visual = vision.HandLandmarker.create_from_options(options)
    return detector_visual


try:
    detector_visual = init_recursos_ia()
except Exception as erro:
    st.error(f"Erro ao carregar o modelo ou MediaPipe: {erro}")
    st.info(
        "Verifique se 'modelo_mlp.joblib' e 'hand_landmarker.task' "
        "estão na pasta do projeto."
    )
    st.stop()


# ==========================================
# 5. ESTADOS DO JOGO
# ==========================================

if "estado_jogo" not in st.session_state:
    st.session_state.estado_jogo = "aguardando"

if "jogada_agente" not in st.session_state:
    st.session_state.jogada_agente = None

if "jogada_usuario" not in st.session_state:
    st.session_state.jogada_usuario = None

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "confianca" not in st.session_state:
    st.session_state.confianca = None


# ==========================================
# 6. REGRAS DO JOGO
# ==========================================

def comparar_jogadas(usuario, agente):
    if usuario == agente:
        return "Empate"

    regras = {
        "pedra": "tesoura",
        "papel": "pedra",
        "tesoura": "papel"
    }

    if regras[usuario] == agente:
        return "Humano ganhou"

    return "Robô ganhou"


# ==========================================
# 7. INICIAR NOVA RODADA
# ==========================================

def iniciar_rodada():
    posicionar_em_papel()
    jogada = escolher_jogada_agente()

    st.session_state.jogada_agente = jogada
    st.session_state.jogada_usuario = None
    st.session_state.resultado = None
    st.session_state.confianca = None
    st.session_state.estado_jogo = "jogando"


# ==========================================
# 8. INTERFACE PRINCIPAL
# ==========================================

col1, col2 = st.columns(2)


# ==========================================
# COLUNA 1 - CONTROLE DO JOGO
# ==========================================

with col1:
    st.markdown("### 🎮 Área do jogo")

    if st.session_state.estado_jogo == "aguardando":
        st.info("Clique no botão abaixo para iniciar o reconhecimento ao vivo.")

        if st.button("▶️ Iniciar rodada", use_container_width=True):
            try:
                iniciar_rodada()
                st.rerun()
            except Exception as erro:
                st.error(f"Erro ao iniciar rodada: {erro}")

    elif st.session_state.estado_jogo == "jogando":
        st.success("O agente já escolheu sua jogada!")
        st.write("Posicione sua mão na frente da câmera e mantenha o gesto firme.")
        st.caption("O sistema está comparando os pontos da sua mão com a base treinada em tempo real.")

        if st.button("⏹️ Cancelar rodada", use_container_width=True):
            st.session_state.estado_jogo = "aguardando"
            st.rerun()

    elif st.session_state.estado_jogo == "resultado":
        st.success("Rodada finalizada!")

        st.markdown(f"**Jogada do usuário:** {st.session_state.jogada_usuario}")
        st.markdown(f"**Jogada do robô:** {st.session_state.jogada_agente.capitalize()}")
        st.markdown(f"**Resultado:** {st.session_state.resultado}")
        st.markdown(f"**Confiança do modelo:** {st.session_state.confianca:.2f}%")

        if st.button("🔄 Próxima rodada", use_container_width=True):
            try:
                posicionar_em_papel()
                st.session_state.estado_jogo = "aguardando"
                st.session_state.jogada_agente = None
                st.rerun()
            except Exception as erro:
                st.error(f"Erro ao reiniciar rodada: {erro}")


# ==========================================
# COLUNA 2 - CÂMERA AO VIVO E RECONHECIMENTO AUTOMÁTICO
# ==========================================

with col2:
    st.markdown("### 📷 Reconhecimento ao vivo da mão")

    if st.session_state.estado_jogo == "jogando":
        placeholder_video = st.empty()
        placeholder_barra = st.progress(0, text="Aguardando detecção da mão...")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        frames_estaveis = 0
        gesto_anterior = None
        FRAMES_NECESSARIOS = 8
        CONFIANCA_MINIMA = 0.55

        try:
            while cap.isOpened() and st.session_state.estado_jogo == "jogando":
                ret, frame = cap.read()
                if not ret:
                    st.error("Não foi possível acessar a webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                deteccao = detector_visual.detect(mp_image)

                if deteccao.hand_landmarks:
                    landmarks = deteccao.hand_landmarks[0]
                    h, w, _ = frame.shape
                    for lm in landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

                    jogada_detectada, confianca = classificar_landmarks(landmarks)

                    if confianca >= CONFIANCA_MINIMA:
                        if jogada_detectada == gesto_anterior:
                            frames_estaveis += 1
                        else:
                            gesto_anterior = jogada_detectada
                            frames_estaveis = 1

                        progresso = min(int((frames_estaveis / FRAMES_NECESSARIOS) * 100), 100)
                        placeholder_barra.progress(
                            progresso,
                            text=f"Associando: {jogada_detectada} ({confianca * 100:.1f}%) — Confirmando {progresso}%"
                        )
                        cor = (0, 255, 0)
                    else:
                        frames_estaveis = 0
                        gesto_anterior = None
                        placeholder_barra.progress(
                            0,
                            text=f"Baixa confiança: {jogada_detectada} ({confianca * 100:.1f}%) — Mantenha a mão firme"
                        )
                        cor = (0, 255, 255)

                    cv2.putText(
                        frame,
                        f"{jogada_detectada.upper()} ({confianca * 100:.0f}%)",
                        (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cor,
                        3
                    )
                else:
                    frames_estaveis = 0
                    gesto_anterior = None
                    placeholder_barra.progress(0, text="Nenhuma mão detectada — Evite luz forte atrás da mão")

                frame_exibicao = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder_video.image(frame_exibicao, channels="RGB", use_container_width=True)

                if frames_estaveis >= FRAMES_NECESSARIOS:
                    jogada_agente = st.session_state.jogada_agente
                    resultado = comparar_jogadas(gesto_anterior.lower(), jogada_agente)

                    movimentar_arduino(jogada_agente)
                    registrar_partida(gesto_anterior, jogada_agente, resultado)

                    st.session_state.jogada_usuario = gesto_anterior
                    st.session_state.resultado = resultado
                    st.session_state.confianca = confianca * 100
                    st.session_state.estado_jogo = "resultado"
                    break

                time.sleep(0.03)

        finally:
            cap.release()

        if st.session_state.estado_jogo == "resultado":
            st.rerun()


# ==========================================
# 9. HISTÓRICO DE PARTIDAS
# ==========================================

st.divider()
st.markdown("### 📊 Histórico das partidas")

try:
    partidas = consultar_ultimas_partidas(10)

    if partidas:
        dados = []
        for partida in partidas:
            dados.append({
                "ID": partida[0],
                "Usuário": partida[1].capitalize(),
                "Agente": partida[2].capitalize(),
                "Resultado": partida[3],
                "Origem": partida[4],
                "Data": partida[5]
            })

        st.dataframe(
            dados,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Ainda não existem partidas registradas.")

except Exception as erro:
    st.error(f"Erro ao consultar histórico: {erro}")

st.sidebar.markdown("### ⚙️ Status do sistema")

if arduino is not None and arduino.is_open:
    st.sidebar.success("Arduino conectado")
else:
    st.sidebar.warning("Arduino desconectado")

st.sidebar.caption("RPS Vision - Visão Computacional com MediaPipe & MLP")