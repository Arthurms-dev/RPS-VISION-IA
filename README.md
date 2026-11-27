# 🤖 RPS Vision — Pedra, Papel e Tesoura com IA e Mão Robótica

Sistema completo de **Visão Computacional**, **Machine Learning**, **Persistência Híbrida de Dados** e **Robótica** que reconhece gestos manuais ao vivo pela webcam, decide a jogada da máquina de forma estocástica e imparcial, armazena o histórico e as métricas de validação em banco de dados e aciona fisicamente uma mão animatrônica controlada por Arduino.

---

## 📋 Visão Geral e Motivação

O **RPS Vision** foi desenvolvido para integrar uma pipeline completa de Inteligência Artificial a um atuador mecânico físico. Em vez de processar pixels brutos — sensíveis a fundo e iluminação —, o sistema extrai os 21 pontos articulados (*landmarks*) da mão humana, aplica normalização espacial e de escala e utiliza uma Rede Neural Artificial (`MLPClassifier`) para classificar as jogadas em tempo real.

### Principais Funcionalidades

* **Reconhecimento Contínuo ao Vivo:** Captura em tempo real pela webcam (`OpenCV`) sem necessidade de fotos manuais, confirmando o gesto automaticamente após 8 frames estáveis consecutivos.
* **Invariância de Escala e Posição:** Normalização matemática dos 21 pontos da mão em relação ao pulso e à abertura máxima, permitindo reconhecer gestos tanto de perto quanto de longe da câmera.
* **Agente Estocástico Imparcial:** Escolha aleatória uniforme (`33,33%` por jogada) sem viés histórico.
* **Persistência Híbrida:** Grava partidas reais e validações em lote em **PostgreSQL (NeonDB)** na nuvem com *fallback* automático para **SQLite** local (`rps_vision_local.db`).
* **Sincronização Físico-Digital:** Acionamento de 2 servomotores de rotação contínua (360°) no **Arduino Uno** com controle lógico de estado para evitar sobre-enrolamento das linhas de tração dos dedos.

---

## 🧠 Arquitetura da Pipeline

```text
┌────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────────┐
│ Webcam ao Vivo (OpenCV)│ ──▶ │ MediaPipe (21 Landmarks X,Y) │ ──▶ │ MLPClassifier (Scikit-Learn)│
└────────────────────────┘     └──────────────────────────────┘     └──────────────┬──────────────┘
                                                                                   │
┌─────────────────────────────┐     ┌─────────────────────────┐                    ▼
│ Arduino Uno (2 Servos 360°) │ ◀── │ PySerial (b'1',b'2',b'3')│ ◀── ┌────────────────────────┐
└─────────────────────────────┘     └─────────────────────────┘     │ Orquestrador Streamlit │
                                                                    └───────────┬────────────┘
                                                                                ▼
                                                                    ┌────────────────────────┐
                                                                    │ PostgreSQL  /  SQLite  │
                                                                    └────────────────────────┘
```

### Fluxo de Dados

1. **Extração de Características (`MediaPipe Tasks`):** O modelo `hand_landmarker.task` localiza a mão no quadro de vídeo e extrai 21 pontos anatômicos $(X, Y)$. As coordenadas são subtraídas do pulso $(x_0, y_0)$ e divididas pelo maior deslocamento absoluto, gerando **42 atributos numéricos** normalizados entre $-1$ e $1$.
2. **Base de Dados e Treinamento (`70/30`):** O script `coletar_dataset.py` salva os vetores normalizados em `dataset_treino.db` (além de capturas anotadas em `fotos_dataset/`). O `treinar_modelo.py` divide os dados de forma estratificada em **70% treino** e **30% teste**, treina o `MLPClassifier` e exporta o modelo para `modelo_mlp.joblib`.
3. **Inferência e Decisão:** Durante a rodada no Streamlit, cada frame válido é avaliado via `predict_proba()`. Ao atingir confiança mínima ($\ge 55\%$) e estabilidade contínua, o sistema compara o gesto com a escolha prévia do `agent.py`.
4. **Atuação Mecânica e Log:** O comando serial (`b'1'`, `b'2'` ou `b'3'`) é enviado via `PySerial` (`9600 baud`) ao Arduino Uno, enquanto o resultado é registrado na tabela `partidas`.

---

## 🔧 Arquitetura de Hardware (Arduino)

A mão animatrônica utiliza **2 servomotores de rotação contínua (360°)** conectados aos pinos digitais PWM do **Arduino Uno**, dividindo o tracionamento dos 5 dedos:

| Pino Digital | Identificador no Firmware | Dedos Controlados | Comportamento nas Jogadas |
| :--- | :--- | :--- | :--- |
| **Pino 8** | `servo3Dedos` | Polegar, Anelar e Mindinho (3 dedos) | Puxa em **Pedra (`'1'`)** e **Tesoura (`'3'`)**; solta em **Papel (`'2'`)**. |
| **Pino 9** | `servo2Dedos` | Indicador e Médio (2 dedos) | Puxa apenas em **Pedra (`'1'`)**; permanece solto em **Papel (`'2'`)** e **Tesoura (`'3'`)**. |

* **Proteção Mecânica:** Variáveis de estado (`dedos3Fechados` e `dedos2Fechados`) impedem que o motor gire duas vezes consecutivas no mesmo sentido e arrebente as linhas de nylon.
* **Alimentação Recomendada:** Utilizar fonte externa de 5V dedicada para os servos (com `GND` comum interligado ao Arduino) para evitar quedas de tensão (*brownout*) durante o acionamento simultâneo em *Pedra* e *Papel*.

---

## 📁 Estrutura do Repositório

```text
RPS-VISION-IA/
├── arduino/
│   └── mao_robotica.ino       # Firmware C++ de controle dos 2 servos 360° (pinos 8 e 9)
├── main.py                    # Interface web (Streamlit), vídeo ao vivo e comunicação serial
├── classifier.py              # Carregamento do modelo MLP, normalização e inferência
├── agent.py                   # Lógica de decisão aleatória imparcial do agente
├── database.py                # Camada híbrida PostgreSQL / SQLite e consultas de histórico
├── coletar_dataset.py         # Coleta supervisionada de landmarks (DB) e imagens (.jpg)
├── treinar_modelo.py          # Divisão 70/30, treino do MLPClassifier, métricas e testes
├── modelo_mlp.joblib          # Modelo neural treinado e serializado
├── hand_landmarker.task       # Modelo pré-treinado do MediaPipe para detecção de mãos
├── dataset_treino.db          # Base de dados SQLite com vetores normalizados coletados
├── requirements.txt           # Dependências Python do projeto
├── .env.example               # Modelo de variáveis de ambiente
└── .gitignore                 # Arquivos locais e temporários ignorados pelo Git
```

### Descrição dos Módulos

* **`coletar_dataset.py`:** Captura frames da webcam, desenha os 21 *landmarks* na tela, salva os vetores normalizados de 42 posições no banco `dataset_treino.db` e exporta imagens anotadas em `fotos_dataset/`.
* **`treinar_modelo.py`:** Executa o `train_test_split` (70% treino / 30% teste), treina o `MLPClassifier`, exibe **Acurácia**, **Precision**, **Recall**, **F1-Score** e **Matriz de Confusão**, exporta `modelo_mlp.joblib` e registra validações em lote (`origem = 'teste_mlp'`) na tabela `partidas`.
* **`classifier.py`:** Encapsula a extração normalizada de características e calcula as probabilidades por classe em tempo real.
* **`agent.py`:** Implementa `escolher_jogada_agente()` utilizando `random.choice(["pedra", "papel", "tesoura"])`.
* **`database.py`:** Gerencia a conexão com PostgreSQL (`psycopg`) e alterna automaticamente para `rps_vision_local.db` (`sqlite3`) caso a rede externa esteja indisponível.
* **`main.py`:** Interface web interativa dividida em painel de controle de rodada, feed de vídeo ao vivo com barra de confirmação e tabela histórica das últimas partidas.

---

## ⚙️ Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/RPS-VISION-IA.git
cd RPS-VISION-IA
```

### 2. Criar e ativar o ambiente virtual
```bash
python -m venv venv

# Windows (PowerShell / CMD):
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente
Crie um arquivo `.env` na raiz do projeto a partir do `.env.example`:
```env
DATABASE_URL=postgresql://usuario:senha@host/banco?sslmode=require
ARDUINO_PORT=COM4
MODEL_PATH=modelo_mlp.joblib
```
> **Nota:** Caso `DATABASE_URL` não seja configurado ou a rede esteja indisponível, o sistema utiliza automaticamente o banco SQLite local `rps_vision_local.db`.

---

## 🚀 Como Executar

### Iniciar o Jogo (Aplicação Principal)
1. Carregue o firmware `arduino/mao_robotica.ino` na placa **Arduino Uno** e conecte-a na porta USB configurada no `.env` (**feche o Monitor Serial** da Arduino IDE antes de iniciar o Python para não bloquear a porta `COM`).
2. Execute a interface Streamlit:
```bash
streamlit run main.py
```
3. Clique em **▶️ Iniciar rodada** e posicione sua mão em frente à webcam até a barra de confirmação atingir **100%**.

### Coletar Amostras e Retreinar o Modelo (Opcional)
1. **Coleta de dados:** Execute o script abaixo para cada classe (`0` = Pedra, `1` = Papel, `2` = Tesoura) e pressione **`S`** para gravar 100 amostras (salva os vetores em `dataset_treino.db` e as capturas anotadas em `fotos_dataset/`):
```bash
python coletar_dataset.py
```
2. **Treinamento e validação:** Execute o script de treino para gerar o `modelo_mlp.joblib`, visualizar a acurácia e a matriz de confusão no terminal e registrar os testes na tabela `partidas`:
```bash
python treinar_modelo.py
```

---

## 🛠️ Tecnologias Utilizadas

* **Linguagens:** Python 3.10+, C++ (Arduino)
* **Visão Computacional:** OpenCV (`opencv-python`), Google MediaPipe Tasks (`mediapipe`)
* **Machine Learning:** Scikit-Learn (`MLPClassifier`), Joblib, NumPy
* **Interface Web:** Streamlit
* **Banco de Dados:** PostgreSQL (`psycopg`), SQLite3
* **Comunicação Hardware:** Arduino Uno, Servomotores 360°, PySerial (`pyserial`), Python-Dotenv