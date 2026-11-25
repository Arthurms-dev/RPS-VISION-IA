# RPS Vision — Pedra, Papel e Tesoura com IA e Mão Robótica

Sistema completo de **Visão Computacional**, **Aprendizado de Máquina**, **Persistência Híbrida de Dados** e **Robótica** que interpreta gestos manuais em tempo real pela webcam, decide a jogada da máquina de forma estocástica e imparcial, registra o histórico de partidas e aciona fisicamente uma mão animatrônica controlada por Arduino.

---

## Visão Geral

O **RPS Vision** nasceu do objetivo de construir uma interface humano-máquina interativa ponta a ponta, unindo software de Inteligência Artificial a um atuador mecânico real. Em vez de processar imagens brutas (pixels suscetíveis a variações de fundo e iluminação), a solução extrai a geometria articulada da mão humana e utiliza uma Rede Neural Artificial para classificar as jogadas com baixa latência.

### Principais Funcionalidades
* **Reconhecimento Contínuo ao Vivo:** Detecção automática pela webcam sem necessidade de captura manual de fotos, confirmando a jogada após estabilidade contínua de frames.
* **Invariância de Escala e Posição:** Normalização matemática dos 21 pontos articulados da mão em relação ao pulso e à abertura máxima, permitindo jogar de perto ou de longe da câmera.
* **Agente Imparcial (Equilíbrio Estocástico):** Sorteio uniforme (`33,33%` por jogada) livre de viés histórico.
* **Banco de Dados Resiliente:** Persistência primária em nuvem (**PostgreSQL / NeonDB**) com *fallback* automático transparente para **SQLite** local.
* **Sincronização Físico-Digital:** Acionamento de servomotores de rotação contínua (360°) no Arduino Uno com controle lógico de fim de curso para proteger os cabos de tração dos dedos.

---

## Arquitetura da Pipeline

```text
┌───────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────────┐
│  Webcam (OpenCV)  │ ──▶ │ MediaPipe (21 Landmarks X,Y) │ ──▶ │ MLPClassifier (Scikit-Learn)│
└───────────────────┘     └──────────────────────────────┘     └──────────────┬──────────────┘
                                                                              │
┌─────────────────────────────┐     ┌─────────────────────────┐               ▼
│ Arduino Uno (2 Servos 360°) │ ◀── │ PySerial (b'1',b'2',b'3')│ ◀── ┌────────────────────────┐
└─────────────────────────────┘     └─────────────────────────┘     │ Orquestrador Streamlit │
                                                                    └───────────┬────────────┘
                                                                                ▼
                                                                    ┌────────────────────────┐
                                                                    │ PostgreSQL  /  SQLite  │
                                                                    └────────────────────────┘
```

### Etapas do Fluxo de Dados
1. **Captura e Extração (`OpenCV` + `MediaPipe Tasks`):** O modelo `hand_landmarker.task` localiza a mão no quadro de vídeo e extrai 21 pontos anatômicos $(X, Y)$. As coordenadas são subtraídas da base do pulso $(x_0, y_0)$ e divididas pelo maior deslocamento absoluto, gerando **42 atributos numéricos** normalizados entre $-1$ e $1$.
2. **Construção da Base e Treinamento (`70/30`):** A base própria (`dataset_treino.db`) reúne amostras balanceadas das três classes (`0 = Pedra`, `1 = Papel`, `2 = Tesoura`), divididas de forma estratificada em **70% treino** e **30% teste**. Uma rede **`MLPClassifier`** com camadas ocultas `(64, 32)` e ativação `ReLU` aprende os padrões geométricos e é serializada em `modelo_mlp.joblib`.
3. **Inferência e Decisão:** Durante o jogo no Streamlit, cada frame válido é classificado via `predict_proba()`. Ao atingir confiança mínima e estabilidade por 8 frames seguidos, o sistema compara o resultado com a escolha prévia do `agent.py`.
4. **Atuação Mecânica e Log:** O comando correspondente (`b'1'`, `b'2'` ou `b'3'`) é enviado via porta serial (`9600 baud`) ao Arduino Uno, enquanto o histórico completo é gravado na tabela `partidas`.

---

## Arquitetura de Hardware (Arduino)

A mão animatrônica utiliza **2 servomotores de rotação contínua (360°)** conectados aos pinos digitais PWM do **Arduino Uno**, dividindo o tracionamento dos 5 dedos:

| Pino Digital | Identificador no Firmware | Dedos Controlados | Comportamento nas Jogadas |
| :--- | :--- | :--- | :--- |
| **Pino 8** | `servo3Dedos` | Polegar, Anelar e Mindinho (3 dedos) | Puxa em **Pedra (`'1'`)** e **Tesoura (`'3'`)**; solta em **Papel (`'2'`)**. |
| **Pino 9** | `servo2Dedos` | Indicador e Médio (2 dedos) | Puxa apenas em **Pedra (`'1'`)**; permanece solto em **Papel (`'2'`)** e **Tesoura (`'3'`)**. |

* **Proteção Mecânica:** O firmware mantém variáveis de estado (`dedos3Fechados` e `dedos2Fechados`) para impedir que o motor gire duas vezes consecutivas no mesmo sentido e rompa as linhas de nylon.
* **Alimentação Recomendada:** Utilizar fonte externa de 5V dedicada para os servos (com `GND` comum interligado ao Arduino) para evitar quedas de tensão (*brownout*) durante o acionamento simultâneo em *Pedra* e *Papel*.

---

## Mapeamento de Scripts e Arquivos

```text
RPS-VISION-IA/
├── arduino/
│   └── mao_robotica.ino       # Firmware C++ de controle dos 2 servos 360° (Pinos 8 e 9)
├── main.py                    # Orquestrador Streamlit, vídeo ao vivo e comunicação serial
├── classifier.py              # Carregamento do modelo MLP e inferência em tempo real
├── agent.py                   # Lógica de decisão estocástica imparcial do agente
├── database.py                # Persistência híbrida PostgreSQL / SQLite e consultas SQL
├── coletar_dataset.py         # Coleta supervisionada de landmarks normalizados via webcam
├── treinar_modelo.py          # Split 70/30, treino do MLPClassifier, métricas e logs em lote
├── modelo_mlp.joblib          # Pesos serializados da Rede Neural treinada
├── hand_landmarker.task       # Modelo binário pré-treinado do MediaPipe Tasks
├── dataset_treino.db          # Banco SQLite com as amostras rotuladas de treinamento
├── requirements.txt           # Manifesto de dependências Python
├── .env.example               # Modelo de variáveis de ambiente
└── .gitignore                 # Regras de exclusão de credenciais e arquivos temporários
```

### Descrição Detalhada dos Módulos
* **`coletar_dataset.py`:** Captura frames da webcam, desenha os 21 *landmarks* na tela e salva os vetores normalizados de 42 posições no banco `dataset_treino.db`.
* **`treinar_modelo.py`:** Executa o `train_test_split` (70% treino / 30% teste), treina o `MLPClassifier`, imprime **Acurácia**, **Precision**, **Recall**, **F1-Score** e **Matriz de Confusão**, exporta `modelo_mlp.joblib` e executa validações em lote (`origem = 'teste_mlp'`) na tabela `partidas`.
* **`classifier.py`:** Encapsula a extração normalizada de características e calcula as probabilidades por classe em tempo real.
* **`agent.py`:** Implementa `escolher_jogada_agente()` utilizando `random.choice(["pedra", "papel", "tesoura"])`.
* **`database.py`:** Gerencia a conexão com PostgreSQL (`psycopg`) e alterna automaticamente para `rps_vision_local.db` (`sqlite3`) caso a rede externa esteja indisponível.
* **`main.py`:** Interface web interativa dividida em painel de controle de rodada, feed de vídeo ao vivo com barra de confirmação e tabela histórica das últimas partidas.

---

## Instalação e Configuração

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/RPS-VISION-IA.git
cd RPS-VISION-IA
```

### 2. Criar e Ativar o Ambiente Virtual

```bash
python -m venv venv

# Windows (PowerShell / CMD):
venv\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente
Copie o arquivo `.env.example` para `.env` na raiz do projeto e ajuste conforme sua máquina:
```env
DATABASE_URL=postgresql://usuario:senha@host/banco?sslmode=require
ARDUINO_PORT=COM4
MODEL_PATH=modelo_mlp.joblib
```
> **Nota:** Se `DATABASE_URL` for deixado em branco ou estiver sem conexão com a internet, o sistema utilizará automaticamente o banco SQLite local (`rps_vision_local.db`).

---

## Como Executar o Projeto

### Passo 1: Preparar o Hardware (Opcional para testes apenas em software)
1. Carregue o código `arduino/mao_robotica.ino` na placa **Arduino Uno** pela Arduino IDE.
2. Verifique a porta serial atribuída (ex.: `COM4`) e **feche o Monitor Serial** da Arduino IDE antes de iniciar o Python para não bloquear a porta.

### Passo 2: Iniciar a Aplicação Principal
```bash
streamlit run main.py
```
1. Clique em **Iniciar rodada** (a mão robótica posiciona-se em *Papel* e o agente trava sua escolha).
2. Posicione a mão diante da webcam (evitando luz forte diretamente atrás da mão).
3. Mantenha o gesto firme até a barra atingir **100%** para confirmar a jogada, acionar os servos e registrar o resultado.

### Passo 3: Coletar Novos Dados e Retreinar (Opcional)
Caso queira criar um novo conjunto de treinamento do zero:
1. Remova o arquivo `dataset_treino.db` antigo.
2. Execute `python coletar_dataset.py` três vezes (`0` = Pedra, `1` = Papel, `2` = Tesoura), pressionando **`S`** quando os pontos verdes aparecerem sobre a mão.
3. Execute `python treinar_modelo.py` para gerar um novo `modelo_mlp.joblib` e visualizar as métricas de avaliação no terminal.

---

## Stack Tecnológica

* **Linguagens:** Python 3.10+, C++ (Arduino)
* **Visão Computacional:** OpenCV (`opencv-python`), Google MediaPipe Tasks (`mediapipe`)
* **Machine Learning:** Scikit-Learn (`MLPClassifier`), Joblib, NumPy
* **Interface Web:** Streamlit
* **Banco de Dados:** PostgreSQL (`psycopg`), SQLite3
* **Comunicação Hardware:** PySerial (`pyserial`), Python-Dotenv