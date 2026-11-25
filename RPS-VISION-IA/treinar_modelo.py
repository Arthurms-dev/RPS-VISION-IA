import sqlite3
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from database import inicializar_banco, registrar_partida
from agent import escolher_jogada_agente

DB_PATH = "dataset_treino.db"
CLASSES = ["pedra", "papel", "tesoura"]

with sqlite3.connect(DB_PATH) as conn:
    dados = np.array(conn.execute("SELECT * FROM dataset_treino").fetchall())

X = dados[:, 1:43].astype(np.float32)  
y = dados[:, 43].astype(np.int32)      

print(f"Total de amostras carregadas: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

modelo = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAcurácia na base de teste: {acc * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=CLASSES))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(modelo, "modelo_mlp.joblib")
print("\nModelo salvo em 'modelo_mlp.joblib'!")

inicializar_banco()
regras = {"pedra": "tesoura", "papel": "pedra", "tesoura": "papel"}
todas_predicoes = modelo.predict(X)

for pred in todas_predicoes:
    jogada_user = CLASSES[pred]
    jogada_ai = escolher_jogada_agente()
    if jogada_user == jogada_ai:
        res = "Empate"
    elif regras[jogada_user] == jogada_ai:
        res = "Humano ganhou"
    else:
        res = "Robô ganhou"
    registrar_partida(jogada_user, jogada_ai, res, origem="teste_mlp")

print(f"{len(todas_predicoes)} testes registrados no banco de dados!")