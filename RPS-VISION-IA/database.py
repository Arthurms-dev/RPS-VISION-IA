import os
import sqlite3
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_DB = "rps_vision_local.db"
USANDO_SQLITE = False


def conectar_banco():
    global USANDO_SQLITE
    if DATABASE_URL and not USANDO_SQLITE:
        try:
            return psycopg.connect(DATABASE_URL, connect_timeout=5)
        except Exception as erro:
            print(f"[Aviso] PostgreSQL indisponível ({erro}). Alternando para SQLite local...")
            USANDO_SQLITE = True
    else:
        USANDO_SQLITE = True

    return sqlite3.connect(SQLITE_DB)


def inicializar_banco():
    conn = conectar_banco()
    try:
        cursor = conn.cursor()
        if USANDO_SQLITE:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS partidas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    jogada_usuario TEXT NOT NULL,
                    jogada_agente TEXT NOT NULL,
                    resultado TEXT NOT NULL,
                    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    origem TEXT NOT NULL DEFAULT 'real'
                );
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS partidas (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    jogada_usuario TEXT NOT NULL,
                    jogada_agente TEXT NOT NULL,
                    resultado TEXT NOT NULL,
                    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    origem TEXT NOT NULL DEFAULT 'real'
                );
            """)
        conn.commit()
    finally:
        conn.close()


def buscar_historico():
    conn = conectar_banco()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT LOWER(jogada_usuario), COUNT(*)
            FROM partidas
            GROUP BY LOWER(jogada_usuario);
        """)
        return cursor.fetchall()
    finally:
        conn.close()


def buscar_sequencia_jogadas(limite=100):
    conn = conectar_banco()
    try:
        cursor = conn.cursor()
        placeholder = "?" if USANDO_SQLITE else "%s"
        cursor.execute(f"""
            SELECT LOWER(jogada_usuario)
            FROM partidas
            WHERE origem = 'real'
            ORDER BY id DESC
            LIMIT {placeholder};
        """, (limite,))
        return [row[0] for row in cursor.fetchall()][::-1]
    finally:
        conn.close()


def registrar_partida(jogada_usuario, jogada_agente, resultado, origem="real"):
    conn = conectar_banco()
    try:
        cursor = conn.cursor()
        placeholder = "?" if USANDO_SQLITE else "%s"
        cursor.execute(f"""
            INSERT INTO partidas (
                jogada_usuario,
                jogada_agente,
                resultado,
                origem
            )
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder});
        """, (
            jogada_usuario.lower(),
            jogada_agente.lower(),
            resultado,
            origem
        ))
        conn.commit()
    finally:
        conn.close()


def consultar_ultimas_partidas(limite=10):
    conn = conectar_banco()
    try:
        cursor = conn.cursor()
        placeholder = "?" if USANDO_SQLITE else "%s"
        cursor.execute(f"""
            SELECT
                id,
                jogada_usuario,
                jogada_agente,
                resultado,
                origem,
                data_hora
            FROM partidas
            ORDER BY id DESC
            LIMIT {placeholder};
        """, (limite,))
        return cursor.fetchall()
    finally:
        conn.close()