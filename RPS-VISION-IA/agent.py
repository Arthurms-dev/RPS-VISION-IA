import random

JOGADAS = ["pedra", "papel", "tesoura"]


def escolher_jogada_agente():
    return random.choice(JOGADAS)