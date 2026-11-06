"""
Prova da anonimizacao (requisito 1: preparar os dados com anonimizacao).

O gerador registra cada dado pessoal que inseriu; aqui verificamos contra esse
gabarito que nenhum sobrevive — no texto anonimizado e no dataset final.
"""

import json
import random

from src.config import PASTA_PROCESSADO
from src.dados.gerador_interno import Pessoas, gerar_paciente, pii_do_paciente
from src.prontuario.resumo import contexto_paciente
from src.seguranca.anonimizacao import anonimizar, residuos_pii


def test_identificadores_estruturados():
    texto = ("Paciente Maria Souza Lima, CPF 123.456.789-09, prontuário 0045821, tel (11) 98765-4321, "
             "nascida em 12/03/1958, mora na Rua Augusta, 1200, e-mail maria@exemplo.com. Dr. Carlos Mendes CRM-SP 123456.")
    a = anonimizar(texto)
    for valor in ("Maria", "123.456.789-09", "0045821", "98765-4321", "12/03/1958", "Augusta", "maria@exemplo.com",
                  "Carlos", "123456"):
        assert valor not in a, valor
    assert "[PACIENTE]" in a and "[CPF]" in a and "[MEDICO]" in a


def test_nome_solto_sem_titulo():
    assert "Mariana" not in anonimizar("conversei com Mariana Costa sobre o caso")


def test_texto_clinico_nao_e_destruido():
    texto = "Lactato 4,2 mmol/L, plaquetas 18.400/mm³, PAM 63 mmHg, PROT-001 §3."
    assert anonimizar(texto) == texto


def test_contexto_de_paciente_nunca_leva_pii():
    rng = random.Random(7)
    pessoas = Pessoas(rng)
    for i in range(200):
        p = gerar_paciente(rng, pessoas, i)
        assert residuos_pii(contexto_paciente(p), pii_do_paciente(p)) == [], p["id"]


def test_dataset_final_sem_padroes_de_pii():
    for nome in ("treino", "validacao", "teste"):
        with open(PASTA_PROCESSADO / f"{nome}.jsonl", encoding="utf-8") as f:
            for linha in f:
                ex = json.loads(linha)
                if ex["origem"] == "interno":
                    assert residuos_pii(ex["input"] + ex["output"]) == [], ex["id"]
