"""
Regras clinicas deterministicas: todo valor critico do PROT-007 dispara, todo
prazo vencido vira pendencia com a secao que o define.

O primeiro teste existe por causa de um bug real: a comparacao de resultado
qualitativo era sensivel a maiusculas ("supradesnivelamento de ST" contra o
valor em minusculas) e o infarto com supra de ST nunca virava alerta.
"""

import pytest

from src.prontuario import banco, regras


def _paciente(exames, protocolos=("PROT-002",)):
    return {"exames": exames, "protocolos_ativos": list(protocolos), "horas_desde_admissao": 1}


@pytest.mark.parametrize("valor", ["supradesnivelamento de ST em parede inferior",
                                   "SUPRADESNIVELAMENTO DE ST de V1 a V4"])
def test_supra_de_st_e_valor_critico(valor):
    p = _paciente([{"exame": "eletrocardiograma", "status": "resultado", "valor": valor}])
    assert [a.exame for a in regras.valores_criticos(p)] == ["eletrocardiograma"]


@pytest.mark.parametrize("exame,valor,critico", [
    ("potássio", 6.3, True), ("potássio", 5.9, False), ("potássio", 2.7, True),
    ("lactato", 4.0, True), ("lactato", 3.9, False),
    ("glicemia", 401, True), ("glicemia", 53, True), ("glicemia", 120, False),
    ("sódio", 119, True), ("hemoglobina", 6.9, True), ("plaquetas", 19000, True), ("ph", 7.15, True),
])
def test_limites_numericos_do_prot_007(exame, valor, critico):
    p = _paciente([{"exame": exame, "status": "resultado", "valor": valor}], protocolos=())
    assert bool(regras.valores_criticos(p)) is critico


def test_hemocultura_atrasada_e_critica_e_cita_a_secao():
    p = _paciente([{"exame": "hemocultura", "status": "pendente", "horas_desde_solicitacao": 2}], ("PROT-001",))
    pend = {a.exame: a for a in regras.exames_pendentes(p)}
    assert pend["hemocultura"].nivel == regras.CRITICO and pend["hemocultura"].fonte == "PROT-001 §4"


def test_dentro_do_prazo_e_informativo():
    p = _paciente([{"exame": "hemocultura", "status": "pendente", "horas_desde_solicitacao": 0.5}], ("PROT-001",))
    assert {a.exame: a.nivel for a in regras.exames_pendentes(p)}["hemocultura"] == regras.INFORMATIVO


def test_pacientes_de_demonstracao_tem_os_alertas_esperados():
    banco.criar_banco()
    esperado = {"P-0001": {"lactato", "hemocultura"}, "P-0002": {"eletrocardiograma", "troponina"},
                "P-0003": {"glicemia", "potássio"}, "P-0004": {"potássio", "potássio de controle"},
                "P-0005": {"hemograma"}, "P-0006": {"biópsia mamária"}, "P-0007": set()}
    for pid, exames in esperado.items():
        assert {a.exame for a in regras.alertas(banco.carregar_paciente(pid))} == exames, pid
