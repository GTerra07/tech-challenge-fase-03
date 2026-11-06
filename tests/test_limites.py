"""Barreiras deterministicas de entrada e saida (requisito 3, limites de atuacao)."""

import pytest

from src.dados import conhecimento_interno as ci
from src.seguranca.limites import triagem_entrada, verificar_saida


@pytest.mark.parametrize("modelo", ci.PEDIDOS_PRESCRICAO)
def test_todos_os_pedidos_indevidos_do_dataset_sao_barrados(modelo):
    texto = modelo.format(med="vancomicina", leito="21A", pid="paciente P-0001")
    assert not triagem_entrada(texto).permitido, texto


@pytest.mark.parametrize("pergunta", [item.perguntas[0] for item in ci.FAQ])
def test_perguntas_legitimas_passam(pergunta):
    """Nenhuma pergunta frequente legitima pode ser barrada (falso positivo)."""
    assert triagem_entrada(pergunta).permitido, pergunta


def test_saida_com_dose_e_reprovada():
    v = verificar_saida("Faça enoxaparina 40 mg (Fonte: PROT-006 §4)", {"PROT-006 §4"})
    assert not v.aprovada and "contém dose de medicamento" in v.problemas


def test_saida_sem_fonte_e_reprovada():
    assert not verificar_saida("Repita o lactato em 2 a 4 horas.", {"PROT-001 §4"}).aprovada


def test_fonte_fora_do_contexto_e_reprovada():
    v = verificar_saida("Repita em 2 a 4 horas (Fonte: PROT-002 §3)", {"PROT-001 §4"})
    assert not v.aprovada and v.fontes_nao_recuperadas == ["PROT-002 §3"]


def test_sugestao_sem_marca_de_rascunho_e_reprovada():
    v = verificar_saida("Coletar hemocultura (PROT-001 §3)", {"PROT-001 §3"}, exige_rascunho=True)
    assert "sugestão não marcada como rascunho para validação médica" in v.problemas


def test_resposta_correta_e_aprovada():
    assert verificar_saida("Lactato > 2 deve ser repetido em 2 a 4 horas. (Fonte: PROT-001 §4)",
                           {"PROT-001 §4", "PROT-001 §3"}).aprovada
