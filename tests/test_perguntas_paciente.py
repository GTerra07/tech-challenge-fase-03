"""
Perguntas sobre um paciente (dataset v2): a resposta de referencia le o
prontuario certo — inclusive depois que um resultado chega.
"""

from src.avaliacao.metricas import situacao, verificar
from src.dados.perguntas_paciente import resposta
from src.prontuario import banco


def test_exame_pendente_e_depois_positivo():
    """O caso exato em que o modelo v1 errou na demonstracao."""
    banco.criar_banco()
    antes = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    assert "ATRASADO" in antes and situacao(antes) == "pendente"
    banco.registrar_resultado("P-0001", "hemocultura", "positiva para bacilo gram-negativo")
    depois = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    assert "positiva" in depois and situacao(depois) == "critico"
    banco.criar_banco()


def test_paciente_sem_pendencias():
    banco.criar_banco()
    r = resposta(banco.carregar_paciente("P-0007"), "pendencias")
    assert situacao(r) == "nada"


def test_resultado_sem_criterio_critico():
    banco.criar_banco()
    r = resposta(banco.carregar_paciente("P-0007"), "exame", "ureia")
    assert situacao(r) == "resultado_normal" and "34 mg/dL" in r


def test_rubrica_reprova_leitura_errada_do_prontuario():
    banco.criar_banco()
    banco.registrar_resultado("P-0001", "hemocultura", "positiva para bacilo gram-negativo")
    ref = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    ex = {"tarefa": "pergunta_paciente", "output": ref}
    assert verificar(ex, ref)["situacao_correta"]
    assert not verificar(ex, "As hemoculturas estão pendentes há 0,1 h.")["situacao_correta"]
    banco.criar_banco()
