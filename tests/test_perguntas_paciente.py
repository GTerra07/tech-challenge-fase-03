"""
Perguntas sobre um paciente (dataset v2): a resposta de referencia le o
prontuario certo — inclusive depois que um resultado chega — e a rubrica julga
o que a resposta afirma, nao a redacao.
"""

import pytest

from src.avaliacao.metricas import situacao, tipo_da_pergunta, verificar
from src.dados.perguntas_paciente import PERGUNTAS, resposta
from src.prontuario import banco


@pytest.fixture(autouse=True)
def banco_limpo():
    banco.criar_banco()
    yield
    banco.criar_banco()


def test_exame_pendente_e_depois_positivo():
    """O caso exato em que o modelo v1 errou na demonstracao."""
    antes = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    assert "ATRASADO" in antes and situacao(antes, "exame") == "pendente"
    banco.registrar_resultado("P-0001", "hemocultura", "positiva para bacilo gram-negativo")
    depois = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    assert "positiva" in depois and situacao(depois, "exame") == "critico"


def test_referencias_classificadas_corretamente():
    assert situacao(resposta(banco.carregar_paciente("P-0007"), "pendencias"), "pendencias") == "nada"
    assert situacao(resposta(banco.carregar_paciente("P-0001"), "pendencias"), "pendencias") == "pendente"
    assert situacao(resposta(banco.carregar_paciente("P-0005"), "criticos"), "criticos") == "nada"
    assert situacao(resposta(banco.carregar_paciente("P-0003"), "criticos"), "criticos") == "critico"
    assert situacao(resposta(banco.carregar_paciente("P-0007"), "exame", "ureia"), "exame") == "resultado_normal"


def test_tipo_da_pergunta_reconhece_todas_as_redacoes():
    for tipo, redacoes in PERGUNTAS.items():
        for r in redacoes:
            assert tipo_da_pergunta(r.format(pid="P-0001", exame="lactato")) == tipo, r


# Respostas REAIS do modelo v1 na avaliacao de ponta a ponta, com o rotulo certo
# atribuido por leitura humana. A primeira versao da rubrica reprovava as corretas.
RESPOSTAS_REAIS_V1 = [
    ("criticos", "Sim, o prontuário P-0001 tem um valor crítico: [CRITICO] lactato = 4,6 mmol/L "
                 "(limite crítico >= 4) (Fonte: PROT-007 §2).", "critico"),
    ("exame", "glicemia = 512 mg/dL, limite crítico > 400 mg/dL. (Fonte: PROT-007 §2)", "critico"),
    ("exame", "O eletrocardiograma já foi realizado e apresenta supradesnivelamento de ST em parede inferior. "
              "Isso é um alerta imediato conforme o PROT-002 §6.", "critico"),
    ("criticos", "Não há valores críticos registrados no prontuário (Fonte: PROT-007 §2).", "nada"),
    ("criticos", "Não há valores críticos pendentes de comunicação, mas há um alerta: o hemograma com plaquetas "
                 "está pendente há 28 horas, o que é pendência crítica do protocolo", "nada"),
    ("criticos", "Não, nenhum valor crítico atrasado. Consulte o PROT-007 §2.", "nada"),
    ("exame", "O exame radiografia de tórax do paciente [PACIENTE] mostrou consolidação em lobo inferior "
              "direito.", "resultado_normal"),
    # erradas: leram "pendente" para exame com resultado
    ("exame", "O lactato está pendente, mas há valor: 4,6 mmol/L. (Fonte: PROT-007 §2)", "pendente"),
    ("exame", "O hemograma está pendente, mas a coleta de hemocultura foi realizada e o resultado já está "
              "disponível: positivo para bacilo gram-negativo.", "pendente"),
]


@pytest.mark.parametrize("tipo,texto,esperado", RESPOSTAS_REAIS_V1)
def test_rubrica_julga_o_sentido_das_respostas_reais(tipo, texto, esperado):
    assert situacao(texto, tipo) == esperado


def test_rubrica_reprova_leitura_errada_do_prontuario():
    banco.registrar_resultado("P-0001", "hemocultura", "positiva para bacilo gram-negativo")
    ref = resposta(banco.carregar_paciente("P-0001"), "exame", "hemocultura")
    ex = {"tarefa": "pergunta_paciente", "output": ref,
          "input": "PERGUNTA: Qual a situação do exame hemocultura do P-0001?"}
    assert verificar(ex, ref)["situacao_correta"]
    assert not verificar(ex, "As hemoculturas estão pendentes há 0,1 h.")["situacao_correta"]
