"""
Formato dos exemplos de treino e dos prompts de inferencia — definido uma vez.

A Aula 2 de Fine-Tuning usa o template Alpaca (instruction / input / response)
com o token EOS no fim, para o modelo aprender a parar. O mesmo template e
usado: (1) aqui, para gerar o dataset; (2) no notebook do Colab, para treinar e
avaliar; (3) no Modelfile do Ollama, para servir o modelo; (4) no assistente,
para montar o prompt. Se o formato da inferencia diferir do treino, o modelo
ajustado se comporta como se nao tivesse sido treinado.
"""

from __future__ import annotations

ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- Instrucoes por tarefa (portugues para o conteudo interno) ---
INSTRUCAO_PERGUNTA = (
    "Você é o assistente virtual médico do Hospital Universitário. Responda à pergunta do médico usando apenas "
    "os trechos de protocolo do contexto e cite a fonte no formato (Fonte: CÓDIGO §seção). Se o contexto não "
    "contiver a resposta, diga isso. Nunca prescreva, nunca indique dose e nunca dispense a validação médica."
)
INSTRUCAO_CASO = (
    "Você é o assistente virtual médico do Hospital Universitário. Com os dados do paciente, os achados calculados "
    "pelas regras e os trechos de protocolo, escreva o resumo, os alertas para a equipe e a sugestão como rascunho "
    "para validação médica, citando as fontes. Nunca indique medicamento nem dose."
)
INSTRUCAO_LAUDO = (
    "Redija o rascunho do laudo no modelo interno do hospital (MOD-001) a partir dos achados ditados, "
    "sem acrescentar achados."
)
INSTRUCAO_RECEITA = (
    "Formate no modelo interno (MOD-002) a prescrição já definida pelo médico, como rascunho para conferência. "
    "Não altere, não remova e não acrescente medicamentos, doses ou posologias."
)
# --- Instrucoes dos datasets publicos (em ingles, como os dados) ---
INSTRUCAO_PUBMEDQA = (
    "Answer the biomedical research question using only the abstract context. Start the answer with yes, no or "
    "maybe, then give a brief justification."
)
INSTRUCAO_MEDQUAD = "Answer the health question in clear, factual language and state the source of the information."

PEDIDO_CASO = "PEDIDO: Analise o caso e sugira os próximos passos para validação médica."


def entrada_pergunta(contexto: str, pergunta: str) -> str:
    return f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"


def entrada_caso(dados_paciente: str, achados: str, contexto: str) -> str:
    return f"DADOS DO PACIENTE:\n{dados_paciente}\n\n{achados}\n\nCONTEXTO:\n{contexto}\n\n{PEDIDO_CASO}"


FONTE_PRONTUARIO = "PRONTUÁRIO"


def contexto_com_prontuario(paciente_id: str, dados_paciente: str, contexto_protocolos: str) -> str:
    """
    Contexto de uma pergunta sobre um paciente: o prontuario atualizado como um
    bloco citavel, seguido dos trechos de protocolo. Usado pelo grafo de perguntas
    e pelo dataset — o modelo v1 nunca tinha visto este formato e errou a
    leitura do prontuario na demonstracao; o v2 treina exatamente com ele.
    """
    return f"[{FONTE_PRONTUARIO} {paciente_id}] Prontuário atualizado\n{dados_paciente}\n\n{contexto_protocolos}"


def prompt(instrucao: str, entrada: str, resposta: str = "") -> str:
    """Prompt completo; com resposta vazia, e o prompt de inferencia."""
    return ALPACA.format(instrucao, entrada, resposta)


def com_revisao(entrada: str, problemas: list[str]) -> str:
    """
    Nova tentativa com a critica da verificacao automatica (autocorrecao no
    estilo Reflexion, ChatGPT Aula 4). Sem isso, com temperatura baixa, o modelo
    devolve a mesma resposta reprovada — medido com o modelo substituto.
    """
    if not problemas:
        return entrada
    return (f"{entrada}\n\nREVISÃO OBRIGATÓRIA: a resposta anterior foi reprovada pela verificação automática "
            f"({'; '.join(problemas)}). Responda novamente corrigindo isso.")
