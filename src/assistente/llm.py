"""
Integracao da LLM customizada com o LangChain (requisito 2).

O modelo ajustado no Colab e servido localmente pelo Ollama (caminho da Aula 5
de LangChain) e entra numa cadeia LCEL:

    ChatPromptTemplate (instrucao como system, entrada como human)
        | ChatOllama("assistente-medico")
        | StrOutputParser

O Modelfile do Ollama monta, a partir dessas duas mensagens, exatamente o
template Alpaca do treino (`src/fine_tuning/formato.py`). Todo prompt enviado e
toda resposta recebida vao juntos para o log de auditoria.

Para testes, qualquer chat model do LangChain pode ser injetado no lugar do
Ollama (ex.: `FakeListChatModel`) — o resto do pipeline nao muda.
"""

from __future__ import annotations

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import MODELO_AJUSTADO, OLLAMA_BASE_URL, TEMPERATURA
from src.fine_tuning import formato
from src.seguranca.auditoria import Auditoria

PROMPT = ChatPromptTemplate.from_messages([("system", "{instrucao}"), ("human", "{entrada}")])


def criar_llm_ajustada(modelo: str = MODELO_AJUSTADO, max_tokens: int = 700):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURA,
                      num_predict=max_tokens, num_ctx=4096)


class Redator:
    """Cadeia LangChain com a LLM ajustada + registro de cada chamada."""

    def __init__(self, llm=None, auditoria: Auditoria | None = None):
        self.llm = llm if llm is not None else criar_llm_ajustada()
        self.cadeia = PROMPT | self.llm | StrOutputParser()
        self.auditoria = auditoria or Auditoria()
        self.nome_modelo = getattr(self.llm, "model", type(self.llm).__name__)

    def redigir(self, instrucao: str, entrada: str, execucao: str, tarefa: str) -> str:
        inicio = time.perf_counter()
        resposta = self.cadeia.invoke({"instrucao": instrucao, "entrada": entrada}).strip()
        # Alguns servidores devolvem o marcador de parada junto; cortamos ali.
        resposta = resposta.split("### Instruction:")[0].strip()
        self.auditoria.registrar(
            "llm", execucao, tarefa=tarefa, modelo=self.nome_modelo,
            ms=round((time.perf_counter() - inicio) * 1000, 1),
            prompt=formato.prompt(instrucao, entrada), resposta=resposta)
        return resposta
