"""
Anonimizacao de dados pessoais — ponto UNICO do projeto.

Tudo que sai de uma fonte com dado pessoal passa por aqui: as exportacoes
brutas que viram dataset de treino e o contexto de paciente que o assistente
monta para o modelo. Na Fase 2 a anonimizacao existia em dois lugares e eles
divergiram; aqui existe uma funcao, `anonimizar`, e um verificador,
`residuos_pii`, que a curadoria e os testes usam para provar o resultado.

A estrategia e substituir cada dado por um marcador de categoria ([PACIENTE],
[CPF], ...) em vez de apagar: o texto continua legivel e o modelo aprende que
esses marcadores existem, sem nunca ver o dado real.

Cobertura (dados identificadores da LGPD mais comuns em texto clinico):
CPF, cartao SUS, telefone, e-mail, CRM/COREN, numero de prontuario/atendimento,
datas completas, enderecos e nomes de pessoas (por contexto — "Dr.", "paciente",
"Nome:" — e por lista de prenomes, para nomes soltos no meio da frase).
"""

from __future__ import annotations

import re
import unicodedata

# Prenomes brasileiros frequentes. Um nome solto ("... conversei com Mariana
# Costa ontem") nao tem titulo antes; a lista pega esses casos. Nao precisa
# ser exaustiva: o teste de vazamento compara contra os nomes que o gerador
# realmente inseriu, e os nomes do gerador saem desta mesma populacao.
PRENOMES = {
    "Ana", "Maria", "Mariana", "Juliana", "Fernanda", "Patricia", "Aline", "Camila",
    "Beatriz", "Larissa", "Gabriela", "Renata", "Carla", "Luciana", "Vanessa",
    "Helena", "Isabela", "Leticia", "Sandra", "Tereza", "Rosa", "Lucia", "Paula",
    "Joao", "José", "Jose", "Carlos", "Paulo", "Pedro", "Lucas", "Marcos", "Rafael",
    "Bruno", "Felipe", "Gustavo", "Ricardo", "Rodrigo", "Eduardo", "Fernando",
    "Antonio", "Antônio", "Francisco", "Luiz", "Luis", "Sergio", "Sérgio", "Marcelo",
    "Andre", "André", "Diego", "Thiago", "Roberto", "Mateus", "Vinicius", "Leandro",
    "João", "Patrícia", "Letícia", "Lúcia", "Sérgio", "Vinícius",
}

_PALAVRA_NOME = r"[A-ZÀ-Ý][a-zà-ÿ]+"
_CONECTIVO = r"(?:\s+(?:de|da|do|dos|das|e))?"
_NOME_COMPLETO = rf"{_PALAVRA_NOME}(?:{_CONECTIVO}\s+{_PALAVRA_NOME}){{0,4}}"

# Ordem importa: identificadores numericos antes de datas e telefones, para
# um CPF nao ser lido como telefone; nomes por ultimo.
_REGRAS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("[CPF]", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("[CNS]", re.compile(r"\b\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\b")),
    ("[CRM]", re.compile(r"\b(?:CRM|COREN)[\s/-]*[A-Z]{2}[\s/-]*\d{4,7}\b")),
    ("[PRONTUARIO]", re.compile(
        r"(?i)\b(?:prontu[aá]rio|atendimento|registro|matr[ií]cula)\s*(?:n[ºo°.]*\s*|:\s*|#\s*)?\d{5,}\b")),
    ("[TELEFONE]", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
    ("[DATA]", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("[ENDERECO]", re.compile(
        r"\b(?:Rua|R\.|Avenida|Av\.|Travessa|Alameda|Estrada|Praça)\s+[^,;\n]{2,60},?\s*(?:n[ºo°.]*\s*)?\d{1,5}\b")),
]

_TITULO_MEDICO = re.compile(rf"\b(?:Dr|Dra|Doutor|Doutora)\.?\s+{_NOME_COMPLETO}")
_TITULO_PACIENTE = re.compile(
    rf"(?i:\b(?:paciente|sr|sra|senhor|senhora|nome|acompanhante|filha|filho|esposa|marido)\b\.?:?)\s+{_NOME_COMPLETO}")
_NOME_SOLTO = re.compile(rf"\b(?:{'|'.join(sorted(PRENOMES, key=len, reverse=True))})(?:{_CONECTIVO}\s+{_PALAVRA_NOME}){{1,4}}")


def _manter_rotulo(m: re.Match[str], marcador: str) -> str:
    """Troca so o nome, preservando a palavra que o introduziu ("Paciente [PACIENTE]")."""
    texto = m.group(0)
    partes = re.match(r"(\S+\.?:?\s+)", texto)
    return (partes.group(1) if partes else "") + marcador


def anonimizar(texto: str) -> str:
    """Substitui dados pessoais por marcadores de categoria."""
    if not texto:
        return texto
    for marcador, padrao in _REGRAS:
        texto = padrao.sub(marcador, texto)
    texto = _TITULO_MEDICO.sub(lambda m: _manter_rotulo(m, "[MEDICO]"), texto)
    texto = _TITULO_PACIENTE.sub(lambda m: _manter_rotulo(m, "[PACIENTE]"), texto)
    texto = _NOME_SOLTO.sub("[NOME]", texto)
    return texto


def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def residuos_pii(texto: str, conhecidos: list[str] | None = None) -> list[str]:
    """
    O que ainda parece dado pessoal num texto ja anonimizado.

    Duas verificacoes: padroes estruturados que nao deveriam ter sobrado e,
    quando informados, os valores exatos que se sabe estarem na origem (o
    gerador sintetico registra cada dado que inseriu). Lista vazia = limpo.
    """
    achados = []
    for marcador, padrao in _REGRAS:
        achados += [f"{marcador}:{m.group(0)}" for m in padrao.finditer(texto)]
    if conhecidos:
        alvo = _normalizar(texto)
        achados += [f"conhecido:{c}" for c in conhecidos if c and _normalizar(c) in alvo]
    return achados
