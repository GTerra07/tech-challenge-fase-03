"""
Leitura dos documentos internos do hospital: protocolos, procedimentos (POPs)
e modelos de documento.

Cada documento e um Markdown com frontmatter e secoes numeradas (`## 3. ...`).
A secao e a menor unidade citavel: e o que vira a fonte "[PROT-001 §3]" nas
respostas do assistente. Por isso o mesmo parser alimenta duas coisas que
precisam concordar entre si — o dataset de fine-tuning e o indice de busca.
Se cada um lesse os documentos do seu jeito, a citacao aprendida no treino
poderia nao existir no indice (regra duplicada sempre diverge).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import PASTAS_DOCUMENTOS

TIPOS = {"PROT": "protocolo", "POP": "procedimento", "MOD": "modelo"}

_RE_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_RE_SECAO = re.compile(r"^## (\d+)\. (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Secao:
    codigo: str          # PROT-001
    titulo_documento: str
    versao: str
    numero: int          # 3
    titulo: str          # Pacote da primeira hora
    texto: str

    @property
    def fonte(self) -> str:
        """Identificador citavel da secao, ex.: 'PROT-001 §3'."""
        return f"{self.codigo} §{self.numero}"

    @property
    def tipo(self) -> str:
        return TIPOS[self.codigo.split("-")[0]]

    def como_contexto(self) -> str:
        """Formato em que a secao entra no prompt, no treino e na inferencia."""
        return f"[{self.fonte}] {self.titulo_documento} — {self.titulo}\n{self.texto}"


def _frontmatter(texto: str) -> dict[str, str]:
    m = _RE_FRONTMATTER.match(texto)
    if not m:
        raise ValueError("documento sem frontmatter")
    campos = {}
    for linha in m.group(1).splitlines():
        chave, _, valor = linha.partition(":")
        campos[chave.strip()] = valor.strip().strip('"')
    return campos


def ler_documento(caminho: Path) -> list[Secao]:
    texto = caminho.read_text(encoding="utf-8").replace("\r\n", "\n")
    meta = _frontmatter(texto)
    marcas = list(_RE_SECAO.finditer(texto))
    secoes = []
    for i, m in enumerate(marcas):
        fim = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        corpo = " ".join(texto[m.end():fim].split())
        secoes.append(
            Secao(
                codigo=meta["codigo"],
                titulo_documento=meta["titulo"],
                versao=meta["versao"],
                numero=int(m.group(1)),
                titulo=m.group(2).strip(),
                texto=corpo,
            )
        )
    return secoes


def carregar_secoes(pastas=PASTAS_DOCUMENTOS) -> list[Secao]:
    """Todas as secoes de todos os documentos internos, em ordem estavel."""
    secoes = []
    for pasta in pastas:
        for caminho in sorted(Path(pasta).glob("*.md")):
            secoes.extend(ler_documento(caminho))
    return secoes


def indice_por_fonte(secoes: list[Secao] | None = None) -> dict[str, Secao]:
    secoes = secoes if secoes is not None else carregar_secoes()
    return {s.fonte: s for s in secoes}
