"""
Busca nos documentos internos com fonte (requisito 3: explainability).

Base: Aula 3 de Fine-Tuning e RAG (documento + fonte no metadado) e Aula 3 de
LangGraph (FAISS, MMR, busca hibrida, confianca). Decisoes:

- **Chunk = secao do documento.** As secoes sao curtas e tematicamente fechadas;
  e o chunking semantico da aula, e cada pedaco recuperado ja tem uma fonte
  citavel exata ("PROT-001 §3") no metadado.
- **Hibrido: denso (FAISS) + lexical (BM25), fundidos por RRF.** O vetor entende
  "quando repito o lactato"; o BM25 acerta "CURB-65", "BI-RADS 4B" e codigos.
- **Embeddings locais e multilingues**: nenhum dado sai da maquina.
- **Confianca** = similaridade de cosseno do melhor trecho denso; vai para o
  estado do grafo e para o log de auditoria.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import MODELO_EMBEDDINGS, PASTA_INDICE
from src.dados.documentos import Secao, carregar_secoes

K_PADRAO = 4
RRF_K = 60  # constante usual da Reciprocal Rank Fusion


@dataclass
class Trecho:
    fonte: str
    texto: str
    score: float

    def como_contexto(self) -> str:
        return self.texto


def _documento(s: Secao) -> Document:
    return Document(page_content=s.como_contexto(),
                    metadata={"fonte": s.fonte, "codigo": s.codigo, "tipo": s.tipo, "versao": s.versao,
                              "titulo": f"{s.titulo_documento} — {s.titulo}"})


def _tokens(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texto)


class BuscaProtocolos:
    def __init__(self, embeddings=None, secoes: list[Secao] | None = None, usar_cache: bool = True):
        from langchain_community.vectorstores import FAISS
        from rank_bm25 import BM25Okapi

        self.secoes = secoes or carregar_secoes()
        self.documentos = [_documento(s) for s in self.secoes]
        self.por_fonte = {d.metadata["fonte"]: d for d in self.documentos}
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS,
                                               encode_kwargs={"normalize_embeddings": True})
        indice = PASTA_INDICE / "faiss"
        if usar_cache and (indice / "index.faiss").exists() and self._cache_valido(indice):
            self.vetorial = FAISS.load_local(str(indice), embeddings, allow_dangerous_deserialization=True)
        else:
            self.vetorial = FAISS.from_documents(self.documentos, embeddings)
            if usar_cache:
                self.vetorial.save_local(str(indice))
                (indice / "fontes.txt").write_text("\n".join(self.por_fonte), encoding="utf-8")
        self.bm25 = BM25Okapi([_tokens(d.page_content) for d in self.documentos])

    def _cache_valido(self, indice) -> bool:
        """O indice em disco so vale se tiver exatamente as mesmas secoes dos documentos atuais."""
        arq = indice / "fontes.txt"
        return arq.exists() and arq.read_text(encoding="utf-8").split("\n") == list(self.por_fonte)

    def buscar(self, consulta: str, k: int = K_PADRAO) -> tuple[list[Trecho], float]:
        """Trechos mais relevantes (fusao denso + BM25) e a confianca da busca."""
        densos = self.vetorial.similarity_search_with_score(consulta, k=12)
        # FAISS devolve distancia L2 entre vetores normalizados: cos = 1 - d²/2
        cos = {d.metadata["fonte"]: 1 - float(dist) / 2 for d, dist in densos}
        ranking_denso = [d.metadata["fonte"] for d, _ in densos]
        notas_bm25 = self.bm25.get_scores(_tokens(consulta))
        ranking_bm25 = [self.documentos[i].metadata["fonte"]
                        for i in sorted(range(len(notas_bm25)), key=lambda i: -notas_bm25[i])[:12]
                        if notas_bm25[i] > 0]
        rrf: dict[str, float] = {}
        for ranking in (ranking_denso, ranking_bm25):
            for pos, fonte in enumerate(ranking):
                rrf[fonte] = rrf.get(fonte, 0.0) + 1.0 / (RRF_K + pos + 1)
        melhores = sorted(rrf, key=lambda f: -rrf[f])[:k]
        trechos = [Trecho(f, self.por_fonte[f].page_content, round(cos.get(f, 0.0), 3)) for f in melhores]
        confianca = round(max(cos.values()), 3) if cos else 0.0
        return trechos, confianca

    def por_fontes(self, fontes) -> list[Trecho]:
        """Busca direta pela fonte (ex.: as secoes citadas pelas regras de um alerta)."""
        return [Trecho(f, self.por_fonte[f].page_content, 1.0) for f in fontes if f in self.por_fonte]


def contexto(trechos: list[Trecho]) -> str:
    return "\n\n".join(t.como_contexto() for t in trechos)
