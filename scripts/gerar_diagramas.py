"""
Gera os diagramas dos fluxos LangGraph a partir dos grafos compilados e os
injeta em docs/arquitetura.md, entre os marcadores <!-- grafo:NOME --> e
<!-- /grafo:NOME -->. O diagrama do relatorio e, portanto, o proprio grafo que
roda — nao um desenho feito a mao que pode divergir do codigo.

Uso:
    python scripts/gerar_diagramas.py
"""

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from src.assistente.grafo_paciente import criar_grafo_paciente  # noqa: E402
from src.assistente.grafo_pergunta import criar_grafo_pergunta  # noqa: E402
from src.assistente.llm import Redator  # noqa: E402
from src.rag.indice import BuscaProtocolos  # noqa: E402


def mermaid(grafo) -> str:
    texto = grafo.get_graph().draw_mermaid()
    # Remove o bloco de configuracao YAML, que alguns renderizadores nao aceitam.
    return re.sub(r"^---.*?---\n", "", texto, flags=re.DOTALL).strip()


def main():
    busca = BuscaProtocolos(embeddings=DeterministicFakeEmbedding(size=8), usar_cache=False)
    redator = Redator(FakeListChatModel(responses=["x"]))
    grafos = {"paciente": criar_grafo_paciente(redator, busca), "pergunta": criar_grafo_pergunta(redator, busca)}
    pasta = RAIZ / "docs" / "diagramas"
    pasta.mkdir(parents=True, exist_ok=True)
    blocos = {}
    for nome, grafo in grafos.items():
        mmd = mermaid(grafo)
        (pasta / f"fluxo_{nome}.mmd").write_text(mmd + "\n", encoding="utf-8")
        blocos[nome] = f"<!-- grafo:{nome} -->\n```mermaid\n{mmd}\n```\n<!-- /grafo:{nome} -->"
        print(f"fluxo_{nome}.mmd atualizado")
    for doc in (RAIZ / "docs" / "arquitetura.md", RAIZ / "reports" / "relatorio-tecnico.md"):
        if not doc.exists():
            continue
        texto = doc.read_text(encoding="utf-8")
        for nome, bloco in blocos.items():
            texto = re.sub(rf"<!-- grafo:{nome} -->.*?<!-- /grafo:{nome} -->", lambda _, b=bloco: b, texto,
                           flags=re.DOTALL)
        doc.write_text(texto, encoding="utf-8")
        print("injetado em", doc.relative_to(RAIZ))


if __name__ == "__main__":
    main()
