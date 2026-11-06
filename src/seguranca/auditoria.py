"""
Log de auditoria (requisito 3: "logging detalhado para rastreamento e auditoria").

Um evento por linha em JSON Lines (`logs/auditoria.jsonl`), com o identificador
da execucao (o `thread_id` do LangGraph) para reconstruir uma consulta inteira:

- `no_inicio` / `no_fim` / `no_erro` — cada no do grafo, com tempo de execucao
  (decorator `auditar_no`, padrao da Aula 2 de LangGraph);
- `llm` — o prompt COMPLETO enviado ao modelo e a resposta, juntos (regra de ouro
  da Fase 2: sem o prompt, "o modelo nao inventou" e afirmacao sem lastro);
- `bloqueio`, `alerta`, `validacao_humana`, `verificacao_saida` — as decisoes.

O que vai para o log ja esta anonimizado: o prompt so contem o que passou por
`src/seguranca/anonimizacao.py`. Nao ha dado pessoal para vazar pelo log.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from langgraph.errors import GraphInterrupt

from src.config import ARQUIVO_AUDITORIA

_trava = threading.Lock()
logger = logging.getLogger("assistente")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class Auditoria:
    def __init__(self, arquivo: Path = ARQUIVO_AUDITORIA):
        self.arquivo = Path(arquivo)
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, evento: str, execucao: str, **dados) -> dict:
        registro = {"momento": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "evento": evento, "execucao": execucao, "id": uuid.uuid4().hex[:12], **dados}
        with _trava, open(self.arquivo, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        return registro

    def eventos(self, execucao: str | None = None) -> list[dict]:
        if not self.arquivo.exists():
            return []
        with open(self.arquivo, encoding="utf-8") as f:
            todos = [json.loads(l) for l in f if l.strip()]
        return [e for e in todos if execucao is None or e["execucao"] == execucao]


def auditar_no(auditoria_de):
    """
    Decorator de no do LangGraph: registra entrada, saida, tempo e erro.

    `auditoria_de(estado)` devolve o objeto Auditoria e o id da execucao — o
    proprio estado carrega o id, para o log de cada no ser correlacionavel.
    """
    def decorador(func):
        @wraps(func)
        def wrapper(estado: dict) -> dict:
            auditoria, execucao = auditoria_de(estado)
            nome = func.__name__
            inicio = time.perf_counter()
            auditoria.registrar("no_inicio", execucao, no=nome, chaves_entrada=sorted(estado.keys()))
            logger.info(f"[{execucao}] -> {nome}")
            try:
                saida = func(estado)
            except GraphInterrupt:
                # interrupt() do LangGraph: o no pausou para a validacao humana.
                # Nao e erro; o no roda de novo do inicio quando o medico responder.
                auditoria.registrar("no_pausa", execucao, no=nome, motivo="aguardando validação humana")
                logger.info(f"[{execucao}] || {nome} aguardando validação humana")
                raise
            except Exception as e:
                ms = round((time.perf_counter() - inicio) * 1000, 1)
                auditoria.registrar("no_erro", execucao, no=nome, erro=f"{type(e).__name__}: {e}", ms=ms)
                logger.error(f"[{execucao}] x {nome}: {e}")
                raise
            ms = round((time.perf_counter() - inicio) * 1000, 1)
            auditoria.registrar("no_fim", execucao, no=nome, ms=ms, chaves_saida=sorted((saida or {}).keys()))
            logger.info(f"[{execucao}] <- {nome} ({ms} ms)")
            return saida
        return wrapper
    return decorador
