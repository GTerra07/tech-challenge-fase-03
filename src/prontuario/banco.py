"""
Base de dados estruturada do prontuario (SQLite).

Requisito 2: "realizar consultas em base de dados estruturadas (como prontuarios
e registros)". Dois tipos de acesso, separados de proposito:

- **Leitura** (`conectar_leitura`): a unica conexao que o assistente usa. Aberta
  em modo somente leitura pelo proprio SQLite (`mode=ro`) — mesmo que alguem
  consiga injetar SQL por uma ferramenta, o banco recusa escrita. E o principio
  de minimo privilegio da Aula 6 de LangChain aplicado no nivel do driver.
- **Escrita** (`registrar_resultado`): simula a integracao com o sistema do
  laboratorio que atualiza o prontuario. O assistente nao tem acesso a ela.

Os pacientes de demonstracao sao fixos (cenarios escolhidos para exercitar cada
ramo do fluxo); os nomes, CPFs e prontuarios sao falsos, gerados com semente.
"""

from __future__ import annotations

import json
import random
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

from src.config import ARQUIVO_BANCO, SEMENTE
from src.dados.gerador_interno import Pessoas

ESQUEMA = """
CREATE TABLE pacientes (
    id TEXT PRIMARY KEY, nome TEXT NOT NULL, cpf TEXT NOT NULL, prontuario TEXT NOT NULL,
    data_nascimento TEXT, telefone TEXT, endereco TEXT,
    idade INTEGER, sexo TEXT, leito TEXT, diagnostico TEXT, alergias TEXT,
    horas_desde_admissao REAL, atualizado_em TEXT
);
CREATE TABLE protocolos_ativos (paciente_id TEXT REFERENCES pacientes(id), protocolo TEXT);
CREATE TABLE sinais_vitais (paciente_id TEXT REFERENCES pacientes(id), parametro TEXT, valor REAL);
CREATE TABLE exames (
    paciente_id TEXT REFERENCES pacientes(id), exame TEXT, status TEXT,
    valor_texto TEXT, valor_num REAL, unidade TEXT, horas_desde_solicitacao REAL,
    PRIMARY KEY (paciente_id, exame)
);
"""


def _e(exame, valor=None, unidade="", status="resultado", horas=0.0):
    return {"exame": exame, "valor": valor, "unidade": unidade, "status": status, "horas": horas}


# Cenarios de demonstracao: cada um exercita um ramo do fluxo automatizado.
CENARIOS_DEMO = [
    ("P-0001", "F", 67, "21A", "sepse de foco urinário", ["PROT-001"], ["penicilina"], 2.5,
     {"pam": 61, "fc": 118, "fr": 26, "temp": 38.9},
     [_e("lactato", 4.6, "mmol/L", horas=1.5), _e("hemocultura", status="pendente", horas=2.4),
      _e("hemograma", "leucócitos 19.300/mm³", horas=2.0), _e("creatinina", 1.8, "mg/dL", horas=2.0),
      _e("gasometria arterial", status="pendente", horas=2.2)]),
    ("P-0002", "M", 58, "UCO-3", "dor torácica a esclarecer", ["PROT-002"], [], 1.2,
     {"pas": 148, "fc": 96},
     [_e("eletrocardiograma", "supradesnivelamento de ST em parede inferior", horas=1.1),
      _e("troponina", "acima do percentil 99", horas=1.0), _e("troponina de controle", status="pendente", horas=0.6)]),
    ("P-0003", "F", 34, "12C", "suspeita de cetoacidose diabética", ["PROT-004"], [], 1.5,
     {"fc": 122, "fr": 30},
     [_e("glicemia", 512, "mg/dL", horas=1.3), _e("potássio", status="pendente", horas=1.4),
      _e("gasometria venosa", "pH 7,12", horas=1.2)]),
    ("P-0004", "M", 74, "33B", "lesão renal aguda com hipercalemia", ["PROT-005"], ["contraste iodado"], 9.0,
     {"pas": 132},
     [_e("potássio", 6.8, "mEq/L", horas=4.0), _e("potássio de controle", status="pendente", horas=3.1),
      _e("creatinina", 3.4, "mg/dL", horas=8.0)]),
    ("P-0005", "F", 81, "41D", "internação clínica por insuficiência cardíaca descompensada", ["PROT-006"], [], 30.0,
     {}, [_e("hemograma", status="pendente", horas=28.0)]),
    ("P-0006", "F", 52, "AMB", "nódulo mamário em investigação", ["PROT-009"], [], 480.0,
     {}, [_e("ultrassonografia mamária", "BI-RADS 5", horas=480.0), _e("biópsia mamária", status="pendente", horas=470.0)]),
    ("P-0007", "M", 45, "18A", "pneumonia adquirida na comunidade", ["PROT-003"], [], 6.0,
     {"fr": 22, "sat": 95},
     [_e("radiografia de tórax", "consolidação em lobo inferior direito", horas=5.0),
      _e("hemograma", "leucócitos 13.100/mm³", horas=5.5), _e("ureia", 34, "mg/dL", horas=5.5)]),
]


def criar_banco(caminho: Path = ARQUIVO_BANCO, semente: int = SEMENTE) -> Path:
    """Recria o banco com os pacientes de demonstracao."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.unlink(missing_ok=True)
    pessoas = Pessoas(random.Random(semente))
    agora = datetime.now().isoformat(timespec="seconds")
    # closing(): o 'with' do sqlite3 so faz commit, nao fecha — no Windows o
    # arquivo ficaria travado e o banco nao poderia ser recriado.
    with closing(sqlite3.connect(caminho)) as con, con:
        con.executescript(ESQUEMA)
        for pid, sexo, idade, leito, diag, prots, alergias, horas_adm, sv, exames in CENARIOS_DEMO:
            con.execute("INSERT INTO pacientes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, pessoas.nome(sexo), pessoas.cpf(), pessoas.prontuario(), pessoas.data(),
                         pessoas.telefone(), pessoas.endereco(), idade, sexo, leito, diag,
                         json.dumps(alergias, ensure_ascii=False), horas_adm, agora))
            con.executemany("INSERT INTO protocolos_ativos VALUES (?,?)", [(pid, p) for p in prots])
            con.executemany("INSERT INTO sinais_vitais VALUES (?,?,?)", [(pid, k, v) for k, v in sv.items()])
            for e in exames:
                num = e["valor"] if isinstance(e["valor"], (int, float)) else None
                txt = None if num is not None or e["valor"] is None else str(e["valor"])
                con.execute("INSERT INTO exames VALUES (?,?,?,?,?,?,?)",
                            (pid, e["exame"], e["status"], txt, num, e["unidade"], e["horas"]))
    return caminho


@contextmanager
def conectar_leitura(caminho: Path = ARQUIVO_BANCO):
    """Conexao SOMENTE LEITURA — a unica que o assistente recebe."""
    if not caminho.exists():
        criar_banco(caminho)
    con = sqlite3.connect(f"file:{caminho.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def carregar_paciente(paciente_id: str, caminho: Path = ARQUIVO_BANCO) -> dict | None:
    """Registro completo do paciente, no formato usado pelas regras e pelo resumo."""
    with conectar_leitura(caminho) as con:
        p = con.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
        if p is None:
            return None
        prots = [r["protocolo"] for r in con.execute(
            "SELECT protocolo FROM protocolos_ativos WHERE paciente_id = ?", (paciente_id,))]
        sv = {r["parametro"]: r["valor"] for r in con.execute(
            "SELECT parametro, valor FROM sinais_vitais WHERE paciente_id = ?", (paciente_id,))}
        exames = []
        for r in con.execute("SELECT * FROM exames WHERE paciente_id = ? ORDER BY rowid", (paciente_id,)):
            e = {"exame": r["exame"], "status": r["status"], "horas_desde_solicitacao": r["horas_desde_solicitacao"]}
            if r["status"] == "resultado":
                e["valor"] = r["valor_num"] if r["valor_num"] is not None else r["valor_texto"]
                e["unidade"] = r["unidade"] or ""
            exames.append(e)
    return {**dict(p), "alergias": json.loads(p["alergias"] or "[]"), "protocolos_ativos": prots,
            "sinais_vitais": sv, "exames": exames}


def listar_pacientes(caminho: Path = ARQUIVO_BANCO) -> list[dict]:
    with conectar_leitura(caminho) as con:
        return [dict(r) for r in con.execute("SELECT id, leito, diagnostico FROM pacientes ORDER BY id")]


def registrar_resultado(paciente_id: str, exame: str, valor, unidade: str = "",
                        caminho: Path = ARQUIVO_BANCO) -> None:
    """
    Integracao simulada com o laboratorio: um resultado chega e atualiza o prontuario.
    Usada na demonstracao para mostrar que a resposta do assistente acompanha o
    dado atualizado. O assistente nao tem acesso a esta funcao.
    """
    num = valor if isinstance(valor, (int, float)) else None
    with closing(sqlite3.connect(caminho)) as con, con:
        con.execute("""INSERT INTO exames VALUES (?,?,'resultado',?,?,?,0)
                       ON CONFLICT(paciente_id, exame) DO UPDATE SET status='resultado',
                       valor_texto=excluded.valor_texto, valor_num=excluded.valor_num, unidade=excluded.unidade""",
                    (paciente_id, exame, None if num is not None else str(valor), num, unidade))
        con.execute("UPDATE pacientes SET atualizado_em = ? WHERE id = ?",
                    (datetime.now().isoformat(timespec="seconds"), paciente_id))


if __name__ == "__main__":
    print("banco criado em", criar_banco())
    for p in listar_pacientes():
        print(p)
