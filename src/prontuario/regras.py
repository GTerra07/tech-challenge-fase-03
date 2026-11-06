"""
Regras clinicas deterministicas: exames pendentes e valores criticos.

Estas regras sao a traducao direta dos prazos e limites escritos nos
protocolos (cada uma carrega a secao que a define). Ficam em codigo, e nao na
LLM, por tres motivos:

1. **Alerta nao pode depender de geracao de texto.** Um valor critico tem que
   disparar sempre, com a mesma resposta, e isso se prova com teste.
2. **A LLM narra, nunca inventa** (regra de ouro herdada da Fase 2): ela recebe
   as pendencias e os alertas ja calculados e escreve a sugestao em cima deles.
3. **O mesmo motor gera os casos de treino** do fine-tuning, entao o que o modelo
   aprende a narrar e exatamente o que o assistente vai calcular em producao.
"""

from __future__ import annotations

from dataclasses import dataclass

CRITICO, ALTO, INFORMATIVO = "CRITICO", "ALTO", "INFORMATIVO"
_ORDEM_NIVEL = {CRITICO: 0, ALTO: 1, INFORMATIVO: 2}


@dataclass(frozen=True)
class Exigencia:
    """Um exame que um protocolo exige, com prazo e gravidade do atraso."""

    exame: str
    prazo_horas: float
    nivel_atraso: str
    fonte: str
    motivo: str


# Exigencias por protocolo — cada linha cita a secao que a define.
EXIGENCIAS: dict[str, list[Exigencia]] = {
    "PROT-001": [
        Exigencia("hemocultura", 1, CRITICO, "PROT-001 §4",
                  "hemoculturas devem ser coletadas na primeira hora, antes do antimicrobiano"),
        Exigencia("lactato", 1, CRITICO, "PROT-001 §3", "lactato faz parte do pacote da primeira hora"),
        Exigencia("hemograma", 6, ALTO, "PROT-001 §4", "exame obrigatório do protocolo de sepse"),
        Exigencia("creatinina", 6, ALTO, "PROT-001 §4", "exame obrigatório do protocolo de sepse"),
        Exigencia("gasometria arterial", 6, ALTO, "PROT-001 §4", "exame obrigatório do protocolo de sepse"),
    ],
    "PROT-002": [
        Exigencia("eletrocardiograma", 1 / 6, CRITICO, "PROT-002 §2",
                  "ECG de 12 derivações em até 10 minutos da chegada"),
        Exigencia("troponina", 1, ALTO, "PROT-002 §3", "troponina na admissão e de controle conforme o algoritmo"),
        Exigencia("troponina de controle", 3, ALTO, "PROT-002 §3", "troponina de controle no horário previsto"),
    ],
    "PROT-003": [
        Exigencia("radiografia de tórax", 4, ALTO, "PROT-003 §2", "radiografia de tórax obrigatória na suspeita de PAC"),
        Exigencia("hemograma", 6, ALTO, "PROT-003 §4", "exame da internação por PAC"),
        Exigencia("ureia", 6, ALTO, "PROT-003 §4", "necessária para o escore CURB-65"),
    ],
    "PROT-004": [
        Exigencia("potássio", 1, CRITICO, "PROT-004 §4",
                  "o potássio deve ser conhecido antes da insulina endovenosa na CAD"),
        Exigencia("gasometria venosa", 1, ALTO, "PROT-004 §3", "coleta urgente na suspeita de CAD"),
    ],
    "PROT-005": [
        Exigencia("potássio de controle", 2, CRITICO, "PROT-005 §5",
                  "potássio de controle em até 2 horas após tratar a hipercalemia"),
        Exigencia("creatinina", 24, ALTO, "PROT-005 §5", "creatinina a cada 24 horas enquanto houver LRA"),
    ],
    "PROT-006": [
        Exigencia("hemograma", 24, ALTO, "PROT-006 §3",
                  "plaquetas pendentes impedem liberar a profilaxia farmacológica"),
    ],
    "PROT-009": [
        Exigencia("biópsia mamária", 15 * 24, CRITICO, "PROT-009 §4",
                  "biópsia em até 15 dias para BI-RADS 4 ou 5"),
    ],
}

# Valores criticos do PROT-007 §2: (exame, comparador, limite, unidade)
VALORES_CRITICOS = [
    ("potássio", "<", 2.8, "mEq/L"), ("potássio", ">", 6.0, "mEq/L"),
    ("potássio de controle", "<", 2.8, "mEq/L"), ("potássio de controle", ">", 6.0, "mEq/L"),
    ("sódio", "<", 120, "mEq/L"), ("sódio", ">", 160, "mEq/L"),
    ("glicemia", "<", 54, "mg/dL"), ("glicemia", ">", 400, "mg/dL"),
    ("lactato", ">=", 4.0, "mmol/L"),
    ("hemoglobina", "<", 7.0, "g/dL"),
    ("plaquetas", "<", 20000, "/mm3"),
    ("ph", "<", 7.2, ""), ("ph", ">", 7.6, ""),
]
# Resultados qualitativos criticos (PROT-007 §2, PROT-002 §6)
RESULTADOS_CRITICOS = {
    "hemocultura": ("positiva", "PROT-007 §2"),
    "troponina": ("acima do percentil 99", "PROT-007 §2"),
    "troponina de controle": ("acima do percentil 99", "PROT-007 §2"),
    "eletrocardiograma": ("supradesnivelamento de ST", "PROT-002 §6"),
}


@dataclass(frozen=True)
class Achado:
    tipo: str     # "pendencia" | "valor_critico"
    nivel: str
    exame: str
    descricao: str
    fonte: str

    def linha(self) -> str:
        return f"[{self.nivel}] {self.descricao} (Fonte: {self.fonte})"


def _compara(valor: float, op: str, limite: float) -> bool:
    return {"<": valor < limite, ">": valor > limite, ">=": valor >= limite}[op]


def exames_pendentes(paciente: dict) -> list[Achado]:
    """Exames exigidos pelos protocolos ativos que nao tem resultado."""
    por_nome = {e["exame"]: e for e in paciente.get("exames", [])}
    achados = []
    for protocolo in paciente.get("protocolos_ativos", []):
        for ex in EXIGENCIAS.get(protocolo, []):
            registro = por_nome.get(ex.exame)
            if registro and registro.get("status") == "resultado":
                continue
            horas = (registro or {}).get("horas_desde_solicitacao",
                                           paciente.get("horas_desde_admissao", 0))
            situacao = "solicitado e sem resultado" if registro else "não solicitado"
            atrasado = horas > ex.prazo_horas
            nivel = ex.nivel_atraso if atrasado else INFORMATIVO
            prazo = _formata_prazo(ex.prazo_horas)
            descricao = (f"{ex.exame}: {situacao} há {_formata_prazo(horas)}"
                         f" (prazo do protocolo: {prazo}{', ATRASADO' if atrasado else ''}) — {ex.motivo}")
            achados.append(Achado("pendencia", nivel, ex.exame, descricao, ex.fonte))
    return _ordenar(achados)


def valores_criticos(paciente: dict) -> list[Achado]:
    """Resultados que atingem a lista de valores criticos do PROT-007."""
    achados = []
    for e in paciente.get("exames", []):
        if e.get("status") != "resultado":
            continue
        nome, valor = e["exame"], e.get("valor")
        if isinstance(valor, (int, float)):
            for exame, op, limite, unidade in VALORES_CRITICOS:
                if exame == nome and _compara(valor, op, limite):
                    achados.append(Achado(
                        "valor_critico", CRITICO, nome,
                        f"{nome} = {_num(valor)} {unidade}".strip() + f" (limite crítico {op} {_num(limite)})",
                        "PROT-007 §2"))
        elif nome in RESULTADOS_CRITICOS:
            alvo, fonte = RESULTADOS_CRITICOS[nome]
            if alvo.lower() in str(valor).lower():
                achados.append(Achado("valor_critico", CRITICO, nome, f"{nome}: {valor}", fonte))
    return _ordenar(achados)


def alertas(paciente: dict) -> list[Achado]:
    """O que deve virar alerta para a equipe: tudo que e CRITICO ou ALTO."""
    todos = valores_criticos(paciente) + exames_pendentes(paciente)
    return _ordenar([a for a in todos if a.nivel in (CRITICO, ALTO)])


# Acao que o protocolo preve para cada achado, com a secao que a define.
# E o alvo dos casos de treino: o modelo aprende a sugerir isto, sempre como
# rascunho e sem medicamento nem dose (PROT-008 §3).
ACOES: dict[tuple[str, str], str] = {
    ("pendencia", "hemocultura"): "Coletar as duas hemoculturas antes do antimicrobiano (POP-001 §2), sem atrasar o antimicrobiano mais de 45 minutos (PROT-001 §3).",
    ("pendencia", "lactato"): "Coletar o lactato, item do pacote da primeira hora (PROT-001 §3).",
    ("pendencia", "hemograma"): "Liberar a coleta do hemograma; na profilaxia de TEV, plaquetas pendentes impedem a profilaxia farmacológica (PROT-006 §3).",
    ("pendencia", "creatinina"): "Coletar creatinina conforme o protocolo ativo (PROT-001 §4 / PROT-005 §5).",
    ("pendencia", "gasometria arterial"): "Coletar gasometria arterial, exame obrigatório do protocolo de sepse (PROT-001 §4).",
    ("pendencia", "eletrocardiograma"): "Realizar ECG de 12 derivações imediatamente, com interpretação médica (PROT-002 §2).",
    ("pendencia", "troponina"): "Coletar troponina de alta sensibilidade conforme o algoritmo institucional (PROT-002 §3).",
    ("pendencia", "troponina de controle"): "Coletar a troponina de controle e interpretar a variação entre as dosagens (PROT-002 §3).",
    ("pendencia", "radiografia de tórax"): "Realizar a radiografia de tórax, obrigatória na suspeita de PAC (PROT-003 §2).",
    ("pendencia", "ureia"): "Coletar ureia para completar o CURB-65 (PROT-003 §3).",
    ("pendencia", "potássio"): "Não iniciar insulina endovenosa antes do resultado do potássio (PROT-004 §4).",
    ("pendencia", "gasometria venosa"): "Coletar gasometria com urgência na suspeita de CAD (PROT-004 §3).",
    ("pendencia", "potássio de controle"): "Coletar o potássio de controle; o prazo é de 2 horas após o tratamento da hipercalemia (PROT-005 §5).",
    ("pendencia", "biópsia mamária"): "Agendar a biópsia percutânea guiada por imagem; o prazo para BI-RADS 4 ou 5 é de 15 dias (PROT-009 §4).",
    ("valor_critico", "lactato"): "Lactato ≥ 4 mmol/L: o protocolo prevê reposição volêmica com cristaloide conforme prescrição médica (PROT-001 §3) e reavaliação em até 6 horas (PROT-001 §5).",
    ("valor_critico", "potássio"): "Potássio em faixa crítica: realizar ECG imediatamente e repetir a dosagem para afastar hemólise; medidas terapêuticas conforme prescrição médica (PROT-005 §4).",
    ("valor_critico", "potássio de controle"): "Potássio de controle em faixa crítica: reavaliação médica imediata (PROT-005 §4).",
    ("valor_critico", "glicemia"): "Glicemia em faixa crítica (< 54 ou > 400 mg/dL): reavaliação médica imediata; na hiperglicemia, investigar cetoacidose (PROT-004 §3); na hipoglicemia, seguir o POP-004 (PROT-004 §5).",
    ("valor_critico", "sódio"): "Sódio em faixa crítica: avaliação médica imediata (PROT-007 §2).",
    ("valor_critico", "hemoglobina"): "Hemoglobina < 7 g/dL: avaliação médica imediata (PROT-007 §2).",
    ("valor_critico", "plaquetas"): "Plaquetas < 20.000/mm³: avaliação médica imediata; reavaliar anticoagulantes em uso (PROT-006 §6).",
    ("valor_critico", "ph"): "pH em faixa crítica: avaliação médica imediata (PROT-007 §2).",
    ("valor_critico", "hemocultura"): "Hemocultura positiva: comunicar o médico para reavaliar o antimicrobiano conforme a CCIH (PROT-007 §2).",
    ("valor_critico", "troponina"): "Troponina acima do percentil 99: avaliar com o ECG e a troponina de controle; acionar a cardiologia (PROT-002 §5 e §6).",
    ("valor_critico", "troponina de controle"): "Troponina de controle acima do percentil 99: acionar a cardiologia (PROT-002 §6).",
    ("valor_critico", "eletrocardiograma"): "Supradesnivelamento de ST: acionar imediatamente a hemodinâmica, meta porta-balão < 90 min (PROT-002 §4).",
}


def acao_sugerida(achado: Achado) -> str:
    return ACOES.get((achado.tipo, achado.exame),
                     f"Avaliar {achado.exame} conforme o protocolo ({achado.fonte}).")


# Secoes sempre presentes no contexto de um caso: o limite de atuacao do
# assistente e, havendo alerta critico, o prazo e o registro da comunicacao.
FONTES_FIXAS = ("PROT-008 §3",)
FONTES_SE_CRITICO = ("PROT-007 §3", "PROT-007 §4")


def fontes_para_contexto(paciente: dict) -> list[str]:
    """
    Toda secao que a sugestao de referencia pode citar para este paciente.

    Definida aqui, uma vez, e usada pelo dataset de treino e pelo grafo: a
    sugestao so pode citar o que estava no contexto (explainability verificavel),
    entao o contexto tem que conter as secoes das acoes, e nao so as dos alertas.
    """
    from src.avaliacao.metricas import fontes_citadas

    alertas_ = alertas(paciente)
    fontes = {a.fonte for a in alertas_} | set(FONTES_FIXAS)
    # O objetivo (§1) de cada protocolo ativo situa o caso
    fontes |= {f"{prot} §1" for prot in paciente.get("protocolos_ativos", [])}
    for a in alertas_:
        fontes |= fontes_citadas(acao_sugerida(a))
    if any(a.nivel == CRITICO for a in alertas_):
        fontes |= set(FONTES_SE_CRITICO)
    return sorted(fontes)


def fontes_da_pergunta_paciente(paciente: dict) -> list[str]:
    """
    Secoes que entram no contexto de uma pergunta sobre um paciente: as que os
    achados dele citam, mais a lista e o prazo de valores criticos (PROT-007 §2
    e §3). Usada pelo grafo de perguntas e pelo dataset (v2).
    """
    achados = valores_criticos(paciente) + exames_pendentes(paciente)
    return sorted({a.fonte for a in achados} | {"PROT-007 §2", "PROT-007 §3"})


def _ordenar(achados: list[Achado]) -> list[Achado]:
    return sorted(achados, key=lambda a: (_ORDEM_NIVEL[a.nivel], a.exame))


def _num(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def _formata_prazo(horas: float) -> str:
    if horas < 1:
        return f"{round(horas * 60)} min"
    if horas >= 48:
        return f"{horas / 24:g} dias".replace(".", ",")
    return f"{horas:g} h".replace(".", ",")
