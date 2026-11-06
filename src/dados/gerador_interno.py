"""
Gerador das exportacoes brutas sinteticas do hospital.

O enunciado pede fine-tuning com "dados medicos internos" preparados com
anonimizacao. Como o hospital e ficticio, geramos aqui o que seria a exportacao
dos sistemas internos: mensagens de medicos no canal de duvidas, laudos, receitas
e casos de pacientes — COM dados pessoais falsos (nome, CPF, telefone,
prontuario, CRM, endereco, datas), exatamente para a etapa de anonimizacao ter
o que remover e para podermos provar que removeu.

Cada registro guarda em `pii` a lista dos valores pessoais que inseriu. A
curadoria e os testes usam essa lista como gabarito: nenhum desses valores pode
sobreviver no dataset final.

Tudo com semente fixa: rodar de novo produz os mesmos arquivos.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import PASTA_BRUTO, SEMENTE
from src.dados import conhecimento_interno as ci

SOBRENOMES = ["Silva", "Souza", "Oliveira", "Santos", "Pereira", "Lima", "Carvalho", "Ferreira", "Rodrigues",
              "Almeida", "Costa", "Gomes", "Martins", "Araujo", "Ribeiro", "Barbosa", "Rocha", "Mendes",
              "Teixeira", "Moreira", "Cardoso", "Nogueira", "Freitas", "Vieira"]
PRENOMES_F = ["Ana", "Maria", "Mariana", "Juliana", "Fernanda", "Patricia", "Aline", "Camila", "Beatriz",
              "Larissa", "Gabriela", "Renata", "Carla", "Luciana", "Helena", "Sandra", "Tereza", "Lucia"]
PRENOMES_M = ["Joao", "Jose", "Carlos", "Paulo", "Pedro", "Lucas", "Marcos", "Rafael", "Bruno", "Felipe",
              "Gustavo", "Ricardo", "Rodrigo", "Eduardo", "Antonio", "Francisco", "Luiz", "Sergio", "Marcelo"]
RUAS = ["Rua das Acácias", "Avenida Paulista", "Rua Vergueiro", "Rua Augusta", "Avenida Rebouças", "Rua Domingos de Morais"]
UFS = ["SP", "RJ", "MG", "PR"]


class Pessoas:
    """Fabrica de identidades falsas, todas registradas para o gabarito."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def nome(self, sexo: str | None = None) -> str:
        sexo = sexo or self.rng.choice("FM")
        prenome = self.rng.choice(PRENOMES_F if sexo == "F" else PRENOMES_M)
        return f"{prenome} {self.rng.choice(SOBRENOMES)} {self.rng.choice(SOBRENOMES)}"

    def cpf(self) -> str:
        base = [self.rng.randint(0, 9) for _ in range(9)]
        for peso in (10, 11):
            soma = sum(d * (peso - i) for i, d in enumerate(base))
            base.append(0 if soma % 11 < 2 else 11 - soma % 11)
        d = "".join(map(str, base))
        return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"

    def telefone(self) -> str:
        return f"(11) 9{self.rng.randint(1000, 9999)}-{self.rng.randint(1000, 9999)}"

    def crm(self) -> str:
        return f"CRM-{self.rng.choice(UFS)} {self.rng.randint(10000, 999999)}"

    def prontuario(self) -> str:
        return f"{self.rng.randint(1000000, 9999999)}"

    def data(self, ano_min=1940, ano_max=2005) -> str:
        return f"{self.rng.randint(1, 28):02d}/{self.rng.randint(1, 12):02d}/{self.rng.randint(ano_min, ano_max)}"

    def endereco(self) -> str:
        return f"{self.rng.choice(RUAS)}, {self.rng.randint(10, 2999)}"


# ---------------------------------------------------------------------------
# Casos clinicos (mesma estrutura dos registros do prontuario)
# ---------------------------------------------------------------------------

def _exame(nome, valor=None, unidade="", status="resultado", horas=0.0):
    e = {"exame": nome, "status": status, "horas_desde_solicitacao": round(horas, 2)}
    if status == "resultado":
        e["valor"], e["unidade"] = valor, unidade
    return e


def _talvez_pendente(rng, nome, valor, unidade, prob_pendente, horas_max):
    if rng.random() < prob_pendente:
        return _exame(nome, status="pendente", horas=rng.uniform(0.2, horas_max))
    return _exame(nome, valor, unidade, horas=rng.uniform(0.5, horas_max))


def cenario_sepse(rng):
    lact = rng.choice([1.6, 2.8, 3.4, 4.2, 4.9, 6.1])
    exames = [_talvez_pendente(rng, "lactato", lact, "mmol/L", 0.25, 3),
              _talvez_pendente(rng, "hemocultura", "sem crescimento parcial", "", 0.6, 4),
              _talvez_pendente(rng, "hemograma", "leucócitos 18.400/mm³", "", 0.3, 8),
              _talvez_pendente(rng, "creatinina", rng.choice([0.9, 1.6, 2.3]), "mg/dL", 0.3, 8)]
    if rng.random() < 0.5:
        exames.append(_talvez_pendente(rng, "gasometria arterial", "pH 7,31", "", 0.5, 8))
    return {"diagnostico": rng.choice(["sepse de foco urinário", "sepse de foco pulmonar", "sepse de foco abdominal"]),
            "protocolos_ativos": ["PROT-001"], "exames": exames,
            "sinais_vitais": {"pam": rng.choice([58, 63, 72, 80]), "fc": rng.randint(95, 130), "fr": rng.randint(18, 30),
                              "temp": rng.choice([38.4, 38.9, 39.2])}}


def cenario_dor_toracica(rng):
    ecg = rng.choice(["ritmo sinusal sem alterações", "supradesnivelamento de ST em parede inferior",
                      "inversão de onda T em V4-V6"])
    exames = [_talvez_pendente(rng, "eletrocardiograma", ecg, "", 0.2, 0.5),
              _talvez_pendente(rng, "troponina", rng.choice(["normal", "acima do percentil 99"]), "", 0.3, 2),
              _exame("troponina de controle", status="pendente", horas=rng.uniform(0.5, 5))]
    return {"diagnostico": "dor torácica a esclarecer", "protocolos_ativos": ["PROT-002"], "exames": exames,
            "sinais_vitais": {"pas": rng.randint(95, 170), "fc": rng.randint(60, 115)}}


def cenario_pneumonia(rng):
    exames = [_talvez_pendente(rng, "radiografia de tórax", "consolidação em lobo inferior direito", "", 0.4, 7),
              _talvez_pendente(rng, "hemograma", "leucócitos 14.200/mm³", "", 0.3, 8),
              _talvez_pendente(rng, "ureia", rng.choice([32, 58, 74]), "mg/dL", 0.4, 8)]
    return {"diagnostico": "pneumonia adquirida na comunidade", "protocolos_ativos": ["PROT-003"], "exames": exames,
            "sinais_vitais": {"fr": rng.randint(20, 34), "sat": rng.choice([88, 91, 94, 96])}}


def cenario_cad(rng):
    exames = [_exame("glicemia", rng.choice([310, 420, 512]), "mg/dL", horas=0.5),
              _talvez_pendente(rng, "potássio", rng.choice([3.1, 4.2, 5.3]), "mEq/L", 0.5, 3),
              _talvez_pendente(rng, "gasometria venosa", "pH 7,12", "", 0.4, 3)]
    if rng.random() < 0.5:
        exames.append(_exame("ph", rng.choice([7.08, 7.18, 7.26]), "", horas=1))
    return {"diagnostico": "suspeita de cetoacidose diabética", "protocolos_ativos": ["PROT-004"], "exames": exames,
            "sinais_vitais": {"fc": rng.randint(100, 128), "fr": rng.randint(22, 32)}}


def cenario_renal(rng):
    exames = [_exame("potássio", rng.choice([5.4, 6.3, 6.8]), "mEq/L", horas=3),
              _talvez_pendente(rng, "potássio de controle", rng.choice([5.1, 5.8, 6.2]), "mEq/L", 0.6, 4),
              _talvez_pendente(rng, "creatinina", rng.choice([1.9, 2.7, 3.4]), "mg/dL", 0.3, 30)]
    return {"diagnostico": "lesão renal aguda com hipercalemia", "protocolos_ativos": ["PROT-005"], "exames": exames,
            "sinais_vitais": {"pas": rng.randint(100, 150)}}


def cenario_tev(rng):
    exames = [_talvez_pendente(rng, "hemograma", "plaquetas 182.000/mm³", "", 0.5, 30)]
    if rng.random() < 0.4:
        exames.append(_exame("plaquetas", rng.choice([18000, 45000, 150000]), "/mm3", horas=10))
    return {"diagnostico": "internação clínica por insuficiência cardíaca descompensada",
            "protocolos_ativos": ["PROT-006"], "exames": exames, "sinais_vitais": {}}


def cenario_mama(rng):
    exames = [_exame("ultrassonografia mamária", rng.choice(["BI-RADS 4B", "BI-RADS 5", "BI-RADS 4C"]), "", horas=100),
              _exame("biópsia mamária", status="pendente", horas=rng.choice([72, 200, 400, 520]))]
    return {"diagnostico": "nódulo mamário em investigação", "protocolos_ativos": ["PROT-009"], "exames": exames,
            "sinais_vitais": {}, "sexo": "F"}


CENARIOS = [cenario_sepse, cenario_dor_toracica, cenario_pneumonia, cenario_cad, cenario_renal, cenario_tev,
            cenario_mama]


def gerar_paciente(rng: random.Random, pessoas: Pessoas, indice: int, cenario=None) -> dict:
    base = (cenario or rng.choice(CENARIOS))(rng)
    sexo = base.pop("sexo", rng.choice("FM"))
    p = {
        "id": f"C-{indice:04d}", "nome": pessoas.nome(sexo), "cpf": pessoas.cpf(),
        "prontuario": pessoas.prontuario(), "data_nascimento": pessoas.data(),
        "telefone": pessoas.telefone(), "endereco": pessoas.endereco(),
        "idade": rng.randint(28, 88), "sexo": sexo,
        "leito": f"{rng.randint(1, 9)}{rng.randint(0, 3)}{rng.choice('ABCD')}",
        "alergias": rng.choice([[], [], ["dipirona"], ["penicilina"], ["contraste iodado"]]),
        "horas_desde_admissao": round(rng.uniform(1, 30), 1),
        **base,
    }
    return p


def pii_do_paciente(p: dict) -> list[str]:
    return [p["nome"], p["cpf"], p["prontuario"], p["data_nascimento"], p["telefone"], p["endereco"]]


# ---------------------------------------------------------------------------
# Exportacoes brutas
# ---------------------------------------------------------------------------

def _mensagens_faq(rng, pessoas):
    """Canal de duvidas: a pergunta do medico chega com identificacao e, as vezes, dado de paciente."""
    registros = []
    for i, item in enumerate(ci.FAQ):
        for j, pergunta in enumerate(item.perguntas):
            medico, crm = pessoas.nome(), pessoas.crm()
            pii = [medico, crm]
            cabecalho = f"Dr. {medico} ({crm}) escreveu: "
            if rng.random() < 0.5:
                paciente, pront, tel = pessoas.nome(), pessoas.prontuario(), pessoas.telefone()
                pii += [paciente, pront, tel]
                cabecalho += (f"estou com a paciente {paciente}, prontuário {pront}, familiar no telefone {tel}. ")
            registros.append({"id": f"faq-{i:03d}-{j}", "tipo": "faq", "fonte": item.fonte,
                              "texto_bruto": cabecalho + pergunta, "pergunta": pergunta,
                              "resposta": item.resposta, "pii": pii})
    return registros


def _laudos(rng, pessoas, n_rx=45, n_us=45, n_ecg=30):
    registros = []

    def cabecalho():
        nome, pront, nasc, data_exame = pessoas.nome(), pessoas.prontuario(), pessoas.data(), pessoas.data(2026, 2026)
        return (f"Paciente: {nome} | Atendimento nº {pront} | Nascimento: {nasc} | Data do exame: {data_exame}",
                [nome, pront, nasc, data_exame])

    medico = lambda: (f"Dr. {pessoas.nome()}", pessoas.crm())  # noqa: E731

    for i in range(n_rx):
        cab, pii = cabecalho()
        par, imp_par = rng.choice(ci.RX_PARENQUIMA)
        seio, imp_seio = rng.choice(ci.RX_SEIOS)
        cor, imp_cor = rng.choice(ci.RX_CORACAO)
        med, crm = medico()
        achados = f"{par}; {seio}; {cor}; mediastino centrado; estruturas ósseas sem alterações"
        impressao = [imp_par] + [x for x in (imp_seio, imp_cor) if x]
        registros.append({
            "id": f"laudo-rx-{i:03d}", "tipo": "laudo", "fonte": "MOD-001 §2",
            "pedido": "Radiografia de tórax em PA e perfil", "achados": achados,
            "texto_bruto": f"{cab}\nSolicitante: {med} ({crm})\nExame: Radiografia de tórax em PA e perfil\nAchados ditados: {achados}",
            "laudo": _monta_laudo(cab, "Radiografia de tórax em PA e perfil", achados, impressao, "MOD-001 §2"),
            "pii": pii + [med.removeprefix("Dr. "), crm]})
    for i in range(n_us):
        cab, pii = cabecalho()
        forma, margem, orient, birads, conduta = rng.choice(ci.US_NODULOS)
        mama, quad, mm = rng.choice(ci.MAMAS), rng.choice(ci.QUADRANTES), rng.randint(6, 28)
        achados = (f"nódulo na mama {mama}, {quad}, forma {forma}, margens {margem}, orientação {orient}, "
                   f"maior diâmetro {mm} mm")
        impressao = [f"BI-RADS {birads}.", f"Recomendação: {conduta} (PROT-009 §3)."]
        registros.append({
            "id": f"laudo-us-{i:03d}", "tipo": "laudo", "fonte": "MOD-001 §3",
            "pedido": "Ultrassonografia das mamas", "achados": achados,
            "texto_bruto": f"{cab}\nExame: Ultrassonografia das mamas\nAchados ditados: {achados}",
            "laudo": _monta_laudo(cab, "Ultrassonografia das mamas", achados, impressao, "MOD-001 §3"),
            "pii": pii})
    for i in range(n_ecg):
        cab, pii = cabecalho()
        ritmo, fc, st, imp, _ = rng.choice(ci.ECG_CASOS)
        fc = fc + rng.randint(-4, 4)
        achados = f"{ritmo}, frequência cardíaca {fc} bpm, eixo normal, intervalos normais, {st}"
        registros.append({
            "id": f"laudo-ecg-{i:03d}", "tipo": "laudo", "fonte": "MOD-001 §4",
            "pedido": "Eletrocardiograma de repouso de 12 derivações", "achados": achados,
            "texto_bruto": f"{cab}\nExame: Eletrocardiograma de repouso\nAchados ditados: {achados}",
            "laudo": _monta_laudo(cab, "Eletrocardiograma de repouso de 12 derivações", achados, [imp], "MOD-001 §4"),
            "pii": pii})
    return registros


def _monta_laudo(cabecalho, exame, achados, impressao, fonte):
    topicos = "\n".join(f"- {t}" for t in impressao)
    return (f"RASCUNHO DE LAUDO — requer revisão e assinatura do médico responsável ({fonte}; MOD-001 §1)\n"
            f"IDENTIFICAÇÃO: {cabecalho}\nEXAME: {exame}\nACHADOS: {achados[0].upper() + achados[1:]}.\n"
            f"IMPRESSÃO DIAGNÓSTICA:\n{topicos}\nRESPONSÁVEL: [MEDICO] — registro no conselho e assinatura pendentes.")


def _receitas(rng, pessoas, n=30):
    registros = []
    for i in range(n):
        medico, crm = pessoas.nome(), pessoas.crm()
        paciente, pront = pessoas.nome(), pessoas.prontuario()
        itens = rng.sample(ci.ITENS_RECEITA, rng.randint(1, 3))
        prescricao = "; ".join(f"{m} {c} {f}, {q}, {v}, {pos}, por {d}" for m, c, f, q, v, pos, d in itens)
        corpo = "\n".join(f"{k}. {m.capitalize()} {c} — {f} — quantidade: {q}\n   {v}: {pos}, por {d}."
                          for k, (m, c, f, q, v, pos, d) in enumerate(itens, 1))
        rascunho = ("RASCUNHO PARA CONFERÊNCIA — NÃO É RECEITA VÁLIDA (MOD-002 §4)\n"
                    f"Paciente: {paciente} | Prontuário nº {pront}\nUSO CONFORME PRESCRIÇÃO MÉDICA:\n{corpo}\n"
                    f"Prescrição definida por Dr. {medico} ({crm}). Requer revisão, assinatura e carimbo do médico responsável.")
        registros.append({"id": f"receita-{i:03d}", "tipo": "receita", "fonte": "MOD-002 §4",
                          "texto_bruto": f"Dr. {medico} ({crm}) definiu para o paciente {paciente}, prontuário {pront}: {prescricao}",
                          "receita": rascunho, "pii": [medico, crm, paciente, pront]})
    return registros


POR_TEMPLATE_RECUSA = 15
# Aberturas variadas: o mesmo pedido chega com tons diferentes, e o modelo deve
# recusar todos — inclusive os que alegam urgencia ou autoridade.
ABERTURAS = ["", "Urgente: ", "Rapidinho, ", "Plantão cheio aqui. ", "Por favor, ", "Preciso disso agora: "]


def _pedidos_indevidos(rng, pessoas):
    registros = []
    k = 0
    for modelo in ci.PEDIDOS_PRESCRICAO:
        for _ in range(POR_TEMPLATE_RECUSA):
            nome = pessoas.nome()
            texto = rng.choice(ABERTURAS) + modelo.format(
                med=rng.choice(ci.MEDICAMENTOS), leito=f"{rng.randint(1, 9)}{rng.randint(0, 3)}{rng.choice('ABCD')}",
                pid=f"paciente {nome}")
            registros.append({"id": f"recusa-{k:03d}", "tipo": "recusa", "fonte": "PROT-008 §2",
                              "texto_bruto": texto, "resposta": ci.RESPOSTA_RECUSA, "pii": [nome]})
            k += 1
    return registros


def _casos(rng, pessoas, n=280):
    registros = []
    for i in range(n):
        p = gerar_paciente(rng, pessoas, i, CENARIOS[i % len(CENARIOS)])
        registros.append({"id": f"caso-{i:03d}", "tipo": "caso", "fonte": ",".join(p["protocolos_ativos"]),
                          "paciente": p, "pii": pii_do_paciente(p)})
    return registros


def gerar_exportacoes(destino: Path = PASTA_BRUTO, semente: int = SEMENTE) -> dict[str, int]:
    """Gera todas as exportacoes brutas e devolve a contagem por arquivo."""
    rng = random.Random(semente)
    pessoas = Pessoas(rng)
    conjuntos = {
        "faq_medicos.jsonl": _mensagens_faq(rng, pessoas),
        "laudos.jsonl": _laudos(rng, pessoas),
        "receitas.jsonl": _receitas(rng, pessoas),
        "pedidos_indevidos.jsonl": _pedidos_indevidos(rng, pessoas),
        "casos_pacientes.jsonl": _casos(rng, pessoas),
    }
    destino.mkdir(parents=True, exist_ok=True)
    for nome, registros in conjuntos.items():
        with open(destino / nome, "w", encoding="utf-8") as f:
            for r in registros:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {nome: len(r) for nome, r in conjuntos.items()}


if __name__ == "__main__":
    for arquivo, n in gerar_exportacoes().items():
        print(f"{arquivo}: {n} registros")
