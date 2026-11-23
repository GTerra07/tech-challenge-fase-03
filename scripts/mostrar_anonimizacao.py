"""
Mostra um registro bruto (com dado pessoal FALSO, gerado para o trabalho) e o mesmo
texto depois da anonimização — o ponto único por onde todo texto passa antes do dataset
e antes do prompt.

Uso:
    python scripts/mostrar_anonimizacao.py
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.seguranca.anonimizacao import anonimizar, residuos_pii  # noqa: E402

registro = json.loads((RAIZ / "data" / "bruto" / "laudos.jsonl").open(encoding="utf-8").readline())
bruto = registro["texto_bruto"].split("\n")[:3]
limpo = anonimizar("\n".join(bruto)).split("\n")

print("ANTES (exportação bruta, dado pessoal falso):")
for linha in bruto:
    print("  " + linha)
print("\nDEPOIS (src/seguranca/anonimizacao.py):")
for linha in limpo:
    print("  " + linha)
print(f"\nresíduos de dado pessoal no texto anonimizado: {len(residuos_pii(chr(10).join(limpo)))}")
