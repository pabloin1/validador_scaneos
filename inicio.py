"""Prepara el entorno local del validador con un solo comando."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


def python_del_entorno(entorno: Path) -> Path:
    if os.name == "nt":
        return entorno / "Scripts" / "python.exe"
    return entorno / "bin" / "python"


def main() -> int:
    proyecto = Path(__file__).resolve().parent
    entorno = proyecto / ".venv"
    requisitos = proyecto / "requirements.txt"

    if sys.version_info < (3, 10):
        print("Se requiere Python 3.10 o posterior.", file=sys.stderr)
        return 1
    if not requisitos.is_file():
        print(f"No se encontró el archivo de dependencias: {requisitos}", file=sys.stderr)
        return 1

    try:
        if not python_del_entorno(entorno).is_file():
            print("Creando el entorno virtual .venv...")
            venv.EnvBuilder(with_pip=True).create(entorno)

        python_entorno = python_del_entorno(entorno)
        print("Instalando o verificando las dependencias...")
        subprocess.run(
            [
                str(python_entorno),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(requisitos),
            ],
            cwd=proyecto,
            check=True,
        )
    except Exception as exc:
        print(f"No se pudo preparar el proyecto: {exc}", file=sys.stderr)
        return 1

    print("\nTodo está listo.")
    print("Para analizar PDFs, ejecuta el validador con el Python del entorno:")
    if os.name == "nt":
        print(r'.\.venv\Scripts\python.exe validar_escaneo.py "D:\SE-SSEE-DFE-0808-2026"')
    else:
        print('./.venv/bin/python validar_escaneo.py "/ruta/a/tu/carpeta"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
