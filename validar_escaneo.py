"""Evalúa la calidad de páginas escaneadas en PDF y genera reportes CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

ANCHO_NORMALIZADO = 1240
MAX_PDFS = None
# Cambia este valor para limitar las pruebas; usa None para procesarlos todos.
MAX_WORKERS = min(4, os.cpu_count() or 1)
PUNTAJE_MINIMO_APROBACION = 70
UMBRALES = {
    "ancho_min_px": 900,
    "nitidez_min": 500.0,
    "contraste_min": 120.0,
    "inclinacion_max": 1.5,
}


def pagina_a_imagen(page: fitz.Page, doc: fitz.Document, dpi: int = 150) -> np.ndarray:
    """Extrae la imagen original cuando la página contiene una sola; si no, renderiza."""
    imagenes = page.get_images(full=True)
    if len(imagenes) == 1:
        try:
            datos = doc.extract_image(imagenes[0][0])["image"]
            imagen = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            if imagen is not None:
                return imagen
        except (KeyError, ValueError, cv2.error):
            pass

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    imagen = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)


def normalizar_tamano(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    factor = ANCHO_NORMALIZADO / ancho
    if abs(factor - 1) < 0.10:
        return imagen
    return cv2.resize(imagen, (ANCHO_NORMALIZADO, max(1, int(alto * factor))), interpolation=cv2.INTER_AREA)


def estimar_fondo(gris: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    cerrado = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(cerrado, (0, 0), 25)


def nitidez(gris: np.ndarray) -> float:
    alto, ancho = gris.shape
    valores = []
    for fila in range(6):
        for columna in range(6):
            mosaico = gris[fila * alto // 6:(fila + 1) * alto // 6,
                           columna * ancho // 6:(columna + 1) * ancho // 6]
            if mosaico.size:
                valores.append(cv2.Laplacian(mosaico, cv2.CV_64F).var())
    return float(np.percentile(valores, 90)) if valores else 0.0


def contraste_tinta(normalizada: np.ndarray) -> float:
    return float(255 - np.percentile(normalizada, 0.5))


def inclinacion(normalizada: np.ndarray) -> float:
    pequena = cv2.resize(normalizada, None, fx=0.5, fy=0.5)
    binaria = (pequena < 170).astype(np.uint8) * 255
    alto, ancho = binaria.shape
    def buscar(candidatos: np.ndarray, mejor_actual: float = -1.0,
               angulo_actual: float = 0.0) -> tuple[float, float]:
        mejor_local, angulo_local = mejor_actual, angulo_actual
        for candidato in candidatos:
            matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), float(candidato), 1)
            rotada = cv2.warpAffine(binaria, matriz, (ancho, alto), flags=cv2.INTER_NEAREST)
            puntuacion = np.var(rotada.sum(axis=1).astype(np.float64))
            if puntuacion > mejor_local:
                mejor_local, angulo_local = puntuacion, float(candidato)
        return mejor_local, angulo_local

    # Búsqueda gruesa y refinamiento local conservan la precisión final de 0.25°.
    mejor, angulo = buscar(np.arange(-8, 8.01, 0.5))
    inicio = max(-8.0, angulo - 0.5)
    fin = min(8.0, angulo + 0.5)
    _, angulo = buscar(np.arange(inicio, fin + 0.01, 0.25), mejor, angulo)
    return angulo


def evaluar_pagina(imagen: np.ndarray, ancho_original: int | None = None) -> tuple[dict[str, Any], list[str]]:
    ancho_original = ancho_original or imagen.shape[1]
    imagen = normalizar_tamano(imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    fondo = estimar_fondo(gris)
    normalizada = np.clip(gris.astype(float) / np.maximum(fondo, 1) * 255, 0, 255).astype(np.uint8)

    metricas = {
        "ancho_px": int(ancho_original),
        "nitidez": round(nitidez(gris), 1),
        "contraste": round(contraste_tinta(normalizada), 1),
        "inclinacion": round(inclinacion(normalizada), 2),
    }
    problemas = []
    if metricas["ancho_px"] < UMBRALES["ancho_min_px"]:
        problemas.append("Resolución baja")
    if metricas["nitidez"] < UMBRALES["nitidez_min"]:
        problemas.append("Borrosa / poco legible")
    if metricas["contraste"] < UMBRALES["contraste_min"]:
        problemas.append("Contraste bajo (texto desvanecido)")
    if abs(metricas["inclinacion"]) > UMBRALES["inclinacion_max"]:
        problemas.append(f"Inclinada ({metricas['inclinacion']}°)")
    return metricas, problemas


def validar_pdf(ruta: Path) -> list[tuple[int, dict[str, Any], list[str], str]]:
    filas = []
    with fitz.open(ruta) as doc:
        if doc.needs_pass:
            raise ValueError("El PDF está protegido con contraseña")
        for numero, pagina in enumerate(doc, 1):
            if len(pagina.get_text().strip()) > 50 and not pagina.get_images():
                filas.append((numero, {}, [], "PDF digital (no escaneado)"))
                continue
            imagen = pagina_a_imagen(pagina, doc)
            metricas, problemas = evaluar_pagina(imagen, imagen.shape[1])
            filas.append((numero, metricas, problemas, "OK" if not problemas else "REVISAR"))
    if not filas:
        raise ValueError("El PDF no contiene páginas")
    return filas


def puntuar_pagina(metricas: dict[str, Any]) -> int:
    """Devuelve un puntaje de 0 a 100 basado en legibilidad y orientación."""
    nitidez_valor = metricas["nitidez"]
    if nitidez_valor >= UMBRALES["nitidez_min"]:
        puntaje_nitidez = 100.0
    elif nitidez_valor < 250:
        puntaje_nitidez = 60 + 20 * nitidez_valor / 250
    else:
        puntaje_nitidez = 80 + 20 * (nitidez_valor - 250) / 250

    angulo = abs(metricas["inclinacion"])
    puntaje_angulo = 100.0 if angulo <= 2 else max(10.0, 100 - (angulo - 2) * 15)

    contraste = metricas["contraste"]
    puntaje_contraste = 100.0 if contraste >= UMBRALES["contraste_min"] else 50 + 50 * contraste / UMBRALES["contraste_min"]

    ancho = metricas["ancho_px"]
    puntaje_resolucion = 100.0 if ancho >= UMBRALES["ancho_min_px"] else 50 + 50 * ancho / UMBRALES["ancho_min_px"]
    ponderado = (
        puntaje_angulo * 0.60
        + puntaje_nitidez * 0.30
        + puntaje_contraste * 0.05
        + puntaje_resolucion * 0.05
    )
    return round(max(1, min(100, ponderado)))


def resumir_pdf(ruta: Path) -> dict[str, Any]:
    """Evalúa todas las páginas y resume resultados en una fila por documento."""
    paginas = validar_pdf(ruta)
    escaneadas = [(numero, metricas) for numero, metricas, _, _ in paginas if metricas]
    digitales = len(paginas) - len(escaneadas)

    if escaneadas:
        puntajes = [puntuar_pagina(metricas) for _, metricas in escaneadas]
        puntuacion = round(sum(puntajes) / len(puntajes))
        total = len(escaneadas)
        angulos = sum(abs(metricas["inclinacion"]) > UMBRALES["inclinacion_max"]
                      for _, metricas in escaneadas)
        borrosas = sum(metricas["nitidez"] < UMBRALES["nitidez_min"]
                       for _, metricas in escaneadas)
        contraste_bajo = sum(metricas["contraste"] < UMBRALES["contraste_min"]
                             for _, metricas in escaneadas)
        resolucion_baja = sum(metricas["ancho_px"] < UMBRALES["ancho_min_px"]
                              for _, metricas in escaneadas)
        observaciones = []
        for cantidad, nombre in (
            (angulos, "inclinación fuera del rango recomendado"),
            (borrosas, "nitidez reducida"),
            (contraste_bajo, "contraste bajo"),
            (resolucion_baja, "resolución baja"),
        ):
            if cantidad:
                observaciones.append(f"{nombre} en {cantidad} de {total} páginas escaneadas")
        if digitales:
            observaciones.append(f"{digitales} páginas digitales se excluyeron del puntaje visual")
    else:
        puntuacion = 100
        observaciones = ["PDF digital con texto; no requiere evaluación de escaneo"]

    resultado = "APROBADO" if puntuacion >= PUNTAJE_MINIMO_APROBACION else "REVISAR"
    return {
        "archivo": ruta.name,
        "paginas": len(paginas),
        "puntuacion": puntuacion,
        "resultado": resultado,
        "observaciones": "; ".join(observaciones) if observaciones else "Sin problemas relevantes detectados",
    }


def _inicializar_proceso() -> None:
    """Evita que cada proceso inicie a su vez varios hilos de OpenCV."""
    cv2.setNumThreads(1)


def _procesar_pdf_seguro(ruta: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return resumir_pdf(Path(ruta)), None
    except Exception as exc:
        return None, str(exc)


def procesar(
    destino: Path,
    archivo_csv: Path,
    max_pdfs: int | None = MAX_PDFS,
    workers: int = MAX_WORKERS,
) -> int:
    pdfs = sorted(p for p in destino.glob("*.pdf") if p.is_file()) if destino.is_dir() else [destino]
    if not pdfs:
        print(f"No se encontraron archivos PDF en: {destino}", file=sys.stderr)
        return 2
    if workers < 1:
        print("El número de procesos debe ser mayor que cero.", file=sys.stderr)
        return 2
    if destino.is_dir() and max_pdfs is not None:
        if max_pdfs < 1:
            print("El límite debe ser mayor que cero o None.", file=sys.stderr)
            return 2
        total = len(pdfs)
        pdfs = pdfs[:max_pdfs]
        print(f"Procesando {len(pdfs)} de {total} PDFs (límite: {max_pdfs}).")

    reporte = []
    hubo_error = False
    if workers > 1 and len(pdfs) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_inicializar_proceso) as pool:
            resultados = pool.map(_procesar_pdf_seguro, map(str, pdfs))
            procesados = zip(pdfs, resultados)
            for pdf, (fila, error) in procesados:
                if error:
                    hubo_error = True
                    print(f"\n{pdf.name}: ERROR - {error}", file=sys.stderr)
                    continue
                print(f"\n{fila['archivo']}: {fila['puntuacion']}% - {fila['resultado']}")
                print(f"  Observaciones: {fila['observaciones']}")
                reporte.append(fila)
    else:
        for pdf in pdfs:
            fila, error = _procesar_pdf_seguro(str(pdf))
            if error:
                hubo_error = True
                print(f"\n{pdf.name}: ERROR - {error}", file=sys.stderr)
                continue
            print(f"\n{fila['archivo']}: {fila['puntuacion']}% - {fila['resultado']}")
            print(f"  Observaciones: {fila['observaciones']}")
            reporte.append(fila)

    if reporte:
        archivo_csv.parent.mkdir(parents=True, exist_ok=True)
        columnas = ["archivo", "paginas", "puntuacion", "resultado", "observaciones"]
        with archivo_csv.open("w", newline="", encoding="utf-8-sig") as salida:
            escritor = csv.DictWriter(salida, fieldnames=columnas, extrasaction="ignore")
            escritor.writeheader()
            escritor.writerows(reporte)
        print(f"\nReporte por PDF guardado en: {archivo_csv}")
    return 1 if hubo_error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa escaneos PDF hechos con celular.")
    parser.add_argument("destino", nargs="?", type=Path, default=Path("documento.pdf"),
                        help="PDF individual o carpeta con PDFs")
    parser.add_argument("--csv", type=Path, default=Path("reporte.csv"),
                        help="Ruta de salida del CSV al procesar una carpeta")
    parser.add_argument("--max-pdfs", type=int, default=None,
                        help=f"Límite de PDFs (por defecto: {MAX_PDFS}; no aplica a un archivo individual)")
    parser.add_argument("--todos", action="store_true",
                        help="Procesa todos los PDFs de la carpeta, ignorando MAX_PDFS")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Procesos paralelos (por defecto: {MAX_WORKERS}; usa 1 para desactivarlos)")
    args = parser.parse_args()
    if not args.destino.exists():
        parser.error(f"No existe la ruta: {args.destino}")
    if args.destino.is_file() and args.destino.suffix.lower() != ".pdf":
        parser.error("El archivo debe tener extensión .pdf")
    limite = None if args.todos else (args.max_pdfs if args.max_pdfs is not None else MAX_PDFS)
    if limite is not None and limite < 1:
        parser.error("--max-pdfs debe ser mayor que cero")
    if args.workers < 1:
        parser.error("--workers debe ser mayor que cero")
    return procesar(args.destino, args.csv, limite, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
