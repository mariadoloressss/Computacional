# ================================================================================
# PATRONES
#
# Generacion de patrones y estados iniciales para la red de Hopfield.
# Todos los patrones son matrices N x N con valores 0/1 (neurona en reposo /
# disparando).
#
#   - patron_aleatorio   : patrones aleatorios (sitios activos con probabilidad p)
#   - estado_aleatorio   : configuracion inicial aleatoria
#   - deformar           : cambia una fraccion aleatoria de neuronas de un patron
#   - patron_texto       : renderiza un caracter (digito, letra, simbolo) en N x N
#   - patrones_texto     : varios caracteres de una cadena -> array (P, N, N)
# ================================================================================

import os
import glob
import numpy as np


# --------------------------------------------------------------------------------
# Patrones y estados aleatorios
# --------------------------------------------------------------------------------
def patron_aleatorio(N, P, p=0.5, semilla=None):
    """P patrones aleatorios N x N (cada sitio activo con probabilidad p)."""
    rng = np.random.default_rng(semilla)
    return (rng.random((P, N, N)) < p).astype(np.int64)


def estado_aleatorio(N, p=0.5, semilla=None):
    """Configuracion inicial aleatoria N x N (cada sitio activo con prob. p)."""
    rng = np.random.default_rng(semilla)
    return (rng.random((N, N)) < p).astype(np.int64)


def deformar(patron, fraccion, semilla=None):
    """Devuelve una copia del patron con una 'fraccion' de neuronas cambiadas
    (0 <-> 1) elegidas al azar. Sirve para construir un estado inicial
    'difuminado' a partir de un patron almacenado.
    """
    rng = np.random.default_rng(semilla)
    estado = np.asarray(patron, dtype=np.int64).copy()
    plano = estado.ravel()
    n_cambios = int(round(fraccion * plano.size))
    idx = rng.choice(plano.size, size=n_cambios, replace=False)
    plano[idx] = 1 - plano[idx]
    return estado


# --------------------------------------------------------------------------------
# Patrones de texto (digitos, letras, simbolos) renderizados con PIL
# --------------------------------------------------------------------------------
def _ruta_fuente():
    """Localiza una fuente TrueType incluida con matplotlib (DejaVuSans-Bold).
    Las negritas conservan mejor el trazo al reducir a rejillas pequenas.
    """
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__),
                        'mpl-data', 'fonts', 'ttf')
    for nombre in ('DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        ruta = os.path.join(base, nombre)
        if os.path.exists(ruta):
            return ruta
    # Reserva: cualquier .ttf que encuentre
    candidatos = glob.glob(os.path.join(base, '*.ttf'))
    if candidatos:
        return candidatos[0]
    raise FileNotFoundError("No se encontro ninguna fuente TrueType.")


def patron_texto(caracter, N, fuente=None, margen=0.12, umbral=0.5,
                 supermuestreo=8):
    """Renderiza un caracter en una rejilla binaria N x N.

    El caracter se dibuja a alta resolucion y luego se reduce a N x N con
    suavizado, umbralizando para obtener 0/1 (1 = tinta).

    Parametros
    ----------
    caracter      : cadena de un solo caracter ('A', '7', '@', ...)
    N             : tamano de la rejilla de salida
    fuente        : ruta a un .ttf (por defecto, DejaVuSans-Bold de matplotlib)
    margen        : fraccion de margen alrededor del caracter
    umbral        : nivel de gris (0-1) por encima del cual el pixel se activa
    supermuestreo : factor de sobre-muestreo antes de reducir (mas = mas suave)
    """
    from PIL import Image, ImageDraw, ImageFont

    if fuente is None:
        fuente = _ruta_fuente()

    M = N * supermuestreo
    lado_util = int(M * (1 - 2 * margen))

    # Ajusta el tamano de letra para que el caracter ocupe el area util
    tam = lado_util
    img = Image.new("L", (M, M), color=0)
    draw = ImageDraw.Draw(img)
    while tam > 1:
        fnt = ImageFont.truetype(fuente, tam)
        izq, arr, der, aba = draw.textbbox((0, 0), caracter, font=fnt)
        if (der - izq) <= lado_util and (aba - arr) <= lado_util:
            break
        tam = int(tam * 0.92)

    # Centra el caracter en la imagen
    izq, arr, der, aba = draw.textbbox((0, 0), caracter, font=fnt)
    x = (M - (der - izq)) // 2 - izq
    y = (M - (aba - arr)) // 2 - arr
    draw.text((x, y), caracter, fill=255, font=fnt)

    # Reduce a N x N con suavizado y umbraliza
    img = img.resize((N, N), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return (arr > umbral).astype(np.int64)


def patrones_texto(cadena, N, **kwargs):
    """Convierte cada caracter de 'cadena' en un patron y los apila en (P, N, N).

    Ejemplo: patrones_texto("012", 30) devuelve 3 patrones con los digitos 0,1,2.

    AVISO: los digitos y letras son patrones DISPERSOS (a ~ 0.15) y muy
    CORRELACIONADOS entre si (toda la tinta se concentra en el centro). La regla
    de Hebb solo almacena bien patrones poco correlacionados y de actividad ~0.5,
    de modo que almacenar varios digitos/letras a la vez hace que la red caiga en
    un estado mezcla. Para almacenar varios patrones usar simbolos() o patrones
    aleatorios balanceados (ver simbolos() y patron_aleatorio()).
    """
    return np.stack([patron_texto(c, N, **kwargs) for c in cadena])


# --------------------------------------------------------------------------------
# Simbolos balanceados (a ~ 0.5) y poco correlacionados
#
# A diferencia de los digitos/letras, estos patrones llenan aproximadamente la
# mitad de la red y son estructuralmente distintos entre si, por lo que la regla
# de Hebb los almacena correctamente como atractores separados. Se generan con
# numpy (control total sobre la forma).
# --------------------------------------------------------------------------------
def simbolo(nombre, N):
    """Devuelve un simbolo balanceado N x N (valores 0/1).

    Nombres disponibles: 'disco', 'anillo', 'rombo', 'cuadrado',
    'franjas_v', 'franjas_h', 'tablero', 'cruz', 'aspa', 'triangulo'.
    """
    y, x = np.mgrid[0:N, 0:N]
    cx = cy = (N - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2

    if nombre == "disco":
        s = r2 < (0.42 * N) ** 2
    elif nombre == "anillo":
        s = (r2 < (0.45 * N) ** 2) & (r2 > (0.28 * N) ** 2)
    elif nombre == "rombo":
        s = (np.abs(x - cx) + np.abs(y - cy)) < 0.55 * N
    elif nombre == "cuadrado":
        s = (np.abs(x - cx) < 0.35 * N) & (np.abs(y - cy) < 0.35 * N)
    elif nombre == "franjas_v":
        s = (x // max(1, N // 6)) % 2 == 0
    elif nombre == "franjas_h":
        s = (y // max(1, N // 6)) % 2 == 0
    elif nombre == "tablero":
        b = max(1, N // 5)
        s = ((x // b) + (y // b)) % 2 == 0
    elif nombre == "cruz":
        s = (np.abs(x - cx) < 0.18 * N) | (np.abs(y - cy) < 0.18 * N)
    elif nombre == "aspa":
        s = (np.abs((x - cx) - (y - cy)) < 0.16 * N) | \
            (np.abs((x - cx) + (y - cy)) < 0.16 * N)
    elif nombre == "triangulo":
        s = (x + y) > N
    else:
        raise ValueError(f"Simbolo desconocido: {nombre}")

    return s.astype(np.int64)


def simbolos(nombres, N):
    """Apila varios simbolos en un array (P, N, N).

    Ejemplo: simbolos(['disco', 'franjas_v', 'tablero', 'triangulo'], 40)
    devuelve 4 patrones balanceados y poco correlacionados.
    """
    return np.stack([simbolo(n, N) for n in nombres])
