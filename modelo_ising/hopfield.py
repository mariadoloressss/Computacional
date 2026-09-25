"""
Modelo de Hopfield como red neuronal — Voluntario 2 del problema de Ising.

Implementación con código binario s_i in {0, 1} (neurona disparando o no).
Toda la notación sigue el enunciado del problema:

    Hamiltoniano:   H(s) = -(1/2) sum_{i,j} w_ij s_i s_j + sum_i theta_i s_i
    Pesos (Hebb):   w_ij = 1/[a(1-a)N] sum_{mu=1..P} (xi^mu_i - a)(xi^mu_j - a)
                    w_ii = 0
    Umbrales:       theta_i = (1/2) sum_j w_ij
    Solapamiento:   m_mu(s) = 1/[a(1-a)N] sum_i (xi^mu_i - a)(s_i - 1/2)
    Actividad:      a = <xi^mu_i>

La red es totalmente conectada (interacciones de largo alcance). La dinámica
es Metropolis asíncrono con probabilidad de aceptación min(1, exp(-dE/T)).
A T = 0 sólo se aceptan cambios con dE <= 0 (descenso del Hamiltoniano).

Tareas resueltas:
    Tarea 1 - Recuperación de patrones partiendo de una condición inicial
              deformada, a distintas temperaturas, midiendo m_mu(t). Se
              ejecuta dos veces:
                a) con un subconjunto de dígitos visualmente distintos
                   (patrones prefijados);
                b) con patrones aleatorios independientes con a = 1/2.
    Tarea 2 - Capacidad de almacenamiento alpha_c = P_c/N con N = 400, T = 0
              y patrones aleatorios; criterio de recuperación |m_mu| > 0.75.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Núcleo del modelo de Hopfield
# =============================================================================

def construir_pesos(xi, a):
    """
    Pesos sinápticos por regla Hebbiana:
        w_ij = 1/[a(1-a)N] sum_mu (xi^mu_i - a)(xi^mu_j - a),  con w_ii = 0.

    Parámetros
    ----------
    xi : array (P, N), valores en {0, 1}
    a  : actividad media <xi^mu_i>
    """
    P, N = xi.shape
    centrado = xi - a                                        # (P, N)
    w = (centrado.T @ centrado) / (a * (1.0 - a) * N)        # (N, N)
    np.fill_diagonal(w, 0.0)                                 # sin autoconexiones
    return w


def calcular_umbrales(w):
    """theta_i = (1/2) sum_j w_ij."""
    return 0.5 * np.sum(w, axis=1)


def solapamiento(s, xi_mu, a):
    """
    Solapamiento con un patrón:
        m_mu(s) = 1/[a(1-a)N] sum_i (xi^mu_i - a)(s_i - 1/2)

    El enunciado usa explícitamente (s_i - 1/2), no (s_i - a). Cuando s
    coincide con el patrón y a = 1/2 se cumple m_mu = 1 exactamente; para
    a != 1/2 el valor se acerca pero no es exactamente 1.
    """
    N = len(s)
    return float(np.sum((xi_mu - a) * (s - 0.5)) / (a * (1.0 - a) * N))


def paso_montecarlo(s, w, theta, T, rng):
    """
    Un paso Monte Carlo: N intentos de actualización asíncrona.
    Para cada intento:
        1) Se elige un sitio i al azar (con reemplazo).
        2) Se calcula dE para invertir s_i -> 1 - s_i:
              dE = (1 - 2 s_i)(theta_i - h_i),   h_i = sum_j w_ij s_j
        3) Se acepta el cambio con prob = min(1, exp(-dE/T)).

    A T = 0 sólo se aceptan los cambios con dE <= 0 (descenso de H).
    """
    N = len(s)
    for _ in range(N):
        i = rng.integers(N)
        h_i = w[i] @ s
        dE = (1.0 - 2.0 * s[i]) * (theta[i] - h_i)
        if dE <= 0.0:
            s[i] = 1 - s[i]
        elif T > 0.0 and rng.random() < np.exp(-dE / T):
            s[i] = 1 - s[i]
    return s


def relajar(s_inicial, w, theta, T, n_pasos, rng, xi=None, a=0.5):
    """
    Evoluciona la red durante n_pasos pasos MC. Si se le pasa la matriz xi
    de patrones almacenados, registra los solapamientos paso a paso.
    Devuelve (s_final, historia_m) con historia_m de forma (n_pasos, P)
    o (s_final, None) si no se pasa xi.
    """
    s = s_inicial.copy()
    historia_m = []
    for _ in range(n_pasos):
        s = paso_montecarlo(s, w, theta, T, rng)
        if xi is not None:
            historia_m.append([solapamiento(s, xi[mu], a) for mu in range(len(xi))])
    return s, (np.array(historia_m) if xi is not None else None)


# =============================================================================
# 2. Generación de patrones
# =============================================================================

def patrones_aleatorios(P, N, a=0.5, rng=None):
    """P patrones binarios independientes con Prob(xi_i = 1) = a."""
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random((P, N)) < a).astype(int)


def deformar(patron, frac_ruido, rng):
    """Versión deformada de un patrón: invierte una fracción frac_ruido de bits."""
    s = patron.copy()
    n_flips = int(round(frac_ruido * len(s)))
    indices = rng.choice(len(s), size=n_flips, replace=False)
    s[indices] = 1 - s[indices]
    return s


def patrones_digitos(subset=None):
    """
    Patrones predefinidos: dígitos 0-9 en una rejilla 10x10.
    Si subset es una lista de índices, devuelve sólo esos dígitos.
    Devuelve (xi, (alto, ancho)) con xi de tamaño (P, 100).

    Nota didáctica: si se usan los 10 dígitos completos, la regla Hebbiana
    estándar tiene dificultades porque varios dígitos están fuertemente
    correlacionados (p. ej. 0, 6, 8, 9 comparten gran parte de su estructura)
    y aparece un estado mezcla dominante. Usar un subconjunto visualmente
    distinto (p. ej. [0, 1, 4, 7]) elimina ese problema.
    """
    digitos_ascii = [
        # 0
        ["  ######  ", " ##    ## ", "##      ##", "##      ##", "##      ##",
         "##      ##", "##      ##", "##      ##", " ##    ## ", "  ######  "],
        # 1
        ["    ##    ", "   ###    ", "  ####    ", "    ##    ", "    ##    ",
         "    ##    ", "    ##    ", "    ##    ", "    ##    ", "  ######  "],
        # 2
        ["  ######  ", " ##    ## ", "##      ##", "       ## ", "      ##  ",
         "     ##   ", "    ##    ", "   ##     ", "  ##      ", "##########"],
        # 3
        ["  ######  ", " ##    ## ", "       ## ", "       ## ", "   ####   ",
         "       ## ", "       ## ", "       ## ", " ##    ## ", "  ######  "],
        # 4
        ["      ##  ", "     ###  ", "    # ##  ", "   #  ##  ", "  #   ##  ",
         " #    ##  ", "##########", "      ##  ", "      ##  ", "      ##  "],
        # 5
        ["##########", "##        ", "##        ", "##        ", "########  ",
         "       ## ", "       ## ", "       ## ", "##     ## ", " #######  "],
        # 6
        ["  ######  ", " ##    ## ", "##        ", "##        ", "########  ",
         "##     ## ", "##      ##", "##      ##", " ##    ## ", "  ######  "],
        # 7
        ["##########", "       ## ", "       ## ", "      ##  ", "      ##  ",
         "     ##   ", "     ##   ", "    ##    ", "    ##    ", "    ##    "],
        # 8
        ["  ######  ", " ##    ## ", "##      ##", "##      ##", " ######## ",
         "##      ##", "##      ##", "##      ##", " ##    ## ", "  ######  "],
        # 9
        ["  ######  ", " ##    ## ", "##      ##", "##      ##", " ######## ",
         "       ## ", "       ## ", "       ## ", " ##    ## ", "  ######  "],
    ]
    alto, ancho = 10, 10
    if subset is None:
        subset = list(range(10))

    xi = np.zeros((len(subset), alto * ancho), dtype=int)
    for k, d in enumerate(subset):
        assert len(digitos_ascii[d]) == alto, \
            f"Dígito {d}: {len(digitos_ascii[d])} filas, esperadas {alto}"
        for i, fila in enumerate(digitos_ascii[d]):
            assert len(fila) == ancho, \
                f"Dígito {d}, fila {i}: '{fila}' tiene {len(fila)} cols"
            for j, c in enumerate(fila):
                xi[k, i * ancho + j] = 1 if c == "#" else 0
    return xi, (alto, ancho)


# =============================================================================
# 3. Experimento genérico de recuperación a distintas T
# =============================================================================

def experimento_recuperacion(xi, forma, etiqueta, carpeta_salida,
                             mu_objetivo=0, frac_ruido=0.25, n_pasos=30,
                             temperaturas=(0.0, 0.1, 0.3, 1.0),
                             semilla=0, nombres_patrones=None):
    """
    Experimento de recuperación: almacena los patrones xi en la red por Hebb,
    parte de una versión deformada del patrón xi[mu_objetivo] y mide la
    evolución de m_mu(t) y la configuración final a varias T.

    Produce dos figuras en carpeta_salida:
        - patrones_{etiqueta}.png : los P patrones almacenados.
        - recuperacion_{etiqueta}.png : configuración inicial, intermedia y
          final + evolución de m_mu para cada temperatura.

    Parámetros
    ----------
    xi : array (P, N)
    forma : (alto, ancho) para reshape a imagen
    etiqueta : str, identificador para los archivos de salida
    mu_objetivo : índice del patrón a recuperar
    frac_ruido : fracción de bits invertidos en la condición inicial
    n_pasos : nº de pasos MC en la evolución
    temperaturas : iterable de valores de T a probar
    nombres_patrones : list[str], etiquetas opcionales para cada patrón
    """
    rng = np.random.default_rng(semilla)
    P, N = xi.shape
    a = float(xi.mean())                         # actividad media real
    w = construir_pesos(xi, a)
    theta = calcular_umbrales(w)

    if nombres_patrones is None:
        nombres_patrones = [f"$\\xi^{{{mu}}}$" for mu in range(P)]

    print(f"  Experimento '{etiqueta}': N = {N}, P = {P}, a = {a:.3f}, "
          f"alpha = {P/N:.3f}")

    # ---- Figura A: los P patrones almacenados ----
    n_cols = min(P, 5)
    n_filas = int(np.ceil(P / n_cols))
    fig, axes = plt.subplots(n_filas, n_cols,
                             figsize=(2.2 * n_cols, 2.4 * n_filas),
                             squeeze=False)
    for d in range(n_filas * n_cols):
        ax = axes.flat[d]
        if d < P:
            ax.imshow(xi[d].reshape(forma), cmap="binary", interpolation="nearest")
            ax.set_title(nombres_patrones[d], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Patrones almacenados ({etiqueta}): "
                 f"P = {P}, N = {N}, a = {a:.2f}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(carpeta_salida, f"patrones_{etiqueta}.png"), dpi=130)

    # ---- Figura B: recuperación a varias T ----
    s0 = deformar(xi[mu_objetivo], frac_ruido, rng)

    fig, axes = plt.subplots(len(temperaturas), 4,
                             figsize=(12, 2.7 * len(temperaturas)),
                             squeeze=False)
    for k, T in enumerate(temperaturas):
        rng_T = np.random.default_rng(semilla + 100 + k)
        s_final, historia = relajar(s0, w, theta, T, n_pasos, rng_T, xi=xi, a=a)

        rng_T2 = np.random.default_rng(semilla + 200 + k)
        n_intermedio = max(1, n_pasos // 3)
        s_intermedio, _ = relajar(s0, w, theta, T, n_intermedio, rng_T2)

        # Panel 0: inicial deformado
        axes[k, 0].imshow(s0.reshape(forma), cmap="binary", interpolation="nearest")
        axes[k, 0].set_ylabel(f"T = {T:g}", fontsize=11)
        if k == 0:
            axes[k, 0].set_title(f"Inicial (ruido {int(frac_ruido*100)}%)",
                                 fontsize=10)

        # Panel 1: intermedio
        axes[k, 1].imshow(s_intermedio.reshape(forma), cmap="binary",
                          interpolation="nearest")
        if k == 0:
            axes[k, 1].set_title(f"Tras {n_intermedio} pMC", fontsize=10)

        # Panel 2: final
        axes[k, 2].imshow(s_final.reshape(forma), cmap="binary",
                          interpolation="nearest")
        if k == 0:
            axes[k, 2].set_title(f"Final ({n_pasos} pMC)", fontsize=10)

        for j in range(3):
            axes[k, j].set_xticks([]); axes[k, j].set_yticks([])

        # Panel 3: m_mu(t)
        ax_m = axes[k, 3]
        for mu in range(P):
            es_obj = (mu == mu_objetivo)
            ax_m.plot(historia[:, mu],
                      color="C3" if es_obj else "gray",
                      lw=2.0 if es_obj else 0.8,
                      alpha=1.0 if es_obj else 0.5,
                      label=f"$m_{{{mu_objetivo}}}$ (objetivo)" if es_obj else None)
        ax_m.axhline(0.75, color="green", ls="--", lw=0.8, label="umbral 0.75")
        ax_m.set_xlabel("Paso MC")
        ax_m.set_ylabel("$m_\\mu$")
        ax_m.set_ylim(-0.6, 1.2)
        if k == 0:
            ax_m.set_title("Solapamientos $m_\\mu(t)$", fontsize=10)
        ax_m.legend(loc="lower right", fontsize=8)
        ax_m.grid(alpha=0.3)

    fig.suptitle(f"Recuperacion ({etiqueta}): patron {mu_objetivo} desde "
                 f"ruido {int(frac_ruido*100)}%", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(carpeta_salida, f"recuperacion_{etiqueta}.png"),
                dpi=130)


# =============================================================================
# 4. Tarea 2 — Capacidad de almacenamiento alpha_c
# =============================================================================

def tarea_capacidad(carpeta_salida, N=400, alphas=None, n_repeticiones=4,
                    n_sweeps=8, umbral_recuerdo=0.75, semilla=42):
    """
    Capacidad de almacenamiento del Hopfield a T = 0 con N neuronas.

    Para cada alpha = P/N:
        - Se generan n_repeticiones realizaciones independientes.
        - En cada una se crean P patrones aleatorios (a = 1/2) y se construye
          w por Hebb.
        - Para cada patrón mu se parte de s = xi^mu, se relajan n_sweeps
          pasos MC a T = 0 y se mide si |m_mu| > umbral_recuerdo (criterio
          de recuperación del enunciado: solapamiento > 0.75).
        - Se devuelve la fracción media de patrones recuperados.

    Como alpha_c se reporta el mayor alpha con fracción >= 0.99.
    """
    if alphas is None:
        alphas = np.linspace(0.02, 0.30, 15)
    a = 0.5
    rng = np.random.default_rng(semilla)

    medias, desvs = [], []
    for alpha in alphas:
        P = max(1, int(round(alpha * N)))
        fracciones = []
        for _ in range(n_repeticiones):
            xi = patrones_aleatorios(P, N, a=a, rng=rng)
            w = construir_pesos(xi, a)
            theta = calcular_umbrales(w)

            n_ok = 0
            for mu in range(P):
                s = xi[mu].copy()
                for _ in range(n_sweeps):
                    s = paso_montecarlo(s, w, theta, 0.0, rng)
                m = solapamiento(s, xi[mu], a)
                if abs(m) > umbral_recuerdo:
                    n_ok += 1
            fracciones.append(n_ok / P)
        medias.append(np.mean(fracciones))
        desvs.append(np.std(fracciones))
        print(f"  alpha = {alpha:.3f}  (P = {P:3d})  ->  "
              f"recuperación = {medias[-1]:.3f} ± {desvs[-1]:.3f}")

    medias = np.array(medias)
    desvs = np.array(desvs)

    # alpha_c: mayor alpha con fracción >= 0.99
    perfectos = medias >= 0.99
    alpha_c = float(alphas[perfectos].max()) if np.any(perfectos) else float("nan")

    # ---- Figura ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(alphas, medias, yerr=desvs, marker="o", ms=6, capsize=3,
                color="darkblue", lw=1.5, label="simulacion")
    ax.axvline(0.138, color="red", ls="--", alpha=0.8,
               label=r"$\alpha_c \approx 0.138$ (Amit-Gutfreund-Sompolinsky)")
    if not np.isnan(alpha_c):
        ax.axvline(alpha_c, color="green", ls=":", lw=1.8,
                   label=f"$\\alpha_c$ (sim.) = {alpha_c:.3f}")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_xlabel(r"$\alpha = P / N$", fontsize=12)
    ax.set_ylabel("Fraccion de patrones recuperados", fontsize=12)
    ax.set_title(f"Capacidad de almacenamiento (N = {N}, T = 0, "
                 f"|m| > {umbral_recuerdo})", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "tarea2_capacidad.png"), dpi=130)

    return alphas, medias, desvs, alpha_c


# =============================================================================
# 5. Ejecución principal
# =============================================================================

if __name__ == "__main__":
    CARPETA = "resultados_hopfield"
    os.makedirs(CARPETA, exist_ok=True)

    # -------------------------------------------------------------------------
    # Tarea 1a — Patrones predefinidos (dígitos visualmente distintos)
    # -------------------------------------------------------------------------
    # Nota: con los 10 dígitos completos (alpha = 0.1) la regla Hebbiana
    # estándar tiene dificultades por la fuerte correlación entre algunos
    # patrones (p. ej. 0, 6, 8, 9). Se usa un subconjunto con formas muy
    # distintas para que cada patrón sea un punto fijo estable.
    print("Tarea 1a - Recuperacion con digitos predefinidos (subset)")
    print("-" * 60)
    indices_digitos = [0, 1, 4, 7]
    xi_dig, forma = patrones_digitos(subset=indices_digitos)
    nombres_dig = [f"$\\xi^{{{d}}}$ (digito {d})" for d in indices_digitos]
    experimento_recuperacion(xi_dig, forma, etiqueta="digitos",
                             carpeta_salida=CARPETA,
                             mu_objetivo=2,             # patrón objetivo: dígito 4
                             frac_ruido=0.25, n_pasos=30,
                             temperaturas=(0.0, 0.1, 0.3, 1.0),
                             semilla=0,
                             nombres_patrones=nombres_dig)

    # -------------------------------------------------------------------------
    # Tarea 1b — Patrones aleatorios (caso ideal de la regla Hebbiana)
    # -------------------------------------------------------------------------
    print("\nTarea 1b - Recuperacion con patrones aleatorios")
    print("-" * 60)
    rng_aux = np.random.default_rng(7)
    # P = 8 patrones aleatorios en una rejilla 10x10 (alpha = 0.08).
    # Por debajo del límite teórico 0.138, todos son puntos fijos estables.
    xi_aleat = patrones_aleatorios(P=8, N=100, a=0.5, rng=rng_aux)
    experimento_recuperacion(xi_aleat, forma=(10, 10), etiqueta="aleatorios",
                             carpeta_salida=CARPETA,
                             mu_objetivo=3,
                             frac_ruido=0.30, n_pasos=30,
                             temperaturas=(0.0, 0.1, 0.3, 1.0),
                             semilla=1)

    # -------------------------------------------------------------------------
    # Tarea 2 — Capacidad de almacenamiento
    # -------------------------------------------------------------------------
    print("\nTarea 2 - Capacidad de almacenamiento (N=400, T=0)")
    print("-" * 60)
    alphas, medias, desvs, alpha_c = tarea_capacidad(CARPETA, N=400)

    print()
    print(f"Estimacion final: alpha_c ≈ {alpha_c:.3f}  "
          f"(valor teorico AGS: 0.138)")
    print(f"Figuras guardadas en: {CARPETA}/")

    plt.show()
