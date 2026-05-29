# ================================================================================
# TAREA 3: Recuperacion con VARIOS patrones almacenados
#
# Se almacenan varios patrones a la vez y se observa como la red recupera el
# correspondiente partiendo de (i) el patron deformado y (ii) ruido aleatorio,
# midiendo el solapamiento con TODOS los patrones en funcion del tiempo.
#
# Se usan SIMBOLOS balanceados (a ~ 0.5) y poco correlacionados, no digitos:
# la regla de Hebb solo almacena bien patrones aproximadamente ortogonales. Con
# patrones dispersos y correlacionados (digitos/letras) la red caeria en estados
# mezcla (solapamiento alto con varios patrones a la vez).
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt

import hopfield as hop
import patrones as pat

# --- Parametros ---------------------------------------------------------------
N = 40
NOMBRES = ["disco", "franjas_v", "tablero", "triangulo"]
T = 1e-4
PASOS_MC = 30
FRAC_DEFORM = 0.25
PATRON_OBJETIVO = 0          # cual deformamos en el experimento (i)
SEMILLA = 1
FICHERO_SALIDA = "tarea3_varios_patrones.png"

COLORES = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def main():
    P = pat.simbolos(NOMBRES, N)
    n_pat = len(P)
    pasos = np.arange(PASOS_MC + 1)

    # --- (i) Patron objetivo deformado ---
    s0_def = pat.deformar(P[PATRON_OBJETIVO], FRAC_DEFORM, semilla=SEMILLA)
    sf_def, m_def = hop.evolucionar(s0_def, P, T, PASOS_MC, semilla=SEMILLA)

    # --- (ii) Estado inicial aleatorio ---
    s0_rnd = pat.estado_aleatorio(N, p=0.5, semilla=SEMILLA + 5)
    sf_rnd, m_rnd = hop.evolucionar(s0_rnd, P, T, PASOS_MC, semilla=SEMILLA + 5)

    print("Solapamiento final (inicio = patron deformado):",
          [f"{x:+.2f}" for x in m_def[-1]])
    print("Solapamiento final (inicio aleatorio):        ",
          [f"{x:+.2f}" for x in m_rnd[-1]])

    # --- Figura (3 filas) -----------------------------------------------------
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.1, 1.1], hspace=0.45,
                          wspace=0.3)

    def mostrar(ax, datos, titulo):
        ax.imshow(datos, cmap="binary", vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Fila 0: patrones almacenados
    for k in range(n_pat):
        ax = fig.add_subplot(gs[0, k])
        mostrar(ax, P[k], f"Patron {k}: {NOMBRES[k]}")

    # Fila 1: experimento desde patron deformado
    mostrar(fig.add_subplot(gs[1, 0]), s0_def,
            f"Inicial: '{NOMBRES[PATRON_OBJETIVO]}' {int(FRAC_DEFORM*100)}% def.")
    mostrar(fig.add_subplot(gs[1, 1]), sf_def, f"Tras {PASOS_MC} pMC")
    ax = fig.add_subplot(gs[1, 2:4])
    for k in range(n_pat):
        ax.plot(pasos, m_def[:, k], "o-", ms=3, color=COLORES[k],
                label=f"patron {k} ({NOMBRES[k]})")
    ax.set_title("Solapamiento con cada patron vs tiempo", fontsize=10)
    ax.set_xlabel("paso Monte Carlo"); ax.set_ylabel("m(s)")
    ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")

    # Fila 2: experimento desde estado aleatorio
    mostrar(fig.add_subplot(gs[2, 0]), s0_rnd, "Inicial: aleatorio")
    mostrar(fig.add_subplot(gs[2, 1]), sf_rnd, f"Tras {PASOS_MC} pMC")
    ax = fig.add_subplot(gs[2, 2:4])
    for k in range(n_pat):
        ax.plot(pasos, m_rnd[:, k], "o-", ms=3, color=COLORES[k],
                label=f"patron {k} ({NOMBRES[k]})")
    ax.set_title("Solapamiento con cada patron vs tiempo", fontsize=10)
    ax.set_xlabel("paso Monte Carlo"); ax.set_ylabel("m(s)")
    ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)
    ax.axhline(0, ls=":", color="gray", lw=1)
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle(f"Recuperacion con {n_pat} patrones almacenados  "
                 f"(N={N}, T={T:g})", fontsize=13)
    fig.savefig(FICHERO_SALIDA, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en: {FICHERO_SALIDA}")


if __name__ == "__main__":
    main()
