# ================================================================================
# TAREA 1: Recuperacion de un unico patron almacenado
#
# Se almacena un solo patron en una red de N x N >= 900 neuronas y se observa
# como la red lo recupera a temperatura muy baja (T = 1e-4) partiendo de:
#   (i)  una condicion inicial aleatoria
#   (ii) el patron deformado (un % de sus neuronas cambiadas)
#
# Se mide el solapamiento m(s) con el patron en funcion del tiempo (pasos Monte
# Carlo) para cuantificar como la red se aproxima al patron almacenado.
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt

import hopfield as hop
import patrones as pat

# --- Parametros ---------------------------------------------------------------
N = 30                  # red de N x N = 900 neuronas (minimo pedido)
CARACTER = "A"          # patron a almacenar
T = 1e-4                # temperatura (baja: dinamica practicamente determinista)
PASOS_MC = 30           # pasos Monte Carlo
FRAC_DEFORM = 0.30      # fraccion de neuronas cambiadas en el patron deformado
SEMILLA = 1
FICHERO_SALIDA = "tarea1_recuperacion.png"


def main():
    # Patron unico almacenado
    patron = pat.patron_texto(CARACTER, N)
    patrones = patron[None, :, :]        # forma (1, N, N)

    # --- (i) Condicion inicial aleatoria ---
    s0_aleatorio = pat.estado_aleatorio(N, p=0.5, semilla=SEMILLA)
    sf_aleatorio, m_aleatorio = hop.evolucionar(
        s0_aleatorio, patrones, T, PASOS_MC, semilla=SEMILLA)

    # --- (ii) Patron deformado ---
    s0_deform = pat.deformar(patron, FRAC_DEFORM, semilla=SEMILLA)
    sf_deform, m_deform = hop.evolucionar(
        s0_deform, patrones, T, PASOS_MC, semilla=SEMILLA + 1)

    print(f"Solapamiento final (inicio aleatorio):  {m_aleatorio[-1, 0]:+.4f}")
    print(f"Solapamiento final (patron deformado):  {m_deform[-1, 0]:+.4f}")

    # --- Figura ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    pasos = np.arange(PASOS_MC + 1)

    def mostrar(a, datos, titulo):
        a.imshow(datos, cmap="binary", vmin=0, vmax=1)
        a.set_title(titulo, fontsize=10)
        a.set_xticks([]); a.set_yticks([])

    # Fila 0: experimento desde patron deformado
    mostrar(ax[0, 0], patron, "Patron almacenado")
    mostrar(ax[0, 1], s0_deform, f"Inicial: {int(FRAC_DEFORM*100)}% deformado")
    mostrar(ax[0, 2], sf_deform, f"Tras {PASOS_MC} pMC")
    ax[0, 3].plot(pasos, m_deform[:, 0], "o-", color="tab:blue", ms=3)
    ax[0, 3].set_title("Solapamiento vs tiempo", fontsize=10)
    ax[0, 3].set_xlabel("paso Monte Carlo"); ax[0, 3].set_ylabel("m(s)")
    ax[0, 3].set_ylim(-1.05, 1.05); ax[0, 3].grid(alpha=0.3)
    ax[0, 3].axhline(1, ls=":", color="gray", lw=1)

    # Fila 1: experimento desde estado aleatorio
    mostrar(ax[1, 0], patron, "Patron almacenado")
    mostrar(ax[1, 1], s0_aleatorio, "Inicial: aleatorio")
    mostrar(ax[1, 2], sf_aleatorio, f"Tras {PASOS_MC} pMC")
    ax[1, 3].plot(pasos, m_aleatorio[:, 0], "o-", color="tab:red", ms=3)
    ax[1, 3].set_title("Solapamiento vs tiempo", fontsize=10)
    ax[1, 3].set_xlabel("paso Monte Carlo"); ax[1, 3].set_ylabel("m(s)")
    ax[1, 3].set_ylim(-1.05, 1.05); ax[1, 3].grid(alpha=0.3)
    ax[1, 3].axhline(1, ls=":", color="gray", lw=1)
    ax[1, 3].axhline(-1, ls=":", color="gray", lw=1)

    fig.suptitle(f"Recuperacion de un patron  (N={N}, T={T:g})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FICHERO_SALIDA, dpi=150)
    print(f"Figura guardada en: {FICHERO_SALIDA}")


if __name__ == "__main__":
    main()
