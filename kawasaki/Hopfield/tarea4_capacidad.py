# ================================================================================
# TAREA 4: Capacidad de almacenamiento de la red
#
# Con patrones ALEATORIOS (balanceados, a ~ 0.5) y T = 1e-4 se estudia como
# decae la recuperacion de la memoria al aumentar el numero de patrones
# almacenados P en una red de N = 20 (400 neuronas).
#
# Protocolo: para cada P se almacenan P patrones aleatorios y, para cada patron,
# se inicializa la red EN ese patron y se deja evolucionar (test de estabilidad).
# Un patron "se recuerda" si su solapamiento final supera 0.75. La fraccion de
# patrones recordados se promedia sobre varias realizaciones independientes.
#
# Se estima:
#   - la curva de fraccion de patrones recordados frente a alpha = P/N^2
#   - la capacidad alpha_c = P_c/N^2 (maximo P con TODOS los patrones recordados)
# y se compara con el valor clasico de la literatura alpha_c ~ 0.138.
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt

import hopfield as hop
import patrones as pat

# --- Parametros ---------------------------------------------------------------
N = 20                              # 400 neuronas (pedido en el enunciado)
T = 1e-4
PASOS_MC = 20                       # pasos para comprobar la estabilidad
UMBRAL_RECUERDO = 0.75              # m > 0.75 => patron recordado
LISTA_P = np.arange(4, 80, 4)       # numero de patrones almacenados a probar
N_REALIZACIONES = 12                # realizaciones independientes por cada P
ALPHA_CLASICO = 0.138               # valor de la literatura (Amit-Gutfreund-Sompolinsky)
FICHERO_SALIDA = "tarea4_capacidad.png"


def main():
    N2 = N * N
    alphas = LISTA_P / N2
    frac_media = np.zeros(len(LISTA_P))
    frac_error = np.zeros(len(LISTA_P))

    for k, P in enumerate(LISTA_P):
        fracciones = []
        for real in range(N_REALIZACIONES):
            patrones = pat.patron_aleatorio(N, P, p=0.5, semilla=1000 * real + P)
            recordados = 0
            for mu in range(P):
                _, m_hist = hop.evolucionar(patrones[mu], patrones, T,
                                            PASOS_MC, semilla=mu)
                if m_hist[-1, mu] > UMBRAL_RECUERDO:
                    recordados += 1
            fracciones.append(recordados / P)
        frac_media[k] = np.mean(fracciones)
        frac_error[k] = np.std(fracciones) / np.sqrt(N_REALIZACIONES)
        print(f"P = {P:3d}  alpha = {alphas[k]:.3f}   "
              f"fraccion recordada = {frac_media[k]:.3f} +/- {frac_error[k]:.3f}")

    # Capacidad: maximo P con (practicamente) todos los patrones recordados
    todos = frac_media >= 0.99
    if np.any(todos):
        idx = np.where(todos)[0][-1]
        P_c = LISTA_P[idx]
        alpha_c = alphas[idx]
    else:
        P_c, alpha_c = 0, 0.0
    print(f"\nCapacidad estimada (todos los patrones recordados): "
          f"P_c = {P_c}, alpha_c = {alpha_c:.3f}")
    print(f"Valor clasico de la literatura: alpha_c ~ {ALPHA_CLASICO}")

    # --- Figura ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(alphas, frac_media, yerr=frac_error, fmt="o-",
                color="tab:green", ms=5, capsize=3,
                label="fraccion recordada")
    ax.axvline(ALPHA_CLASICO, ls="--", color="tab:red", lw=1.3,
               label=f"$\\alpha_c \\approx {ALPHA_CLASICO}$ (literatura)")
    ax.axvline(alpha_c, ls=":", color="tab:blue", lw=1.3,
               label=f"$\\alpha_c \\approx {alpha_c:.3f}$ (todos recordados)")
    ax.set_xlabel(r"$\alpha = P / N^2$")
    ax.set_ylabel("Fraccion de patrones recordados")
    ax.set_title(f"Capacidad de la red de Hopfield  (N={N}, {N2} neuronas)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FICHERO_SALIDA, dpi=150)
    print(f"Figura guardada en: {FICHERO_SALIDA}")


if __name__ == "__main__":
    main()
