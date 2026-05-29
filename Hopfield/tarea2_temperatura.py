# ================================================================================
# TAREA 2: Curva de solapamiento frente a temperatura
#
# Con un unico patron almacenado se estudia el solapamiento de equilibrio en
# funcion de la temperatura. Para cada T se parte del patron limpio (rama de
# recuperacion), se deja equilibrar y se promedia el solapamiento sobre los
# ultimos pasos, repitiendo sobre varias realizaciones para estimar el error.
#
# El resultado es una transicion de fase analoga a la magnetizacion del modelo
# de Ising: solapamiento ~1 (fase de recuperacion) a baja T, que cae a ~0 (fase
# de "olvido") por encima de una temperatura critica T_c.
#
# NOTA sobre la escala de T: con la codificacion 0/1 y la normalizacion 1/N^2 de
# los pesos, T_c NO vale 1 como en el Hopfield clasico de espines +-1; ademas,
# al ser un patron disperso (a ~ 0.16) las barreras de energia son pequenas y
# T_c resulta baja (~0.03). La fenomenologia (transicion de fase) es la misma.
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt

import hopfield as hop
import patrones as pat

# --- Parametros ---------------------------------------------------------------
N = 30                              # 900 neuronas
CARACTER = "A"                      # mismo patron que en la Tarea 1
TEMPERATURAS = np.linspace(0.005, 0.06, 20)
PASOS_MC = 120                      # pasos por simulacion
PASOS_EQUILIBRADO = 70              # se descartan los primeros (transitorio)
N_REALIZACIONES = 8                 # para barras de error
FICHERO_SALIDA = "tarea2_temperatura.png"


def main():
    patron = pat.patron_texto(CARACTER, N)
    patrones = patron[None, :, :]

    m_medio = np.zeros(len(TEMPERATURAS))
    m_error = np.zeros(len(TEMPERATURAS))

    for k, T in enumerate(TEMPERATURAS):
        medidas = []
        for r in range(N_REALIZACIONES):
            # Se parte del patron limpio para trazar la rama de recuperacion
            _, m_hist = hop.evolucionar(patron, patrones, T, PASOS_MC,
                                        semilla=100 * k + r)
            # Media sobre la parte de equilibrio (descartado el transitorio)
            medidas.append(m_hist[PASOS_EQUILIBRADO:, 0].mean())
        m_medio[k] = np.mean(medidas)
        m_error[k] = np.std(medidas) / np.sqrt(N_REALIZACIONES)
        print(f"T = {T:.4f}   <m> = {m_medio[k]:+.3f} +/- {m_error[k]:.3f}")

    # Estimacion sencilla de T_c: punto de mayor pendiente (caida del solapamiento)
    derivada = np.gradient(m_medio, TEMPERATURAS)
    Tc = TEMPERATURAS[np.argmin(derivada)]
    print(f"\nTemperatura critica aproximada (maxima pendiente): T_c ~ {Tc:.3f}")

    # --- Figura ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(TEMPERATURAS, m_medio, yerr=m_error, fmt="o-",
                color="tab:purple", ms=5, capsize=3, label="solapamiento")
    ax.axvline(Tc, ls="--", color="gray", lw=1.2,
               label=f"$T_c \\approx {Tc:.3f}$")
    ax.axhline(0, ls=":", color="gray", lw=1)
    ax.set_xlabel("Temperatura $T$")
    ax.set_ylabel("Solapamiento de equilibrio $\\langle m \\rangle$")
    ax.set_title(f"Solapamiento vs temperatura  (N={N}, patron '{CARACTER}')")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FICHERO_SALIDA, dpi=150)
    print(f"Figura guardada en: {FICHERO_SALIDA}")


if __name__ == "__main__":
    main()
