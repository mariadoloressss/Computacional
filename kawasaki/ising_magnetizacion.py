"""
Modelo de Ising 2D - Magnetización en función de la temperatura.

Configuración inicial ordenada (s(i,j) = +1).
Termaliza el sistema y luego promedia |M| sobre muchas configuraciones
para estimar el valor esperado <|M|>.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

# --- Parámetros de la simulación ---
# Total = PASOS_TERMALIZACION + PASOS_MEDIDA = 10^6 pasos MC por temperatura,
# tal y como pide el enunciado.
N = 32                          # Tamaño del retículo (N x N)
PASOS_TERMALIZACION =  10_000   # pasos MC para termalizar (sin medir)
PASOS_MEDIDA        = 990_000   # pasos MC durante los que se acumulan medidas
INTERVALO_MUESTRA   = 10        # se mide |M| cada estos pasos (reduce correlación)
FICHERO_SALIDA = "magnetizacion_vs_temperatura.dat"


@njit(cache=True)
def paso_metropolis(grid, n, temp):
    """Un paso Monte Carlo = N*N intentos de flip (un 'sweep')."""
    for _ in range(n * n):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        s = grid[i, j]
        vecinos = (grid[(i + 1) % n, j] + grid[(i - 1) % n, j]
                 + grid[i, (j + 1) % n] + grid[i, (j - 1) % n])
        de = 2 * s * vecinos
        if de <= 0 or np.random.random() < np.exp(-de / temp):
            grid[i, j] = -s


@njit(cache=True)
def magnetizacion_absoluta(grid, n):
    """|M| por espín de la configuración actual."""
    s = 0
    for i in range(n):
        for j in range(n):
            s += grid[i, j]
    return abs(s) / (n * n)


@njit(cache=True)
def simular_a_temperatura(T, n, pasos_term, pasos_med, intervalo):
    """
    Devuelve <|M|> a temperatura T.
    Configuración inicial ordenada (todos +1).
    """
    grid = np.ones((n, n), dtype=np.int64)

    # Termalización (sin medir)
    for _ in range(pasos_term):
        paso_metropolis(grid, n, T)

    # Acumulación de medidas
    suma_m = 0.0
    n_muestras = 0
    for paso in range(pasos_med):
        paso_metropolis(grid, n, T)
        if paso % intervalo == 0:
            suma_m += magnetizacion_absoluta(grid, n)
            n_muestras += 1

    return suma_m / n_muestras


# --- Rango de temperaturas ---
# Densificamos puntos cerca de Tc ≈ 2.269 para resolver bien la transición.
T_baja  = np.linspace(1.0, 2.0, 6,  endpoint=False)   # fase ordenada
T_cerca = np.linspace(2.0, 2.6, 13)                   # zona crítica (densa)
T_alta  = np.linspace(2.7, 4.0, 8)                    # fase desordenada
temperaturas = np.concatenate([T_baja, T_cerca, T_alta])

magnetizaciones = np.zeros_like(temperaturas)

print("Modelo de Ising 2D — Magnetización vs Temperatura")
print(f"N = {N}, termalización = {PASOS_TERMALIZACION}, "
      f"medida = {PASOS_MEDIDA} (cada {INTERVALO_MUESTRA})")
print("=" * 60)

# Compilación previa de Numba con un punto pequeño (no cuenta en el tiempo)
print("Compilando con Numba...")
_ = simular_a_temperatura(2.0, N, 100, 100, 10)

t_inicio = time.time()
for k, T in enumerate(temperaturas):
    t0 = time.time()
    M = simular_a_temperatura(T, N, PASOS_TERMALIZACION,
                              PASOS_MEDIDA, INTERVALO_MUESTRA)
    magnetizaciones[k] = M
    print(f"  T = {T:5.3f}   <|M|> = {M:.4f}   ({time.time() - t0:5.1f} s)")

print("=" * 60)
print(f"Tiempo total: {time.time() - t_inicio:.1f} s")

# --- Guardar datos ---
np.savetxt(FICHERO_SALIDA,
           np.column_stack((temperaturas, magnetizaciones)),
           fmt='%.4f', delimiter=',',
           header='Temperatura, <|M|>')
print(f"Datos guardados en: {FICHERO_SALIDA}")

# --- Gráfica ---
plt.figure(figsize=(10, 6))
plt.plot(temperaturas, magnetizaciones, 'o-', color='blue',
         markersize=6, label=r'$\langle |M| \rangle$')
plt.axvline(x=2.269, color='red', linestyle='--', alpha=0.7,
            label=r'$T_c \approx 2.269$')
plt.xlabel('Temperatura $T$', fontsize=12)
plt.ylabel(r'Magnetización promedio $\langle |M| \rangle$', fontsize=12)
plt.title(f'Modelo de Ising {N}×{N}: Magnetización vs Temperatura',
          fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(temperaturas.min(), temperaturas.max())
plt.ylim(0, 1.05)
plt.savefig('magnetizacion_vs_temperatura.png', dpi=150, bbox_inches='tight')
print("Gráfica guardada como: magnetizacion_vs_temperatura.png")
plt.show()