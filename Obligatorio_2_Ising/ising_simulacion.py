import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros de la simulación ---
N = 32              # Tamaño del retículo (N x N)
pasos_mc = 1000000  # 10^6 pasos Monte Carlo
fichero_salida = "magnetizacion_vs_temperatura.dat"

def inicializar_sistema(n, ordenada=True):
    """Crea una configuración inicial de espines"""
    if ordenada:
        return np.ones((n, n), dtype=int)
    else:
        return np.random.choice([-1, 1], size=(n, n))

def calcular_delta_e(grid, i, j, n):
    """Calcula el cambio de energía si se girara el espín en (i, j)"""
    s = grid[i, j]
    vecinos = (grid[(i + 1) % n, j] + grid[(i - 1) % n, j] +
               grid[i, (j + 1) % n] + grid[i, (j - 1) % n])
    return 2 * s * vecinos

def paso_metropolis(grid, n, temp):
    """Realiza un paso Monte Carlo (N^2 intentos de cambio)."""
    for _ in range(n**2):
        i, j = np.random.randint(0, n, size=2)
        de = calcular_delta_e(grid, i, j, n)
        if de <= 0:
            grid[i, j] *= -1
        elif np.random.rand() < np.exp(-de / temp):
            grid[i, j] *= -1
    return grid

def magnetizacion_promedio(grid):
    """Calcula el valor esperado de la magnetización"""
    return np.abs(np.mean(grid))

def simular_a_temperatura(T, n=32, pasos=1000000):
    """Ejecuta la simulación a una temperatura dada"""
    # Inicialización ordenada: todos los espines +1
    grid = inicializar_sistema(n, ordenada=True)
    
    # Evolucionar el sistema
    for paso in range(pasos):
        grid = paso_metropolis(grid, n, T)
        if (paso + 1) % 200000 == 0:
            print(f"  T={T:.2f}: {paso+1}/{pasos} pasos", end="\r")
    
    return magnetizacion_promedio(grid)

# --- Rango de temperaturas ---
# Cerca de Tc ~ 2.269, necesitamos un rango que incluya la transición
T_min = 1.0
T_max = 4.0
n_temps = 30

temperaturas = np.linspace(T_min, T_max, n_temps)
magnetizaciones = []

print("Simulando el modelo de Ising...")
print("=" * 50)

for T in temperaturas:
    print(f"\nTemperatura T = {T:.2f}")
    M = simular_a_temperatura(T, N, pasos_mc)
    magnetizaciones.append(M)
    print(f"  Magnetización: {M:.4f}")

# Guardar datos
magnetizaciones = np.array(magnetizaciones)
np.savetxt(fichero_salida, 
           np.column_stack((temperaturas, magnetizaciones)),
           fmt='%.4f', delimiter=',')
print(f"\nDatos guardados en: {fichero_salida}")

# --- Gráfica ---
plt.figure(figsize=(10, 6))
plt.plot(temperaturas, magnetizaciones, 'o-', color='blue', markersize=6, label='<|M|>')
plt.axvline(x=2.269, color='red', linestyle='--', label='Tc ≈ 2.269', alpha=0.7)
plt.xlabel('Temperatura (T)', fontsize=12)
plt.ylabel('Magnetización promedio <|M|>', fontsize=12)
plt.title('Modelo de Ising: Magnetización vs Temperatura', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(T_min, T_max)
plt.ylim(0, 1.05)

# Guardar figura
plt.savefig('magnetizacion_vs_temperatura.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGráfica guardada como: magnetizacion_vs_temperatura.png")