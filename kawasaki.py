"""
============================================================
 Modelo de Ising 2D con dinámica de Kawasaki
 Versión refactorizada con:
   - Configuración inicial separada en fases (no aleatoria)
   - ANNEALING: ramp-up de T baja a T alta, sin reinicializar
     entre temperaturas (corrige falta de equilibrado a T > Tc)
   - Promedio sobre realizaciones independientes
   - Estimación de Tc por ajuste parabólico
   - Grid denso de T cerca de la región crítica
   - Snapshots tomados también con annealing
============================================================
"""

import os
import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# ==========================================
# BLOQUE 1: MOTOR DE SIMULACIÓN (JIT)
# ==========================================

@nb.njit
def seed_numba(s):
    """Semilla del RNG interno de Numba (separado del de NumPy puro)."""
    np.random.seed(s)


@nb.njit
def init_lattice_phase_separated(N, m0):
    """
    Inicializa el retículo con separación de fases ya establecida:
      - Fila 0 (arriba) fijada a -1
      - Fila N-1 (abajo) fijada a +1
      - Filas activas inferiores (fracción x=(1+m0)/2) puestas a +1
      - Filas activas superiores (fracción 1-x) puestas a -1
    La magnetización inicial coincide casi exactamente con m0.
    """
    lattice = np.zeros((N, N), dtype=np.int8)
    lattice[0, :] = -1
    lattice[N - 1, :] = 1

    x = (1.0 + m0) / 2.0
    N_active = (N - 2) * N
    num_up_target = int(round(N_active * x))

    placed_up = 0
    finished = False
    for i in range(N - 2, 0, -1):
        if placed_up + N <= num_up_target:
            for j in range(N):
                lattice[i, j] = 1
            placed_up += N
        else:
            remainder = num_up_target - placed_up
            for j in range(N):
                if j < remainder:
                    lattice[i, j] = 1
                else:
                    lattice[i, j] = -1
            placed_up = num_up_target
            for ii in range(i - 1, 0, -1):
                for j in range(N):
                    lattice[ii, j] = -1
            finished = True
            break

    if not finished:
        for i in range(1, N - 1):
            if lattice[i, 0] == 0:
                for j in range(N):
                    lattice[i, j] = -1
    return lattice


@nb.njit
def calculate_total_energy(lattice):
    """
    Energía total del sistema. Omite bonds horizontales dentro de las filas
    fijas (0 y N-1), que son una constante y no afectan a la dinámica ni a
    las fluctuaciones (Cv, χ).
    """
    N = lattice.shape[0]
    E = 0.0
    for i in range(1, N - 1):
        for j in range(N):
            S = lattice[i, j]
            E += -S * lattice[i, (j + 1) % N]   # bond horizontal (PBC en j)
            E += -S * lattice[i + 1, j]         # bond vertical hacia abajo
    for j in range(N):
        E += -lattice[0, j] * lattice[1, j]
    return E


@nb.njit
def mcmc_step(lattice, E, T, num_attempts):
    """Realiza num_attempts intentos de intercambio Kawasaki."""
    N = lattice.shape[0]
    beta = 1.0 / T

    for _ in range(num_attempts):
        i1 = np.random.randint(1, N - 1)
        j1 = np.random.randint(0, N)

        while True:
            d = np.random.randint(0, 4)
            if d == 0:
                i2 = i1 - 1
                j2 = j1
            elif d == 1:
                i2 = i1
                j2 = (j1 + 1) % N
            elif d == 2:
                i2 = i1 + 1
                j2 = j1
            else:
                i2 = i1
                j2 = (j1 - 1) % N
            if 1 <= i2 < N - 1:
                break

        s1 = lattice[i1, j1]
        s2 = lattice[i2, j2]
        if s1 == s2:
            continue

        sum1 = 0
        if not (i1 - 1 == i2 and j1 == j2):
            sum1 += lattice[i1 - 1, j1]
        if not (i1 + 1 == i2 and j1 == j2):
            sum1 += lattice[i1 + 1, j1]
        jm = (j1 - 1) % N
        if not (i1 == i2 and jm == j2):
            sum1 += lattice[i1, jm]
        jp = (j1 + 1) % N
        if not (i1 == i2 and jp == j2):
            sum1 += lattice[i1, jp]

        sum2 = 0
        if not (i2 - 1 == i1 and j2 == j1):
            sum2 += lattice[i2 - 1, j2]
        if not (i2 + 1 == i1 and j2 == j1):
            sum2 += lattice[i2 + 1, j2]
        jm2 = (j2 - 1) % N
        if not (i2 == i1 and jm2 == j1):
            sum2 += lattice[i2, jm2]
        jp2 = (j2 + 1) % N
        if not (i2 == i1 and jp2 == j1):
            sum2 += lattice[i2, jp2]

        dE = 2 * s1 * sum1 + 2 * s2 * sum2

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            lattice[i1, j1] = s2
            lattice[i2, j2] = s1
            E += dE

    return E


@nb.njit
def measurement_loop(lattice, E, T, mc_steps, N_active, split_idx, N):
    """Bucle de medida acelerado con Numba. Devuelve momentos y rho_y."""
    E_sum = 0.0
    E_sq_sum = 0.0
    M_sum = 0.0
    M_sq_sum = 0.0
    rho_y_sum = np.zeros(N)

    for _ in range(mc_steps):
        E = mcmc_step(lattice, E, T, N_active)
        E_sum += E
        E_sq_sum += E * E

        M_bot = 0.0
        for i in range(split_idx, N - 1):
            for j in range(N):
                M_bot += lattice[i, j]
        M_sum += M_bot
        M_sq_sum += M_bot * M_bot

        for i in range(N):
            s = 0.0
            for j in range(N):
                s += (lattice[i, j] + 1) * 0.5
            rho_y_sum[i] += s / N

    return E, E_sum, E_sq_sum, M_sum, M_sq_sum, rho_y_sum


# ==========================================
# BLOQUE 2: REALIZACIONES Y PROMEDIOS (CON ANNEALING)
# ==========================================

def run_single_realization(N, m0, T_array, mc_steps, term_steps, seed):
    """
    Una realización completa CON ANNEALING.
    
    La red se inicializa UNA SOLA VEZ (cold start, configuración separada
    en fases) en la T más baja. Para cada T sucesiva NO se reinicializa la
    red: simplemente se termaliza el sistema desde su estado anterior. Así,
    conforme T sube por encima de Tc, el sistema se mezcla orgánicamente
    en lugar de tener que arrancar desde un estado totalmente equivocado
    a cada temperatura.
    """
    seed_numba(seed)
    np.random.seed(seed)

    N_active = (N - 2) * N
    x_fraction = (1.0 + m0) / 2.0
    split_idx = int(1 + (N - 2) * (1 - x_fraction))

    # Ordenar T_array de menor a mayor para el annealing
    sort_idx = np.argsort(T_array)
    T_sorted = T_array[sort_idx]

    nT = len(T_array)
    E_avg = np.zeros(nT)
    E_sq_avg = np.zeros(nT)
    M_avg = np.zeros(nT)
    M_sq_avg = np.zeros(nT)
    rho_y = np.zeros((nT, N))

    # Inicialización ÚNICA al principio del barrido
    lattice = init_lattice_phase_separated(N, m0)
    E = calculate_total_energy(lattice)

    for k, T in enumerate(T_sorted):
        # Termalización a la nueva T (SIN reiniciar la red)
        for _ in range(term_steps):
            E = mcmc_step(lattice, E, T, N_active)

        # Medidas
        E, E_sum, E_sq_sum, M_sum, M_sq_sum, rho_y_sum = measurement_loop(
            lattice, E, T, mc_steps, N_active, split_idx, N
        )

        # Guardamos en la posición ORIGINAL de T_array
        orig_k = sort_idx[k]
        E_avg[orig_k] = E_sum / mc_steps
        E_sq_avg[orig_k] = E_sq_sum / mc_steps
        M_avg[orig_k] = M_sum / mc_steps
        M_sq_avg[orig_k] = M_sq_sum / mc_steps
        rho_y[orig_k] = rho_y_sum / mc_steps

    return {'E': E_avg, 'E_sq': E_sq_avg,
            'M': M_avg, 'M_sq': M_sq_avg,
            'rho_y': rho_y}


def run_simulation(N, m0, T_array, mc_steps, term_steps, n_realiz, base_seed=0):
    """Promedia n_realiz realizaciones independientes (cada una con annealing)."""
    print(f"\n--- N={N}, m0={m0}, {n_realiz} realizaciones (annealing) ---")
    t_ini = time.time()

    acc = None
    for r in range(n_realiz):
        seed = base_seed + r * 1000 + N
        t0 = time.time()
        result = run_single_realization(N, m0, T_array, mc_steps, term_steps, seed)
        elapsed = time.time() - t0
        print(f"  Realización {r+1}/{n_realiz} (semilla={seed}) "
              f"completada en {elapsed:.1f}s")

        if acc is None:
            acc = {k: v.copy() for k, v in result.items()}
        else:
            for k in acc:
                acc[k] += result[k]

    for k in acc:
        acc[k] /= n_realiz

    x_fraction = (1.0 + m0) / 2.0
    split_idx = int(1 + (N - 2) * (1 - x_fraction))
    n_bot = (N - 1 - split_idx) * N

    Cv = (acc['E_sq'] - acc['E']**2) / (N**2 * T_array**2)
    Chi = (acc['M_sq'] - acc['M']**2) / (N**2 * T_array)
    M_per_part = acc['M'] / n_bot
    E_per_part = acc['E'] / (N * N)

    print(f"  Tiempo total para N={N}: {time.time() - t_ini:.1f}s")

    return {'T': T_array.copy(),
            'E': E_per_part, 'Cv': Cv,
            'M': M_per_part, 'Chi': Chi,
            'rho_y': acc['rho_y']}


# ==========================================
# BLOQUE 3: SNAPSHOTS CON ANNEALING Y AJUSTE DE Tc
# ==========================================

def take_snapshots_annealed(N, m0, T_array, target_Ts, term_steps, seed=0):
    """
    Genera snapshots usando annealing sobre la misma grilla T_array de la
    simulación principal. Como T_array es densa, el sistema se equilibra
    bien en cada paso del ramp y los snapshots a T altas son realistas
    (sin "memoria" de la separación de fases inicial).
    """
    seed_numba(seed)
    np.random.seed(seed)

    sorted_T = np.sort(T_array)

    # Mapa: índice en sorted_T -> lista de target_Ts asociadas (más cercanas)
    snap_indices = {}
    for tgt in target_Ts:
        idx = int(np.argmin(np.abs(sorted_T - tgt)))
        snap_indices.setdefault(idx, []).append(tgt)

    lattice = init_lattice_phase_separated(N, m0)
    E = calculate_total_energy(lattice)
    N_active = (N - 2) * N

    snapshots = {}
    for k, T in enumerate(sorted_T):
        for _ in range(term_steps):
            E = mcmc_step(lattice, E, T, N_active)
        if k in snap_indices:
            for tgt in snap_indices[k]:
                snapshots[tgt] = lattice.copy()
    return snapshots


def tc_parabolic_fit(T_array, observable):
    """Estima Tc ajustando una parábola alrededor del máximo del observable."""
    k = int(np.argmax(observable))
    if k == 0 or k == len(observable) - 1:
        return T_array[k]
    T_loc = T_array[k - 1:k + 2]
    O_loc = observable[k - 1:k + 2]
    a, b, _ = np.polyfit(T_loc, O_loc, 2)
    if a >= 0:
        return T_array[k]
    return -b / (2 * a)


# ==========================================
# BLOQUE 4: GRÁFICAS Y ANÁLISIS
# ==========================================

def plot_extrapolation(N_list, Tc_cv, Tc_chi, out_dir, m0):
    """Gráfica Tc(N) vs 1/N con extrapolación lineal."""
    inv_N = 1.0 / np.array(N_list)
    fit_cv = np.polyfit(inv_N, Tc_cv, 1)
    fit_chi = np.polyfit(inv_N, Tc_chi, 1)
    line = np.linspace(0, max(inv_N) * 1.1, 50)

    plt.figure(figsize=(8, 5))
    plt.plot(inv_N, Tc_cv, 'bo', label='Picos de $C_v$')
    plt.plot(line, np.polyval(fit_cv, line), 'b--',
             label=f'Fit $C_v$: $T_c(\\infty)={fit_cv[1]:.3f}$')
    plt.plot(inv_N, Tc_chi, 'ro', label=r'Picos de $\chi$')
    plt.plot(line, np.polyval(fit_chi, line), 'r--',
             label=f'Fit $\\chi$: $T_c(\\infty)={fit_chi[1]:.3f}$')
    plt.axhline(2.269, color='gray', linestyle=':',
                label='Onsager $T_c=2.269$')
    plt.title(rf'Extrapolación de $T_c$ ($m_0={m0}$)')
    plt.xlabel('1 / N')
    plt.ylabel('$T_c(N)$')
    plt.legend()
    plt.grid(True)
    fname = os.path.join(out_dir, f"extrapolacion_m0_{m0}.jpg")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Archivo guardado]: {fname}")
    return fit_cv[1], fit_chi[1]


def main_analysis(m0, N_list, T_array, target_T_snaps,
                  mc_steps, term_steps, n_realiz, out_dir):
    """Pipeline completo para un valor de m0."""
    print(f"\n{'=' * 60}")
    print(f"  ANÁLISIS PARA m0 = {m0}")
    print(f"{'=' * 60}")

    results = {}
    Tc_cv_list = []
    Tc_chi_list = []

    for N in N_list:
        res = run_simulation(N, m0, T_array, mc_steps, term_steps, n_realiz)
        results[N] = res
        Tc_cv_list.append(tc_parabolic_fit(res['T'], res['Cv']))
        Tc_chi_list.append(tc_parabolic_fit(res['T'], res['Chi']))
        print(f"  -> Tc(N={N}): C_v={Tc_cv_list[-1]:.3f}, "
              f"chi={Tc_chi_list[-1]:.3f}")

    metrics = [
        ('M', 'Magnetización por dominio', r'$\langle m\rangle$'),
        ('Cv', 'Calor específico', r'$C_v$'),
        ('Chi', 'Susceptibilidad', r'$\chi$'),
        ('E', 'Energía por partícula', r'$\langle E\rangle / N^2$'),
    ]
    for key, title, ylabel in metrics:
        plt.figure(figsize=(8, 5))
        for N in N_list:
            plt.plot(results[N]['T'], results[N][key], 'o-', label=f'N={N}')
        plt.title(f"{title} ($m_0={m0}$)")
        plt.xlabel('T')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{key.lower()}_m0_{m0}.jpg")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Archivo guardado]: {fname}")

    Tc_inf_cv, Tc_inf_chi = plot_extrapolation(
        N_list, Tc_cv_list, Tc_chi_list, out_dir, m0
    )
    print(f"\n  Tc extrapolado (C_v): {Tc_inf_cv:.4f}")
    print(f"  Tc extrapolado (chi): {Tc_inf_chi:.4f}")

    # Perfil de densidad rho(y) para el N mayor
    N_max = N_list[-1]
    plt.figure(figsize=(8, 5))
    step = max(1, len(T_array) // 6)
    for i, T_val in enumerate(results[N_max]['T']):
        if i % step == 0:
            plt.plot(range(N_max), results[N_max]['rho_y'][i],
                     label=f'T={T_val:.2f}')
    plt.title(f'Perfil de densidad $\\rho(y)$ (N={N_max}, $m_0={m0}$)')
    plt.xlabel('y (fila)')
    plt.ylabel(r'$\rho(y)$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    fname = os.path.join(out_dir, f"perfil_densidad_N{N_max}_m0_{m0}.jpg")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Archivo guardado]: {fname}")

    # Snapshots CON ANNEALING (recorre toda la grilla T)
    print(f"\n  Generando snapshots para N={N_max} con annealing...")
    snaps = take_snapshots_annealed(N_max, m0, T_array, target_T_snaps,
                                    term_steps)
    for T_snap, snap in snaps.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(snap, cmap='coolwarm', interpolation='nearest',
                   vmin=-1, vmax=1)
        plt.title(f'Configuración a T={T_snap:.2f} '
                  f'(N={N_max}, $m_0={m0}$)')
        plt.axis('off')
        fname = os.path.join(out_dir,
                             f"snapshot_N{N_max}_T{T_snap:.1f}_m0_{m0}.jpg")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Archivo guardado]: {fname}")


# ==========================================
# BLOQUE 5: EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    N_LIST = [32, 64, 128]

    T_coarse = np.linspace(1.5, 3.5, 11)
    T_dense  = np.linspace(2.0, 2.6, 13)
    T_ARRAY  = np.unique(np.round(np.concatenate([T_coarse, T_dense]), 3))

    TARGET_T_SNAPS = [1.5, 2.2, 3.0]

    MC_STEPS    = 30_000
    TERM_STEPS  = 15_000
    N_REALIZ    = 3

    OUT_DIR = "resultados_ising"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SIMULACIÓN ISING-KAWASAKI (con annealing)")
    print("=" * 60)
    print(f"  N_list     = {N_LIST}")
    print(f"  T_array    = {T_ARRAY}  ({len(T_ARRAY)} puntos)")
    print(f"  mc_steps   = {MC_STEPS}")
    print(f"  term_steps = {TERM_STEPS}")
    print(f"  n_realiz   = {N_REALIZ}")
    print(f"  out_dir    = {OUT_DIR}")

    t_total = time.time()

    main_analysis(m0=0.0, N_list=N_LIST, T_array=T_ARRAY,
                  target_T_snaps=TARGET_T_SNAPS,
                  mc_steps=MC_STEPS, term_steps=TERM_STEPS,
                  n_realiz=N_REALIZ, out_dir=OUT_DIR)

    main_analysis(m0=0.4, N_list=N_LIST, T_array=T_ARRAY,
                  target_T_snaps=TARGET_T_SNAPS,
                  mc_steps=MC_STEPS, term_steps=TERM_STEPS,
                  n_realiz=N_REALIZ, out_dir=OUT_DIR)

    print(f"\nSimulación completada en {(time.time() - t_total)/60:.1f} min")