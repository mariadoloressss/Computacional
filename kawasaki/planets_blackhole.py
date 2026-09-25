import numpy as np

# ==============================================================================
# 1. DATOS Y REESCALADO
# ==============================================================================
M_SOLAR = 1989100  # masa solar en unidades de 10^24 kg
AU_KM = 149.6         # 1 UA en unidades de 10^6 km

# [Nombre, Masa (10^24 kg), Perihelio (10^6 km), Excentricidad, Periodo real (días)]
datos_planetas = [
    ["Sol",      M_SOLAR, 0.0,    0.0,   0.0],
    ["Mercurio", 0.330,   46.0,   0.205, 88.0],
    ["Venus",    4.87,    107.5,  0.007, 224.7],
    ["Tierra",   5.97,    147.1,  0.017, 365.2],
    ["Marte",    0.642,   206.6,  0.094, 687.0],
    ["Jupiter",  1898,    740.5,  0.049, 4331.0],
    ["Saturno",  568,     1352.6, 0.057, 10747.0],
    ["Urano",    86.8,    2741.3, 0.046, 30589.0],
    ["Neptuno",  102,     4444.5, 0.011, 59800.0]
]

# ==============================================================================
# 2. PARÁMETROS DE SIMULACIÓN
# ==============================================================================
file_out = "planets_data.dat"

h = 0.002
pasos = 200000
t_conv = 58.1
frame_skip = 300

# ==============================================================================
# NUEVOS PARÁMETROS: COLISIÓN / CAPTURA
# ==============================================================================

# Factor de masa del Sol (1.0 = normal, 20.0 = agujero negro simulado)
FACTOR_MASA_SOL = 20.0

# Softening gravitacional (en UA). Evita la singularidad numérica cuando
# dos cuerpos se acercan mucho. Un valor pequeño (~0.01 UA) es suficiente.
# Sin softening, la fuerza diverge y los planetas "atraviesan" el Sol.
EPSILON_SOFT = 0.01   # UA

# Radio de captura del Sol (en UA).
# Si un planeta entra dentro de este radio, es absorbido (colapsa hacia el Sol).
# Para el Sol normal: ~0.005 UA (radio solar real ≈ 0.005 UA)
# Para agujero negro: radio de Schwarzschild ≈ 3e-7 UA (físicamente),
#   pero usamos un valor mayor por estabilidad numérica con este paso temporal.
#   ~0.02–0.05 UA es razonable para observar capturas visualmente.
R_CAPTURA_SOL = 0.05  # UA  ← ajusta este valor para ver más/menos capturas

# Índice del Sol en el array de planetas
IDX_SOL = 0

# ==============================================================================
# 3. CONSTRUCCIÓN DE MAGNITUDES INICIALES
# ==============================================================================
n_total = len(datos_planetas)

# Masas reescaladas en masas solares
m_all = np.array([p[1] / M_SOLAR for p in datos_planetas])
m_all[IDX_SOL] = FACTOR_MASA_SOL   # Sol modificado (agujero negro)

# Posiciones y velocidades en 2D
r_all = np.zeros((n_total, 2))
v_all = np.zeros((n_total, 2))

for i, p in enumerate(datos_planetas):
    if i == 0:
        continue
    r_ua = p[2] / AU_KM
    e = p[3]
    r_all[i, 0] = r_ua
    r_all[i, 1] = 0.0
    v_all[i, 0] = 0.0
    v_all[i, 1] = np.sqrt((1.0 + e) / r_ua)

# Lista de índices activos (planetas vivos)
activos = list(range(n_total))

# Registro de capturas
capturas = []  # (nombre_planeta, t_step)

# ==============================================================================
# 4. FUNCIÓN FÍSICA CON SOFTENING Y SOPORTE DE ÍNDICES ACTIVOS
# ==============================================================================
def calcular_fisica(pos, vel, masas, idx_activos, eps=EPSILON_SOFT):
    """
    Calcula aceleraciones y energía total para los cuerpos activos.
    
    - pos, vel, masas: arrays completos (n_total, 2), indexados por idx_activos
    - eps: parámetro de softening gravitacional
    
    La fuerza entre i y j usa dist_soft = sqrt(dist^2 + eps^2)
    en lugar de dist puro, evitando la divergencia en dist→0.
    Cuando dist >> eps, el comportamiento es newtoniano normal.
    """
    acc = np.zeros_like(pos)
    ep = 0.0

    act = idx_activos
    for ii in range(len(act)):
        i = act[ii]
        for jj in range(ii + 1, len(act)):
            j = act[jj]
            diff = pos[j] - pos[i]
            dist2 = np.dot(diff, diff)
            dist_soft = np.sqrt(dist2 + eps**2)   # ← softening aquí

            f_dir = diff / dist_soft**3            # dirección de fuerza suavizada

            acc[i] += masas[j] * f_dir
            acc[j] -= masas[i] * f_dir

            ep -= masas[i] * masas[j] / dist_soft

    ek = 0.5 * np.sum(
        masas[act] * np.sum(vel[act]**2, axis=1)
    )
    return acc, ek + ep


# ==============================================================================
# 5. BUCLE DE INTEGRACIÓN CON DETECCIÓN DE CAPTURAS
# ==============================================================================
historico_r    = []
historico_act  = []   # qué planetas están vivos en cada frame guardado
energias       = []
tiempos_orbitales = [[] for _ in range(n_total)]

r, v = r_all.copy(), v_all.copy()
m = m_all.copy()

a_actual, _ = calcular_fisica(r, v, m, activos)

print(f"Simulando con FACTOR_MASA_SOL={FACTOR_MASA_SOL}, "
      f"R_CAPTURA={R_CAPTURA_SOL} UA, EPSILON_SOFT={EPSILON_SOFT} UA\n")

for t_step in range(pasos):

    # --- Guardar frame ---
    if t_step % frame_skip == 0:
        historico_r.append(r.copy())
        historico_act.append(activos.copy())

    # --- Velocity Verlet: posición ---
    r_nuevo = r.copy()
    for i in activos:
        r_nuevo[i] = r[i] + h * v[i] + 0.5 * h**2 * a_actual[i]

    # --- Detectar capturas: intersección del segmento r[i]→r_nuevo[i]
    #     con el círculo de radio R_CAPTURA_SOL centrado en el Sol.
    #
    #     Esto resuelve el problema de "tunneling numérico": un planeta
    #     puede cruzar el radio de captura y salir al otro lado en un
    #     solo paso sin que la detección por punto final lo detecte.
    #
    #     Geometría: parametrizamos el segmento como P(t) = A + t*(B-A), t∈[0,1]
    #     donde A = r[i], B = r_nuevo[i], C = posición del Sol.
    #     Queremos saber si |P(t) - C| < R para algún t∈[0,1].
    #     Eso es una ecuación cuadrática en t; hay intersección si el
    #     discriminante ≥ 0 y al menos una raíz cae en [0,1].
    capturados_este_paso = []
    for i in activos:
        if i == IDX_SOL:
            continue

        A = r[i]                        # posición al inicio del step
        B = r_nuevo[i]                  # posición al final del step
        C = r_nuevo[IDX_SOL]            # posición del Sol (casi fija)
        R = R_CAPTURA_SOL

        d = B - A          # vector desplazamiento del planeta en este step
        f = A - C          # vector del Sol al punto inicial

        a_coef = np.dot(d, d)
        b_coef = 2 * np.dot(f, d)
        c_coef = np.dot(f, f) - R**2

        discriminante = b_coef**2 - 4 * a_coef * c_coef

        capturado = False
        if discriminante >= 0:
            sqrt_disc = np.sqrt(discriminante)
            t1 = (-b_coef - sqrt_disc) / (2 * a_coef)
            t2 = (-b_coef + sqrt_disc) / (2 * a_coef)
            # Hay captura si alguna raíz está en [0, 1]
            if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
                capturado = True

        if capturado:
            dist_al_sol = np.linalg.norm(A - C)
            capturados_este_paso.append(i)
            capturas.append((datos_planetas[i][0], t_step * h * t_conv))
            print(f"  *** {datos_planetas[i][0]} capturado por el Sol "
                  f"en t={t_step * h * t_conv:.1f} días "
                  f"(dist_inicio={dist_al_sol:.4f} UA) ***")

    # Eliminar capturados de la lista activa
    for i in capturados_este_paso:
        activos.remove(i)

    if len(activos) == 1:   # solo queda el Sol
        print("\nTodos los planetas han sido capturados. Fin de simulación.")
        break

    # --- Nueva aceleración y velocidad ---
    a_nueva, _ = calcular_fisica(r_nuevo, v, m, activos)

    v_nuevo = v.copy()
    for i in activos:
        v_nuevo[i] = v[i] + 0.5 * h * (a_actual[i] + a_nueva[i])

    _, etot = calcular_fisica(r_nuevo, v_nuevo, m, activos)
    energias.append(etot)

    # --- Detección de periodos orbitales ---
    for i in activos:
        if i == IDX_SOL:
            continue
        if r[i, 1] < 0 and r_nuevo[i, 1] >= 0:
            tiempos_orbitales[i].append(t_step * h * t_conv)

    r, v, a_actual = r_nuevo, v_nuevo, a_nueva

# ==============================================================================
# 6. ESCRITURA DEL ARCHIVO .DAT
# ==============================================================================
# Formato: para cada frame, se escriben TODOS los planetas (n_total líneas),
# con (0, 0) para los planetas ya capturados. Así el visualizador externo
# siempre recibe el mismo número de líneas por frame.

with open(file_out, "w") as f:
    for i_frame, (frame, act_frame) in enumerate(zip(historico_r, historico_act)):
        for pi in range(n_total):
            if pi in act_frame:
                x, y = frame[pi]
            else:
                x, y = 0.0, 0.0   # planeta ya absorbido → posición del Sol
            f.write(f"{x}, {y}\n")

        if i_frame != len(historico_r) - 1:
            f.write("\n")

print(f"\nArchivo de datos generado: {file_out}")

# ==============================================================================
# 7. RESULTADOS NUMÉRICOS
# ==============================================================================
print(f"\n{'Planeta':12} | {'Simulado':10} | {'Real':10} | {'Error Relat.'}")
print("-" * 55)

for i in range(1, n_total):
    if len(tiempos_orbitales[i]) > 1:
        p_sim = np.mean(np.diff(tiempos_orbitales[i]))
        p_real = datos_planetas[i][4]
        error = abs(p_sim - p_real) / p_real
        print(f"{datos_planetas[i][0]:12} | {p_sim:8.2f}d | {p_real:8.2f}d | {error:.2e}")
    else:
        print(f"{datos_planetas[i][0]:12} | Sin datos suficientes")

if energias:
    e_media = np.mean(energias)
    fluct = np.std(energias) / abs(e_media)
    print(f"\nConservación Energía: Media={e_media:.6f}, Fluct. Relativa={fluct:.2e}")

print(f"\nFrames guardados: {len(historico_r)}")
print(f"\nResumen de capturas ({len(capturas)} total):")
for nombre, t_dias in capturas:
    print(f"  - {nombre} absorbido en t={t_dias:.1f} días")
