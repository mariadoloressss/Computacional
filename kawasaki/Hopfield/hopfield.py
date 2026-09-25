# ================================================================================
# HOPFIELD
#
# Modulo central para simular una red neuronal de Hopfield mediante el algoritmo
# de Metropolis (mismo metodo Monte Carlo que en el modelo de Ising).
#
# Codificacion de las neuronas:  s(i,j) = 1 (disparando)  /  0 (en reposo).
# Red cuadrada N x N con condiciones de contorno periodicas y totalmente
# conectada (interacciones de largo alcance), sin autoconexiones.
#
# Los pesos sinapticos w_{ij,kl} se construyen con la regla de Hebb a partir de
# P patrones almacenados. NUNCA se almacena la matriz completa de pesos (que
# tendria N^4 elementos): en su lugar se trabaja con las P proyecciones del
# estado sobre los patrones,
#
#       R^mu = sum_{k,l} (xi^mu_{k,l} - a^mu) s_{k,l},
#
# que se actualizan de forma incremental tras cada cambio de neurona. Esto
# reduce el coste de un paso Monte Carlo de O(N^4) a O(N^2 * P) (ver el
# "Consejo" del enunciado sobre expresar Delta H de forma inteligente).
#
# Magnitudes (con eta^mu = xi^mu - a^mu, a^mu = <xi^mu>):
#   - Umbral de disparo:   theta_{ij} = -(1/2N^2) sum_mu (eta^mu_{ij})^2   (constante)
#   - Campo local:         h_{nm} = (1/N^2) sum_mu eta^mu_{nm} (R^mu - eta^mu_{nm} s_{nm})
#   - Cambio de energia:   Delta H = (1 - 2 s_{nm}) (theta_{nm} - h_{nm})
#   - Solapamiento:        m^mu = R^mu / (N^2 a^mu (1 - a^mu))
# ================================================================================

import numpy as np
from numba import njit


# --------------------------------------------------------------------------------
# Preparacion de los patrones (operaciones vectorizadas con numpy)
# --------------------------------------------------------------------------------
def centrar_patrones(patrones):
    """Dado un array de patrones (P, N, N) con valores 0/1, devuelve:
        - eta: array (P, N, N) con los patrones centrados (xi^mu - a^mu)
        - a:   array (P,) con la actividad media a^mu de cada patron
    """
    patrones = np.asarray(patrones, dtype=np.float64)
    a = patrones.mean(axis=(1, 2))
    eta = patrones - a[:, None, None]
    return eta, a


def calcular_umbrales(eta, N):
    """Umbral de disparo theta_{ij}, constante en el tiempo:
        theta_{ij} = (1/2) sum_kl w_{ij,kl} = -(1/2N^2) sum_mu (eta^mu_{ij})^2
    """
    return -np.sum(eta**2, axis=0) / (2.0 * N * N)


def proyecciones_iniciales(estado, eta):
    """R^mu = sum_kl (xi^mu_kl - a^mu) s_kl para el estado inicial."""
    return np.einsum('mij,ij->m', eta, estado.astype(np.float64))


def solapamiento(estado, eta, a):
    """Solapamiento m^mu de un estado con cada patron almacenado.
    Acotado en [-1, 1]: vale +1 si se recupera el patron y -1 con el antipatron.
    """
    N = estado.shape[0]
    R = proyecciones_iniciales(estado, eta)
    norm = (N * N) * a * (1.0 - a)
    return R / norm


# --------------------------------------------------------------------------------
# Nucleo de la dinamica de Metropolis (compilado con Numba)
# --------------------------------------------------------------------------------
@njit(cache=True)
def _semilla(s):
    """Fija la semilla del generador interno de Numba (independiente del de numpy)."""
    np.random.seed(s)


@njit(cache=True)
def _evolucionar(estado, eta, theta, R, norm, T, n_pasos):
    """Evoluciona el sistema n_pasos pasos Monte Carlo a temperatura T.

    Un paso Monte Carlo equivale a N^2 intentos de cambio de neurona. Mide el
    solapamiento con cada patron en t = 0 y despues de cada paso completo.

    Modifica 'estado' y 'R' in place. Devuelve el historial de solapamientos
    de forma (n_pasos + 1, P).
    """
    N = estado.shape[0]
    N2 = N * N
    P = eta.shape[0]

    m_hist = np.empty((n_pasos + 1, P))
    for mu in range(P):
        m_hist[0, mu] = R[mu] / norm[mu]

    for paso in range(1, n_pasos + 1):
        for _ in range(N2):
            # 1. Elegir una neurona al azar
            n = np.random.randint(N)
            m = np.random.randint(N)
            s_nm = estado[n, m]

            # 2. Campo local h_{nm} usando las proyecciones R (coste O(P))
            campo = 0.0
            for mu in range(P):
                e = eta[mu, n, m]
                campo += e * (R[mu] - e * s_nm)
            campo /= N2

            # 3. Cambio de energia al voltear la neurona: s -> 1 - s
            ds = 1 - 2 * s_nm                      # +1 si s=0 ; -1 si s=1
            dH = ds * (theta[n, m] - campo)

            # 4. Criterio de Metropolis
            if dH <= 0.0 or np.random.random() < np.exp(-dH / T):
                for mu in range(P):
                    R[mu] += eta[mu, n, m] * ds    # actualizacion incremental
                estado[n, m] = 1 - s_nm

        for mu in range(P):
            m_hist[paso, mu] = R[mu] / norm[mu]

    return m_hist


@njit(cache=True)
def _evolucionar_con_instantaneas(estado, eta, theta, R, norm, T,
                                  n_pasos, intervalo):
    """Como _evolucionar, pero ademas guarda instantaneas del estado completo
    cada 'intervalo' pasos (para generar animaciones).

    Devuelve (m_hist, instantaneas) con instantaneas de forma (n_inst, N, N).
    """
    N = estado.shape[0]
    N2 = N * N
    P = eta.shape[0]

    m_hist = np.empty((n_pasos + 1, P))
    n_inst = n_pasos // intervalo + 1
    instantaneas = np.empty((n_inst, N, N), dtype=np.int8)

    for mu in range(P):
        m_hist[0, mu] = R[mu] / norm[mu]
    for i in range(N):
        for j in range(N):
            instantaneas[0, i, j] = estado[i, j]
    idx_inst = 1

    for paso in range(1, n_pasos + 1):
        for _ in range(N2):
            n = np.random.randint(N)
            m = np.random.randint(N)
            s_nm = estado[n, m]

            campo = 0.0
            for mu in range(P):
                e = eta[mu, n, m]
                campo += e * (R[mu] - e * s_nm)
            campo /= N2

            ds = 1 - 2 * s_nm
            dH = ds * (theta[n, m] - campo)

            if dH <= 0.0 or np.random.random() < np.exp(-dH / T):
                for mu in range(P):
                    R[mu] += eta[mu, n, m] * ds
                estado[n, m] = 1 - s_nm

        for mu in range(P):
            m_hist[paso, mu] = R[mu] / norm[mu]

        if paso % intervalo == 0 and idx_inst < n_inst:
            for i in range(N):
                for j in range(N):
                    instantaneas[idx_inst, i, j] = estado[i, j]
            idx_inst += 1

    return m_hist, instantaneas


# --------------------------------------------------------------------------------
# Interfaz de alto nivel (envoltorios comodos)
# --------------------------------------------------------------------------------
def evolucionar(estado_inicial, patrones, T, n_pasos, semilla=None):
    """Evoluciona 'estado_inicial' bajo la red definida por 'patrones'.

    Parametros
    ----------
    estado_inicial : (N, N) array de 0/1
    patrones       : (P, N, N) array de 0/1 con los patrones almacenados
    T              : temperatura
    n_pasos        : numero de pasos Monte Carlo
    semilla        : entero opcional para reproducibilidad

    Devuelve
    --------
    estado_final : (N, N) array de 0/1
    m_hist       : (n_pasos + 1, P) solapamiento con cada patron en cada paso
    """
    eta, a = centrar_patrones(patrones)
    N = estado_inicial.shape[0]
    theta = calcular_umbrales(eta, N)
    norm = (N * N) * a * (1.0 - a)

    estado = estado_inicial.astype(np.int64).copy()
    R = proyecciones_iniciales(estado, eta)

    if semilla is not None:
        _semilla(semilla)
    m_hist = _evolucionar(estado, eta, theta, R, norm, float(T), int(n_pasos))
    return estado, m_hist


def evolucionar_con_instantaneas(estado_inicial, patrones, T, n_pasos,
                                 intervalo=1, semilla=None):
    """Como evolucionar(), pero devuelve tambien instantaneas del estado.

    Devuelve (estado_final, m_hist, instantaneas).
    """
    eta, a = centrar_patrones(patrones)
    N = estado_inicial.shape[0]
    theta = calcular_umbrales(eta, N)
    norm = (N * N) * a * (1.0 - a)

    estado = estado_inicial.astype(np.int64).copy()
    R = proyecciones_iniciales(estado, eta)

    if semilla is not None:
        _semilla(semilla)
    m_hist, inst = _evolucionar_con_instantaneas(
        estado, eta, theta, R, norm, float(T), int(n_pasos), int(intervalo))
    return estado, m_hist, inst
