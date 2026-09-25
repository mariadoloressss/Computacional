from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# ==============================================================================
# ANIMACIÓN DEL SISTEMA SOLAR A PARTIR DE UN FICHERO DE DATOS
# ==============================================================================

file_in = "planets_data.dat"
file_out = "planetas"

x_min = -35
x_max = 35
y_min = -35
y_max = 35

interval = 20
show_trail = True
trail_width = 0.45
trail_length = 40
save_to_file = False
dpi = 150

planet_radius = [0.25, 0.05, 0.07, 0.07, 0.1, 0.18, 0.12, 0.12, 0.12]

planet_colors = [
    "yellow",
    "grey",
    "orange",
    "deepskyblue",
    "red",
    "saddlebrown",
    "tan",
    "cyan",
    "royalblue"
]

planet_names = [
    "Sol", "Mercurio", "Venus", "Tierra",
    "Marte", "Jupiter", "Saturno", "Urano", "Neptuno"
]

# ==============================================================================
# LECTURA DEL FICHERO DE DATOS
# ==============================================================================
with open(file_in, "r") as f:
    data_str = f.read()

frames_data = []

for frame_data_str in data_str.split("\n\n"):
    frame_data = []
    for planet_pos_str in frame_data_str.split("\n"):
        planet_pos = np.fromstring(planet_pos_str, sep=",")
        if planet_pos.size > 0:
            frame_data.append(planet_pos)
    if len(frame_data) > 0:
        frames_data.append(frame_data)

nplanets = len(frames_data[0])

# ==============================================================================
# CREACIÓN DE LA FIGURA
# ==============================================================================
fig, ax = plt.subplots()
ax.axis("equal")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_facecolor("black")

if not hasattr(planet_radius, "__iter__"):
    planet_radius = planet_radius * np.ones(nplanets)
else:
    if nplanets != len(planet_radius):
        raise ValueError("El número de radios no coincide con el número de planetas")

if nplanets != len(planet_colors):
    raise ValueError("El número de colores no coincide con el número de planetas")

if nplanets != len(planet_names):
    raise ValueError("El número de nombres no coincide con el número de planetas")

planet_points = []
planet_trails = []

for planet_pos, radius, color, name in zip(frames_data[0], planet_radius, planet_colors, planet_names):
    x, y = planet_pos
    planet_point = Circle((x, y), radius, color=color, label=name)
    ax.add_artist(planet_point)
    planet_points.append(planet_point)

    if show_trail:
        planet_trail, = ax.plot([x], [y], "-", linewidth=trail_width, color=color)
        planet_trails.append(planet_trail)

# ==============================================================================
# FUNCIÓN DE DETECCIÓN: ¿está este planeta capturado en este frame?
# El simulador escribe (0.0, 0.0) para planetas absorbidos.
# El Sol (índice 0) también está en (0,0), así que lo excluimos.
# ==============================================================================
def esta_capturado(j_planet, pos):
    if j_planet == 0:
        return False   # el Sol nunca está "capturado"
    return pos[0] == 0.0 and pos[1] == 0.0

# ==============================================================================
# FUNCIONES DE ANIMACIÓN
# ==============================================================================
def update(j_frame, frames_data, planet_points, planet_trails, show_trail):
    for j_planet, planet_pos in enumerate(frames_data[j_frame]):
        x, y = planet_pos

        if esta_capturado(j_planet, planet_pos):
            # Ocultar el círculo del planeta
            planet_points[j_planet].set_visible(False)
            # Borrar la estela
            if show_trail:
                planet_trails[j_planet].set_data([], [])
        else:
            planet_points[j_planet].set_visible(True)
            planet_points[j_planet].center = (x, y)

            if show_trail:
                xs_old, ys_old = planet_trails[j_planet].get_data()
                xs_new = np.append(xs_old, x)
                ys_new = np.append(ys_old, y)

                if len(xs_new) > trail_length:
                    xs_new = xs_new[-trail_length:]
                    ys_new = ys_new[-trail_length:]

                planet_trails[j_planet].set_data(xs_new, ys_new)

    return planet_points + planet_trails

def init_anim():
    if show_trail:
        for j_planet in range(nplanets):
            x0, y0 = frames_data[0][j_planet]
            planet_trails[j_planet].set_data([x0], [y0])
    return planet_points + planet_trails

# ==============================================================================
# LEYENDA
# ==============================================================================
ax.legend(
    handles=planet_points,
    loc="upper right",
    fontsize="small",
    facecolor="white"
)

# ==============================================================================
# GENERAR ANIMACIÓN O IMAGEN
# ==============================================================================
nframes = len(frames_data)

if nframes > 1:
    animation = FuncAnimation(
        fig,
        update,
        init_func=init_anim,
        fargs=(frames_data, planet_points, planet_trails, show_trail),
        frames=nframes,
        blit=True,
        interval=interval
    )

    if save_to_file:
        animation.save(f"{file_out}.mp4", dpi=dpi)
    else:
        plt.show()
else:
    if save_to_file:
        fig.savefig(f"{file_out}.pdf")
    else:
        plt.show()
