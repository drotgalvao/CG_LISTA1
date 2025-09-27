import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rotation_about_line(theta):
    """Retorna a matriz 4x4 homogênea da rotação em torno da reta x=2, y=1 (paralela a z)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    M = np.array(
        [
            [cos_t, -sin_t, 0, 2 - 2 * cos_t + sin_t],
            [sin_t, cos_t, 0, 1 - cos_t - 2 * sin_t],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return M


def make_unit_square_centered_at(cx, cy, z=0, side=1.0):
    h = side / 2.0
    return np.array(
        [
            [cx - h, cy - h, z],
            [cx + h, cy - h, z],
            [cx + h, cy + h, z],
            [cx - h, cy + h, z],
        ],
        dtype=float,
    )


def to_homogeneous(points):
    pts = np.asarray(points)
    n = pts.shape[0]
    h = np.ones((n, 4), dtype=float)
    h[:, :3] = pts
    return h.T


def from_homogeneous(hpoints):
    h = np.asarray(hpoints)
    return (h[:3, :] / h[3, :]).T


def main():
    frames = 36

    # teste numérico rápido
    theta_test = np.pi / 2  # 90 graus
    M_test = rotation_about_line(theta_test)
    print(f"Matriz de rotação homogênea (θ={np.degrees(theta_test):.1f}°):")
    print(M_test)  # printa a matriz calculada senos e cossenos
    p = np.array([3, 1, 0, 1])
    p_rot = M_test @ p  # operação matricial matriz x ponto
    print("\nPonto original:", p[:3])
    print("Ponto rotacionado:", p_rot[:3])

    # preparar figura e dados
    square = make_unit_square_centered_at(3.0, 1.0, z=0.0, side=1.0)
    hsquare = to_homogeneous(square)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")

    # Pré-calcula todas as rotações para determinar os limites do plot
    # Isso evita que o quadrado saia da tela durante a animação.
    thetas = 2 * np.pi * (np.arange(frames) / frames)
    all_pts = []
    for theta in thetas:
        M = rotation_about_line(theta)
        transformed = M @ hsquare
        pts3 = from_homogeneous(transformed)
        all_pts.append(pts3)
    all_pts = np.vstack(all_pts)

    min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
    max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
    pad = 0.5  # folga em torno da bounding box
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    (line_rot,) = ax.plot([], [], "r-", linewidth=2)
    pts_orig = np.vstack([square, square[0]])
    ax.plot(pts_orig[:, 0], pts_orig[:, 1], "k--", alpha=0.5)
    ax.scatter([2], [1], c="blue", s=50, label="eixo (2,1)")
    ax.legend()

    def init():
        line_rot.set_data([], [])
        return (line_rot,)

    def update(frame):
        # theta = 2 * -np.pi * (frame / frames) # sentido horário
        theta = 2 * np.pi * (frame / frames)  # sentido anti-horário
        M = rotation_about_line(theta)
        transformed = M @ hsquare
        pts3 = from_homogeneous(transformed)
        pts_rot = np.vstack([pts3, pts3[0]])
        line_rot.set_data(pts_rot[:, 0], pts_rot[:, 1])
        return (line_rot,)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=100
    )
    plt.show()


if __name__ == "__main__":
    main()
