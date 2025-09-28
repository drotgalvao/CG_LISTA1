import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rodrigues(u, theta):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    R = (
        np.cos(theta) * np.eye(3)
        + (1 - np.cos(theta)) * np.outer(u, u)
        + np.sin(theta) * K
    )
    return R


def rotation_around_D_with_axis_translation(theta):
    """
    Retorna (R, t, H) para a rotação de ângulo theta em torno da reta
    D = {(-t, 1-t, t) | t in R}, com translação ao longo do eixo de magnitude (2/pi)*theta
    no sentido do eixo.
    Forma final: x' = R x + t, H homogênea 4x4.
    """
    # reta D: ponto P0 e direção d
    P0 = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, -1.0, 1.0])
    u = d / np.linalg.norm(d)

    R = rodrigues(u, theta)
    s = (2.0 / np.pi) * theta
    v_axis = s * u

    t = P0 + v_axis - R.dot(P0)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t

    return R, t, H


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
    np.set_printoptions(precision=6, suppress=True)

    # explicar e imprimir exemplo
    print(
        "Operador afim: rotação em torno de D = {(-t,1-t,t)} com translação axial (2/pi)*theta"
    )
    theta_example = np.pi / 2
    R_ex, t_ex, H_ex = rotation_around_D_with_axis_translation(theta_example)
    print("\nExemplo (theta = 90 deg):")
    print("R =")
    print(R_ex)
    print("t =", t_ex)
    print("H =")
    print(H_ex)

    # preparar figura (quadrado) e animação 3D
    square = make_unit_square_centered_at(1.5, 0.0, z=0.0, side=1.0)
    hsquare = to_homogeneous(square)

    frames = 72

    # calcular posições finais (aplicar H(theta_max)) para determinar bounding box
    thetas = np.linspace(0, 2 * np.pi, frames)
    all_pts = []
    for th in thetas:
        R, t, H = rotation_around_D_with_axis_translation(th)
        transformed = (H @ hsquare).T
        pts3 = transformed[:, :3]
        all_pts.append(pts3)
    all_pts = np.vstack(all_pts)

    # preparar plot 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # desenhar superficie do plano base para referência (usar a mesma P0 e diretores que definem D?)
    # Para visual, desenhamos um plano aproximado perpendicular a u? Aqui desenhamos uma superfície que contém P0 and spans two directions.
    P0 = np.array([0.0, 1.0, 0.0])
    u_dir = np.array([-2.0, 4.0, -2.0])
    v_dir = np.array([-1.0, -1.0, 1.0])
    u_dir = u_dir / np.linalg.norm(u_dir)
    v_dir = v_dir / np.linalg.norm(v_dir)
    U = np.linspace(-2.0, 2.0, 10)
    V = np.linspace(-2.0, 2.0, 10)
    plane_pts = np.array([[P0 + uu * u_dir + vv * v_dir for uu in U] for vv in V])
    X = plane_pts[:, :, 0]
    Y = plane_pts[:, :, 1]
    Z = plane_pts[:, :, 2]
    ax.plot_surface(X, Y, Z, color=(0.9, 0.9, 0.9), alpha=0.5, linewidth=0)

    # desenhar eixo D como uma reta
    d = np.array([-1.0, -1.0, 1.0])
    tline = np.linspace(-3, 3, 10)
    line_pts = np.array([P0 + tt * d for tt in tline])
    ax.plot(
        line_pts[:, 0],
        line_pts[:, 1],
        line_pts[:, 2],
        color="black",
        linewidth=1.5,
        label="eixo D",
    )

    # quadrado inicial e coleção animada
    orig_poly = Poly3DCollection(
        [square], facecolors=(0.2, 0.6, 0.2, 0.4), edgecolors="k"
    )
    ax.add_collection3d(orig_poly)
    anim_poly = Poly3DCollection(
        [square], facecolors=(0.2, 0.4, 0.8, 0.6), edgecolors="k"
    )
    ax.add_collection3d(anim_poly)

    # pontos móveis
    moving_pts = ax.scatter(square[:, 0], square[:, 1], square[:, 2], color="red", s=40)

    # ajustar limites
    mins = all_pts.min(axis=0) - 0.5
    maxs = all_pts.max(axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rotação em torno de D com translação axial (spiral)")
    ax.legend()

    def update(i):
        th = thetas[i]
        R, t, H = rotation_around_D_with_axis_translation(th)
        transformed = (H @ hsquare).T[:, :3]
        # atualizar pontos móveis
        moving_pts._offsets3d = (
            transformed[:, 0],
            transformed[:, 1],
            transformed[:, 2],
        )
        # atualizar polígono animado
        anim_poly.set_verts([transformed.tolist()])
        return moving_pts, anim_poly

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
