import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi


def rodrigues(u, theta):
    """Rodrigues' rotation formula."""
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


def to_homogeneous(pts):
    pts = np.asarray(pts)
    n = pts.shape[0]
    h = np.ones((n, 4), dtype=float)
    h[:, :3] = pts
    return h.T


def from_homogeneous(h):
    h = np.asarray(h)
    return (h[:3, :] / h[3, :]).T


def make_plane_mesh(C, e1, e2, span=1.5, n=10):
    U = np.linspace(-span, span, n)
    V = np.linspace(-span, span, n)
    pts = np.array([[C + uu * e1 + vv * e2 for uu in U] for vv in V])
    return pts


def main():
    np.set_printoptions(precision=6, suppress=True)

    # dados do enunciado
    A = np.array([2.0, -2.0, -3.0])
    B = np.array([2.0, 1.0, 0.0])
    C = np.array([0.0, -1.0, -1.0])

    # vetores relativos ao centro
    CA = A - C
    CB = B - C

    # ângulo entre CA e CB
    dot = CA.dot(CB)
    norm_prod = np.linalg.norm(CA) * np.linalg.norm(CB)
    cos_angle = dot / norm_prod
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    print(f"Ângulo A-C-B (rad) = {angle:.6f}, deg = {np.degrees(angle):.6f}")

    # eixo de rotação
    n = np.cross(CA, CB)
    if np.linalg.norm(n) == 0:
        raise ValueError("CA e CB são colineares; não formam arco")
    u = n / np.linalg.norm(n)

    # rotação elementar de 30 graus
    theta30 = pi / 6.0
    R30 = rodrigues(u, theta30)

    # construir H30 = T(C) * R * T(-C)
    H30 = np.eye(4)
    H30[:3, :3] = R30
    H30[:3, 3] = C - R30.dot(C)

    # Htotal = H30^k onde k = angle / 30deg (deve ser inteiro aqui = 3)
    k = int(round(angle / theta30))
    Htotal = np.linalg.matrix_power(H30, k)

    print("\nH30 (4x4):")
    print(H30)
    print(f"k = {k} (número de arcos de 30°)")
    print("\nHtotal = H30^k (4x4):")
    print(Htotal)

    # verificação: aplicar Htotal em A -> deve dar B
    Ah = to_homogeneous(np.array([A]))
    Bh = Htotal.dot(Ah)
    Bcalc = from_homogeneous(Bh)[0]
    print("\nA (original):", A)
    print("B (esperado):", B)
    print("B (calculado):", Bcalc)
    print("Erro (Bcalc - B):", Bcalc - B)

    # Preparar animação: calcular arco paramétrico rotacionando CA em torno de u
    frames = 90  # suavidade
    thetas = np.linspace(0.0, angle, frames)
    arc_pts = np.array([C + rodrigues(u, th).dot(CA) for th in thetas])

    # plot 3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # desenhar plano do círculo usando e1,e2
    e1 = CA / np.linalg.norm(CA)
    e2 = np.cross(u, e1)
    e2 = e2 / np.linalg.norm(e2)
    plane_pts = make_plane_mesh(C, e1, e2, span=np.linalg.norm(CA) * 1.2, n=12)
    X = plane_pts[:, :, 0]
    Y = plane_pts[:, :, 1]
    Z = plane_pts[:, :, 2]
    ax.plot_surface(X, Y, Z, color=(0.9, 0.9, 0.9), alpha=0.4, linewidth=0)

    # desenhar arco alvo e pontos A,B,C
    ax.plot(
        arc_pts[:, 0],
        arc_pts[:, 1],
        arc_pts[:, 2],
        color="orange",
        linewidth=2,
        label="arco",
    )
    ax.scatter([A[0]], [A[1]], [A[2]], color="green", s=50, label="A")
    ax.scatter([B[0]], [B[1]], [B[2]], color="blue", s=50, label="B")
    ax.scatter([C[0]], [C[1]], [C[2]], color="red", s=30, label="C")

    # ponto animado e trilha
    (point_anim,) = ax.plot(
        [A[0]], [A[1]], [A[2]], marker="o", color="black", markersize=8
    )
    (trail,) = ax.plot([], [], [], color="black", linewidth=1)

    # ajustar limites
    all_pts = np.vstack([arc_pts, A, B, C])
    mins = all_pts.min(axis=0) - 0.5
    maxs = all_pts.max(axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Partícula: A → B por arco centrado em C (rotação afim)")
    ax.legend()

    def init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        point_anim.set_data([A[0]], [A[1]])
        point_anim.set_3d_properties([A[2]])
        return point_anim, trail

    def update(i):
        p = arc_pts[i]
        point_anim.set_data([p[0]], [p[1]])
        point_anim.set_3d_properties([p[2]])
        tracked = arc_pts[: i + 1]
        trail.set_data(tracked[:, 0], tracked[:, 1])
        trail.set_3d_properties(tracked[:, 2])
        return point_anim, trail

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=40, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
