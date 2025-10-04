import numpy as np
from math import pi

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.animation as animation
except Exception as e:
    print("Matplotlib necessário para animação:", e)


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


def homogeneous_from_RT(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def rotation_about_line(point, axis_dir, theta):
    """Matriz homogênea de rotação de ângulo theta em torno da reta definida por (point, axis_dir)."""
    R = rodrigues(axis_dir, theta)
    t = point - R.dot(point)
    return homogeneous_from_RT(R, t)


def spin_operator_about_axis_through_point(point, axis_dir, theta):
    """Retorna H = T(p) * R_axis(theta) * T(-p)"""
    R = rodrigues(axis_dir, theta)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = point - R.dot(point)
    return H


def orthonormal_basis_from_axis(u):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    # pick a vector not parallel to u
    if abs(u[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    v1 = np.cross(u, a)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    v2 = v2 / np.linalg.norm(v2)
    return u, v1, v2


def build_top_geometry(tip, axis_dir, height=0.6, radius=0.2):
    """Constrói vértices do pião (pirâmide com base quadrada) alinhado ao eixo fornecido.
    Retorna array Nx3 com vértices: [tip, base0..3]
    """
    u, a, b = orthonormal_basis_from_axis(axis_dir)
    base_center = tip + u * height
    base = np.array(
        [
            base_center + radius * (a + b),
            base_center + radius * (a - b),
            base_center + radius * (-a - b),
            base_center + radius * (-a + b),
        ]
    )
    verts = np.vstack([tip.reshape(1, 3), base])
    return verts


def main():
    np.set_printoptions(precision=6, suppress=True)

    # parâmetros físicos / temporais
    T = 4.0  # escolhemos t = 4s como período base
    # velocidades angulares
    omega_r = 2 * pi / T  # r gira em torno de s: 1 volta por T
    omega_spin = 4 * 2 * pi / T  # pião gira 4 voltas por T

    # eixo r inicial: passa por p0 com direção v
    p0 = np.array([1.0, 2.0, 0.0])
    v0 = np.array([1.0, -1.0, 0.0])
    u0 = v0 / np.linalg.norm(v0)

    # reta s: linha vertical passando por (2,1,0), direção k=(0,0,1)
    s_point = np.array([2.0, 1.0, 0.0])
    s_dir = np.array([0.0, 0.0, 1.0])

    # top geometry in initial pose: construir o pião "em pé" (eixo vertical global)
    global_up = np.array([0.0, 0.0, 1.0])
    top_verts = build_top_geometry(p0, global_up, height=0.6, radius=0.25)

    # mostrar expressão paramétrica das matrizes
    print(
        "\nParâmetros: T =",
        T,
        "s; omega_r =",
        omega_r,
        "rad/s; omega_spin =",
        omega_spin,
        "rad/s",
    )
    print("\nDefinições (funções de tempo t):")
    print("phi(t) = omega_r * t  -> ângulo de rotação de r ao redor de s")
    print("psi(t) = omega_spin * t -> ângulo de giro do pião em torno de r(t)")

    print("\nMatriz de rotação de r ao redor de s (R_s(phi)):")
    print("R_s(phi) = rotation_about_line(s_point, s_dir, phi)")

    print("\nMatriz do giro do pião em torno de seu eixo r(t):")
    print("H_spin(t) = T(p'(t)) * R_axis(u'(t), psi(t)) * T(-p'(t))")
    print("onde p'(t) = R_s(phi(t)) * p0 e u'(t) = R_s(phi(t)) * u0")

    # preparar animação
    frames = 240
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax_lim = 3.0
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Pião: eixo r girando em torno de s enquanto pião gira em torno de r")

    # linhas para as arestas da pirâmide
    # faces: tip- base edges
    lines = [ax.plot([], [], [], color="C0", lw=2)[0] for _ in range(4)]
    base_loop = ax.plot([], [], [], color="C1", lw=1)[0]

    def transform_points(H, pts):
        n = pts.shape[0]
        ph = np.hstack([pts, np.ones((n, 1))])
        return (H.dot(ph.T)).T[:, :3]

    def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        base_loop.set_data([], [])
        base_loop.set_3d_properties([])
        return lines + [base_loop]

    def update(frame):
        t = (frame / frames) * T  # percorre um período T durante os frames
        phi = omega_r * t
        psi = omega_spin * t

        # rotação de r em torno de s por phi
        H_Rs = rotation_about_line(s_point, s_dir, phi)
        # ponto do bico (tip) após rotação de r em torno de s
        p_prime = (H_Rs.dot(np.append(p0, 1.0)))[:3]

        # queremos que o pião gire "em pé" (eixo vertical global) — aplicar o giro interno
        # em torno do eixo Z que passa pelo bico p_prime
        u_spin = global_up
        H_spin = spin_operator_about_axis_through_point(p_prime, u_spin, psi)

        # combinação: mover o pião por H_Rs (precessão) e então aplicar o giro interno em pé
        H_total = H_spin.dot(H_Rs)

        # transformar vértices do pião
        verts_t = transform_points(H_total, top_verts)

        # desenhar arestas tip->base vertices
        tip = verts_t[0]
        base = verts_t[1:]
        for i in range(4):
            seg = np.vstack([tip, base[i]])
            lines[i].set_data(seg[:, 0], seg[:, 1])
            lines[i].set_3d_properties(seg[:, 2])
        # desenhar loop da base
        base_loop.set_data(
            np.append(base[:, 0], base[0, 0]), np.append(base[:, 1], base[0, 1])
        )
        base_loop.set_3d_properties(np.append(base[:, 2], base[0, 2]))

        return lines + [base_loop]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
