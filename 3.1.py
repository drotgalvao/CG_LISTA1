import numpy as np
from math import pi


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


def reflection_plane(n, c):
    """Retorna a parte linear A e a translação t da reflexão no plano n.x = c.
    A = I - 2 (n n^T) / ||n||^2
    t = 2 c n / ||n||^2
    """
    n = np.asarray(n, dtype=float)
    norm2 = np.dot(n, n)
    if norm2 == 0:
        raise ValueError("vetor normal n não pode ser zero")
    A = np.eye(3) - 2.0 * np.outer(n, n) / norm2
    t = 2.0 * c * n / norm2
    return A, t


def apply_homogeneous(H, pts):
    """Aplica matriz homogênea H a um conjunto de pontos Nx3 e retorna Nx3."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    ph = np.hstack([pts, np.ones((n, 1))])
    transformed = (H.dot(ph.T)).T[:, :3]
    return transformed


def main():
    np.set_printoptions(precision=6, suppress=True)

    # 1) Reflexão no plano x - y = 1
    n = np.array([1.0, -1.0, 0.0])
    c = 1.0
    A_ref, t_ref = reflection_plane(n, c)
    H_ref = homogeneous_from_RT(A_ref, t_ref)

    print("\nReflexão no plano x - y = 1:")
    print("A (linear):\n", A_ref)
    print("t (translação):", t_ref)
    print("H_ref (homogênea):\n", H_ref)

    # Verificação: ponto no plano x-y=1 deve ser fixo
    p_on_plane = np.array([1.0, 0.0, 0.0])  # 1 - 0 = 1
    p_reflected = apply_homogeneous(H_ref, p_on_plane.reshape(1, 3))[0]
    print("\nPonto no plano (1,0,0) após reflexão ->", p_reflected)

    # 2) Rotação anti-horária de 30° em torno da reta (t,0,-t) com direção (1,0,-1)
    u = np.array([1.0, 0.0, -1.0])
    theta = pi / 6.0  # 30 graus
    R_axis = rodrigues(u, theta)
    H_rot = homogeneous_from_RT(R_axis, np.zeros(3))

    print("\nRotação 30° em torno do eixo direção (1,0,-1):")
    print(R_axis)
    print("H_rot (homogênea):\n", H_rot)

    # 3) Composição: aplicar reflexão e depois rotação (rot ∘ ref)
    H_total = H_rot.dot(H_ref)

    print("\nMatriz composta H_total = H_rot * H_ref:")
    print(H_total)

    # 4) Demonstração: aplicar a um quadrado de lado 1 (vértices no plano z=0)
    square = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    square_after_ref = apply_homogeneous(H_ref, square)
    square_after_total = apply_homogeneous(H_total, square)

    print("\nQuadrado original:")
    print(square)
    print("\nQuadrado após reflexão:")
    print(square_after_ref)
    print("\nQuadrado após reflexão + rotação:")
    print(square_after_total)

    # 5) Animação: interpolação suave para reflexão e depois rotação
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.animation as animation
    except Exception:
        print("\nMatplotlib não disponível — não será possível mostrar a animação.")
        return

    # função utilitária para aplicar H a pontos
    def transform_points(H, pts):
        n = pts.shape[0]
        ph = np.hstack([pts, np.ones((n, 1))])
        return (H.dot(ph.T)).T[:, :3]

    # parametros da animação
    frames_ref = 30
    frames_rot = 120
    total_frames = frames_ref + frames_rot

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax_lim = 3.0
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reflexão (interp.) então Rotação 30° em torno do eixo (1,0,-1)")

    # linhas do quadrado (4 arestas)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    line_objs = [ax.plot([], [], [], color="C0", lw=2)[0] for _ in edges]

    def init():
        for ln in line_objs:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return line_objs

    def update(frame):
        if frame < frames_ref:
            # interpolar linearmente entre identidade e reflexão homogênea
            alpha = frame / max(1, frames_ref - 1)
            A_interp = (1 - alpha) * np.eye(3) + alpha * A_ref
            t_interp = alpha * t_ref
            Hf = homogeneous_from_RT(A_interp, t_interp)
        else:
            # rotação suave após reflexão completa
            i = frame - frames_ref
            beta = i / max(1, frames_rot - 1)
            th = beta * theta
            Ralpha = rodrigues(u, th)
            Hrot_alpha = homogeneous_from_RT(Ralpha, np.zeros(3))
            Hf = Hrot_alpha.dot(H_ref)

        pts_t = transform_points(Hf, square)
        for idx, (a, b) in enumerate(edges):
            seg = np.vstack([pts_t[a], pts_t[b]])
            line_objs[idx].set_data(seg[:, 0], seg[:, 1])
            line_objs[idx].set_3d_properties(seg[:, 2])
        return line_objs

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
