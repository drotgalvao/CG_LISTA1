import numpy as np
from math import pi


def rodrigues_R(u, theta):
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


def rotation_about_line_R(point, axis_dir, theta):
    # retorna a matriz 3x3 de rotação efetiva (parte linear) e o ponto rotacionado
    R = rodrigues_R(axis_dir, theta)
    p_rot = (
        rotation_matrix_homogeneous(point, axis_dir, theta).dot(np.append(point, 1.0))
    )[:3]
    return R, p_rot


def rotation_matrix_homogeneous(point, axis_dir, theta):
    u = np.asarray(axis_dir, dtype=float)
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    R = (
        np.cos(theta) * np.eye(3)
        + (1 - np.cos(theta)) * np.outer(u, u)
        + np.sin(theta) * K
    )
    t = point - R.dot(point)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def homogeneous_from_RT(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def demo_simulation():
    # pequena demo que implementa A(t) e B(t) rotacionando em sentidos opostos em torno de D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation

    # Definições iniciais (copiadas do exercício)
    nA0 = np.array([-2.0, 1.0, -1.0])
    cA0 = 1.0
    nB0 = np.array([0.0, 1.0, 1.0])
    cB0 = 1.0

    # encontrar um ponto pA0 no plano A0; resolvemos assumindo z=0 e y=0 para achar x
    # (se degenerar, usar outro procedimento). Aqui encontramos um ponto via least squares.
    def point_on_plane(n, c):
        # solve n·p = c with one coordinate set to zero if possible
        # We'll solve for p = alpha * n + v where v orthogonal to n and choose minimal norm solution
        # simplest: find p by projecting c/norm(n)^2 along n
        return (c / np.dot(n, n)) * n

    pA0 = point_on_plane(nA0, cA0)
    pB0 = point_on_plane(nB0, cB0)

    # eixo D
    pD = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, -1.0, 1.0])
    uD = d / np.linalg.norm(d)

    # parametros
    T = 6.0
    omega = 2 * pi / T

    # base_point para hélice (como antes)
    radius = 0.6
    u = uD
    if abs(u[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    a = np.cross(u, a)
    a = a / np.linalg.norm(a)
    base_point = pD + radius * a

    frames = 300
    trail = []
    trail_len = 180

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    ax.set_title(
        "Demo: planos A e B rotacionando em sentidos opostos; cruzamentos geram reflexão"
    )

    (trail_line,) = ax.plot([], [], [], color="green")
    (head_point,) = ax.plot([], [], [], "ro")

    # precompute H_C (from earlier exercise) for reflection plane C
    pC = np.array([0.0, 1.0, 0.0])
    v = np.array([-2.0, 4.0, -2.0])
    w = np.array([-1.0, -1.0, 1.0])
    nC = np.cross(v, w)
    # normalize integer vector if possible
    if np.allclose(nC, nC.astype(int)):
        from math import gcd

        g = int(abs(np.gcd.reduce(nC.astype(int))))
        if g != 0:
            nC = nC / g
    cC = float(np.dot(nC, pC))
    # reflection matrix
    norm2 = np.dot(nC, nC)
    A_C = np.eye(3) - 2.0 * np.outer(nC, nC) / norm2
    t_C = 2.0 * cC * nC / norm2
    H_C = homogeneous_from_RT(A_C, t_C)

    prev_sA = None
    prev_sB = None

    def init():
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        head_point.set_data([], [])
        head_point.set_3d_properties([])
        return trail_line, head_point

    def update(frame):
        nonlocal trail, prev_sA, prev_sB
        t = (frame / frames) * 3 * T
        phiA = omega * t  # A: sentido H (escolhemos +omega)
        phiB = -omega * t  # B: sentido AH (oposto)

        # compute rotated normals and points on planes
        R_A = rodrigues_R(uD, phiA)
        R_B = rodrigues_R(uD, phiB)
        nA_t = R_A.dot(nA0)
        nB_t = R_B.dot(nB0)
        pA_t = (rotation_matrix_homogeneous(pD, uD, phiA).dot(np.append(pA0, 1.0)))[:3]
        pB_t = (rotation_matrix_homogeneous(pD, uD, phiB).dot(np.append(pB0, 1.0)))[:3]
        cA_t = float(np.dot(nA_t, pA_t))
        cB_t = float(np.dot(nB_t, pB_t))

        # build spiral operator for head position
        phi_spiral = omega * t
        shift_mag = (1.0 / pi) * phi_spiral
        Hrot_spiral = rotation_matrix_homogeneous(pD, uD, phi_spiral)
        Hshift_spiral = homogeneous_from_RT(np.eye(3), shift_mag * uD)
        Hspiral = Hshift_spiral.dot(Hrot_spiral)
        head = Hspiral.dot(np.append(base_point, 1.0))[:3]

        # check crossings using time-varying plane eqs
        sA = float(np.dot(nA_t, head) - cA_t)
        sB = float(np.dot(nB_t, head) - cB_t)
        crossed = False
        if prev_sA is not None and sA * prev_sA < 0:
            crossed = True
        if prev_sB is not None and sB * prev_sB < 0:
            crossed = True
        prev_sA = sA
        prev_sB = sB

        if crossed:
            # reflect trail and head
            if len(trail) > 0:
                pts = np.array(trail)
                ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
                pts_ref = (H_C.dot(ph.T)).T[:, :3]
                trail = [tuple(p) for p in pts_ref]
            head = (H_C.dot(np.append(head, 1.0)))[:3]

        trail.append(tuple(head))
        if len(trail) > trail_len:
            trail = trail[-trail_len:]
        tr = np.array(trail)
        trail_line.set_data(tr[:, 0], tr[:, 1])
        trail_line.set_3d_properties(tr[:, 2])
        head_point.set_data([head[0]], [head[1]])
        head_point.set_3d_properties([head[2]])
        return trail_line, head_point

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    demo_simulation()
