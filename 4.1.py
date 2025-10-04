import numpy as np
from math import pi

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.animation as animation
except Exception as e:
    print("Matplotlib necessário para animação:", e)


def reflection_plane(n, c):
    n = np.asarray(n, dtype=float)
    norm2 = np.dot(n, n)
    A = np.eye(3) - 2.0 * np.outer(n, n) / norm2
    t = 2.0 * c * n / norm2
    return A, t


def homogeneous_from_RT(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def rotation_about_line(point, axis_dir, theta):
    # Rodrigues
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
    return homogeneous_from_RT(R, t)


def eval_plane(n, c, p):
    return float(np.dot(n, p) - c)


def orthonormal_basis_from_axis(u):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    if abs(u[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    v1 = np.cross(u, a)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    v2 = v2 / np.linalg.norm(v2)
    return u, v1, v2


def main():
    np.set_printoptions(precision=6, suppress=True)

    # Planos A, B, C
    nA = np.array([-2.0, 1.0, -1.0])
    cA = 1.0
    nB = np.array([0.0, 1.0, 1.0])
    cB = 1.0

    # plane C from point (0,1,0) and direction vectors v,w -> normal computed
    pC = np.array([0.0, 1.0, 0.0])
    v = np.array([-2.0, 4.0, -2.0])  # (-2,1+3,1-3) -> (-2,4,-2)
    w = np.array([-1.0, -1.0, 1.0])
    nC = np.cross(v, w)
    # normalize the integer vector
    nC = nC / np.gcd.reduce(nC.astype(int)) if np.allclose(nC, nC.astype(int)) else nC
    cC = float(np.dot(nC, pC))

    A_C, t_C = reflection_plane(nC, cC)
    H_C = homogeneous_from_RT(A_C, t_C)

    print("\nPlano C: normal nC =", nC, "cC =", cC)
    print("\nMatriz de reflexão em C (H_C):")
    print(H_C)

    # Eixo D
    pD = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, -1.0, 1.0])
    uD = d / np.linalg.norm(d)

    print("\nEixo D: ponto", pD, "direção (unit) =", uD)

    # parâmetros temporais
    T = 6.0  # período base (s)
    omega = 2 * pi / T  # 1 volta por T
    # axial translation: 2 units per revolution -> axial shift per rad = 2/(2pi) = 1/pi
    axial_per_rad = 1.0 / pi

    print("\nOperador espiral H_spiral(t) = T( shift(t) ) * R_D( phi(t) )")
    print("phi(t) = omega * t; shift(t) = axial_per_rad * phi(t) * uD")

    # Inicial head offset (a pequena distância do eixo para visualizar a hélice)
    radius = 0.5
    u, a, b = orthonormal_basis_from_axis(uD)
    base_point = pD + radius * a  # ponto inicial radial

    # animação: a serpente é uma trajetória de pontos (cabeça + trilha)
    frames = 360
    trail_len = 200

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax_lim = 6.0
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Isneique: serpente espiral com reflexões em C ao cruzar A ou B")

    # line for trail and scatter for head
    (trail_line,) = ax.plot([], [], [], color="green", lw=2)
    (head_point,) = ax.plot([], [], [], marker="o", color="red")

    # draw planes A and B for reference (small squares)
    # Plane drawing helper
    def draw_plane(n, c, size=3.0, color="gray", alpha=0.2):
        # find two orthonormal basis vectors on plane
        n = n / np.linalg.norm(n)
        if abs(n[2]) < 0.9:
            v0 = np.array([0, 0, 1])
        else:
            v0 = np.array([1, 0, 0])
        e1 = np.cross(n, v0)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 = e2 / np.linalg.norm(e2)
        # center point on plane
        center = n * c
        s = np.linspace(-size, size, 2)
        X = []
        Y = []
        Z = []
        for u0 in s:
            for v0 in s:
                p = center + u0 * e1 + v0 * e2
                X.append(p[0])
                Y.append(p[1])
                Z.append(p[2])
        ax.plot_trisurf(X, Y, Z, color=color, alpha=alpha)

    draw_plane(nA, cA, size=3.0, color="blue", alpha=0.15)
    draw_plane(nB, cB, size=3.0, color="orange", alpha=0.15)
    # draw axis D
    pts_axis = np.array([pD + t * uD for t in np.linspace(-6, 6, 10)])
    ax.plot(
        pts_axis[:, 0],
        pts_axis[:, 1],
        pts_axis[:, 2],
        color="black",
        lw=1,
        linestyle="--",
    )

    trail = []
    reflected = False

    # initial sign for crossing detection
    prev_sA = None
    prev_sB = None

    def init():
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        head_point.set_data([], [])
        head_point.set_3d_properties([])
        return trail_line, head_point

    def update(frame):
        nonlocal trail, reflected, prev_sA, prev_sB
        t = (frame / frames) * 3 * T  # simulate several periods so snake climbs
        phi = omega * t
        shift_mag = axial_per_rad * phi
        shift_vec = shift_mag * uD

        # rotation around D by phi
        Hrot = rotation_about_line(pD, uD, phi)
        # axial translation along axis D by shift_vec
        Hshift = homogeneous_from_RT(np.eye(3), shift_vec)
        # total spiral operator
        Hspiral = Hshift.dot(Hrot)

        # apply to base_point to get head
        head_h = Hspiral.dot(np.append(base_point, 1.0))[:3]

        # detect crossing of planes A or B for the head
        sA = eval_plane(nA, cA, head_h)
        sB = eval_plane(nB, cB, head_h)
        crossed = False
        if prev_sA is not None and sA * prev_sA < 0:
            crossed = True
        if prev_sB is not None and sB * prev_sB < 0:
            crossed = True
        prev_sA = sA
        prev_sB = sB

        if crossed:
            # reflect existing trail and head with H_C
            reflected = not reflected
            # apply H_C to all stored points
            if len(trail) > 0:
                pts = np.array(trail)
                ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
                pts_ref = (H_C.dot(ph.T)).T[:, :3]
                trail = [tuple(p) for p in pts_ref]
            # also reflect current head
            head_h = (H_C.dot(np.append(head_h, 1.0)))[:3]

        # append head to trail
        trail.append(tuple(head_h))
        if len(trail) > trail_len:
            trail = trail[-trail_len:]

        trail_arr = np.array(trail)
        trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        trail_line.set_3d_properties(trail_arr[:, 2])
        head_point.set_data([head_h[0]], [head_h[1]])
        head_point.set_3d_properties([head_h[2]])

        return trail_line, head_point

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
