import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def reflection_about_plane(P0, u, v):
    """Retorna (R, t, H) onde x' = R x + t é a reflexão em relação ao plano
    definido por P0 + q u + p v.
    R é 3x3, t é 3, H é 4x4 homogênea.
    """
    # normal do plano
    n = np.cross(u, v)
    norm_n2 = np.dot(n, n)
    if norm_n2 == 0:
        raise ValueError("Vetores u e v são colineares; não definem um plano")
    R = np.eye(3) - 2.0 * np.outer(n, n) / norm_n2
    t = P0 - R.dot(P0)
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


def main(no_show=False):
    # dados do enunciado
    P0 = np.array([0.0, 1.0, 0.0])
    u = np.array([-2.0, 4.0, -2.0])
    v = np.array([-1.0, -1.0, 1.0])

    R, t, H = reflection_about_plane(P0, u, v)
    # normal do plano (usada para desenho)
    n = np.cross(u, v)

    print("Matriz R (3x3) da reflexão:")
    np.set_printoptions(precision=6, suppress=True)
    print(R)
    print("\nVetor t (translação):")
    print(t)
    print("\nMatriz homogênea H (4x4):")
    print(H)

    # ponto teste no plano: P0 + a*u + b*v
    a, b = 0.2, -0.4
    pt_on_plane = P0 + a * u + b * v
    reflected = R.dot(pt_on_plane) + t
    print("\nPonto no plano (de teste):", pt_on_plane)
    print("Reflexão (deve ser igual):", reflected)

    # preparar figura: um quadrado de lado 1 não coincidente com o plano
    square = make_unit_square_centered_at(1.0, 0.0, z=0.0, side=1.0)
    hsquare = to_homogeneous(square)

    frames = 48

    # pré-calcula transformações para ajustar limites (projeção XY)
    all_xy = []
    for i in range(frames):
        tparam = i / (frames - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * tparam)  # ease-in-out
        pts = (1 - s) * square + s * (R.dot(square.T).T + t)
        all_xy.append(pts[:, :2])
    all_xy = np.vstack(all_xy)

    min_x, min_y = all_xy[:, 0].min(), all_xy[:, 1].min()
    max_x, max_y = all_xy[:, 0].max(), all_xy[:, 1].max()
    pad = 0.5

    if no_show:
        print(
            f"\nModo headless: pré-computadas {frames} frames (não será exibido plot)"
        )
        # imprimir alguns pontos de amostra do primeiro e último frame
        print("Primeiro frame (XY):")
        print(all_xy[:4])
        print("Último frame (XY):")
        print(all_xy[-4:])
        return

    # criar animação 3D onde cada vértice se move ao longo do segmento perpendicular
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # calcular vértices refletidos
    square_ref = (R.dot(square.T)).T + t

    # segmentos (orig -> ref) e pontos médios
    segs = [(square[i], square_ref[i]) for i in range(len(square))]
    mids = [0.5 * (a + b) for a, b in segs]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # desenhar o plano (malha)
    u_dir = u / np.linalg.norm(u)
    v_dir = v / np.linalg.norm(v)
    U = np.linspace(-1.5, 1.5, 10)
    V = np.linspace(-1.5, 1.5, 10)
    plane_pts = np.array([[P0 + uu * u_dir + vv * v_dir for uu in U] for vv in V])
    X = plane_pts[:, :, 0]
    Y = plane_pts[:, :, 1]
    Z = plane_pts[:, :, 2]
    ax.plot_surface(X, Y, Z, color=(0.9, 0.9, 0.9), alpha=0.5, linewidth=0)

    # desenhar quadrado original e quadrado refletido (estáticos, com transparência)
    orig_poly = Poly3DCollection(
        [square], facecolors=(0.2, 0.6, 0.2, 0.3), edgecolors="k"
    )
    ref_poly = Poly3DCollection(
        [square_ref], facecolors=(0.2, 0.2, 0.8, 0.3), edgecolors="k"
    )
    ax.add_collection3d(orig_poly)
    ax.add_collection3d(ref_poly)

    # desenhar linhas conectando vértices orig->ref e pontos médios
    for (a, b), m in zip(segs, mids):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color="gray",
            linestyle="--",
            alpha=0.7,
        )
        ax.scatter([m[0]], [m[1]], [m[2]], color="red", s=20)

    # pontos que serão animados (começam na posição original)
    moving_pts = ax.scatter(
        square[:, 0], square[:, 1], square[:, 2], color="green", s=50
    )

    # uma coleção para o polígono animado (fronteira)
    anim_poly = Poly3DCollection(
        [square], facecolors=(0.2, 0.4, 0.8, 0.6), edgecolors="k"
    )
    ax.add_collection3d(anim_poly)

    # desenhar P0 e normal
    n_unit = n / np.linalg.norm(n)
    ax.scatter([P0[0]], [P0[1]], [P0[2]], color="black", s=40)
    ax.quiver(
        P0[0],
        P0[1],
        P0[2],
        n_unit[0] * 0.6,
        n_unit[1] * 0.6,
        n_unit[2] * 0.6,
        color="black",
    )

    # ajustar limites 3D
    all_pts = np.vstack([square, square_ref, plane_pts.reshape(-1, 3)])
    mins = all_pts.min(axis=0) - 0.5
    maxs = all_pts.max(axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reflexão: movimento ao longo das normais (orig → imagem)")

    def update(frame):
        tparam = frame / (frames - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * tparam)
        pts_now = (1 - s) * square + s * square_ref
        # atualizar pontos móveis
        moving_pts._offsets3d = (pts_now[:, 0], pts_now[:, 1], pts_now[:, 2])
        # atualizar polígono animado
        anim_poly.set_verts([pts_now.tolist()])
        return (moving_pts, anim_poly)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
