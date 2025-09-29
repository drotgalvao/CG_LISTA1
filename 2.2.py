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


def main():
    np.set_printoptions(precision=6, suppress=True)

    # dados
    u = np.array([1.0, -1.0, 1.0])  # direção do eixo
    P0 = np.array([-1.0, 1.0, 0.0])  # ponto que define a reta
    theta = pi / 6.0  # 30 graus

    # rotação 3x3
    R = rodrigues(u, theta)

    # H_rot = T(P0) * R * T(-P0)  => t_rot = P0 - R*P0
    t_rot = P0 - R.dot(P0)
    H_rot = homogeneous_from_RT(R, t_rot)

    # escala homogênea
    sx, sy, sz = 3.0, -2.0, 0.5
    S = np.diag([sx, sy, sz, 1.0])

    # translação homogênea
    t = np.array([1.0, -2.0, -3.0])
    T = np.eye(4)
    T[:3, 3] = t

    # composição total: aplica rotação em torno da reta, depois escala, depois translação
    H_total = T.dot(S).dot(H_rot)

    print("\nEixos e parâmetros:")
    print("u (direction) =", u)
    print("P0 (point on axis) =", P0)
    print("theta (deg) =", np.degrees(theta))

    print("\nH_rot (rotação de 30° em torno da reta):")
    print(H_rot)

    print("\nS (escala homogênea):")
    print(S)

    print("\nT (translação):")
    print(T)

    print("\nH_total = T * S * H_rot :")
    print(H_total)

    # Verificação em um ponto de teste: aplicar H_total a um ponto q
    q = np.array([0.5, -0.2, 1.0])
    qh = np.append(q, 1.0)
    q_transformed = H_total.dot(qh)[:3]
    print("\nTeste em q =", q)
    print("q transformado =", q_transformed)

    # Também mostrar a sequência explícita: H_after_rot_then_scale = S * H_rot, e H_after_scale_then_trans = T * (S * H_rot)
    HS = S.dot(H_rot)
    print("\nS * H_rot:")
    print(HS)

    print("\nT * (S * H_rot) (de novo, H_total):")
    print(T.dot(HS))

    # ===================================================================
    # 1) rotação suave 0 -> theta (aplica apenas rotação em torno do eixo)
    # 2) escala suave em magnitude 1 -> |s| (mantém sinais até o final)
    #    e aplica o sinal final no término da fase (evita atravessar zero)
    # 3) translação suave 0 -> t
    # Cada fase aplica a composição apropriada sobre a geometria resultante.
    # ===================================================================
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        print("\nMatplotlib não disponível — pulando animação")
        return

    # vértices de um cubo de lado 1 centrado na origem
    cube_verts = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]

    frames_rot = 90
    frames_scale = 60
    frames_trans = 60
    total_frames = frames_rot + frames_scale + frames_trans

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax_lim = 4.0
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Operador afim: Rot(30° sobre reta) → Scale → Translate")

    # linhas para as faces (wireframe)
    lines = [ax.plot([], [], [], color="black")[0] for _ in faces]

    # pré-calcula a rotação final
    R_final = R
    t_rot_final = t_rot
    Hrot_final = H_rot

    def transform_points(H, pts):
        n = pts.shape[0]
        ph = np.hstack([pts, np.ones((n, 1))])
        return (H.dot(ph.T)).T[:, :3]

    def init_anim():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines

    def update(frame):
        # fase 1: rotação
        if frame < frames_rot:
            alpha = frame / max(1, frames_rot - 1)
            th = alpha * theta
            Ralpha = rodrigues(u, th)
            Hf = homogeneous_from_RT(Ralpha, P0 - Ralpha.dot(P0))

        # fase 2: escala (aplicada sobre a geometria já rotacionada)
        elif frame < frames_rot + frames_scale:
            i = frame - frames_rot
            alpha = i / max(1, frames_scale - 1)
            # interpolar magnitude multiplicativamente para evitar atravessar zero
            sx_mag = 1.0 + alpha * (abs(sx) - 1.0)
            sy_mag = 1.0 + alpha * (abs(sy) - 1.0)
            sz_mag = 1.0 + alpha * (abs(sz) - 1.0)
            Sa = np.diag([sx_mag, sy_mag, sz_mag, 1.0])
            # aplicar escala sobre a rotação completa (Hrot_final)
            Hf = Sa.dot(Hrot_final)
            # ao final da fase de escala, aplicar o sinal de cada componente
            if alpha >= 0.9999:
                signs = np.array([np.sign(sx), np.sign(sy), np.sign(sz)])
                S_signed = np.diag(
                    [signs[0] * abs(sx), signs[1] * abs(sy), signs[2] * abs(sz), 1.0]
                )
                Hf = S_signed.dot(Hrot_final)

        # fase 3: translação suave sobre a geometria já rotacionada e escalada
        else:
            i = frame - (frames_rot + frames_scale)
            alpha = i / max(1, frames_trans - 1)
            Ta = np.eye(4)
            Ta[:3, 3] = alpha * t
            # usar H depois de rotação e escala completa (incluindo sinais)
            S_full = np.diag([sx, sy, sz, 1.0])
            Hf = Ta.dot(S_full).dot(Hrot_final)

        verts_t = transform_points(Hf, cube_verts)
        for idx, face in enumerate(faces):
            poly = np.vstack([verts_t[j] for j in face + [face[0]]])
            lines[idx].set_data(poly[:, 0], poly[:, 1])
            lines[idx].set_3d_properties(poly[:, 2])
        return lines

    import matplotlib.animation as animation

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init_anim, interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
