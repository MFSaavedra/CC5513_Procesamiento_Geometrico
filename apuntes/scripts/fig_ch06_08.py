"""Figures for chapters 6-8: acquisition, ICP (local registration), global registration."""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import gp

gp.add_code_path("icp_demo")
gp.add_code_path("global-registration")
rng = np.random.default_rng(4)


# =============================================================== helpers
def rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def raycast_scan(V, F, view_dir, n=220, noise=0.0, seed=0):
    """Orthographic range scan of a mesh along view_dir (points + normals)."""
    import open3d as o3d
    d = np.asarray(view_dir, float)
    d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(up, d); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    c = V.mean(0)
    ext = np.linalg.norm(V.max(0) - V.min(0)) * 0.6
    s = np.linspace(-ext, ext, n)
    S, T = np.meshgrid(s, s)
    O = c - 2 * ext * d + S.reshape(-1, 1) * e1 + T.reshape(-1, 1) * e2
    rays = np.c_[O, np.tile(d, (len(O), 1))].astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    th = ans["t_hit"].numpy()
    ok = np.isfinite(th)
    P = O[ok] + th[ok, None] * d
    N = ans["primitive_normals"].numpy()[ok]
    N = N * np.where((N @ d) > 0, -1, 1)[:, None]          # towards the scanner
    if noise:
        r = np.random.default_rng(seed)
        P = P + noise * r.standard_normal((len(P), 1)) * N
    return P, N


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rot_err(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# =============================================================== chapter 6
def phase_figure():
    f = 1.0
    t = np.linspace(-1.5, 1.5, 1500)
    tau = 0.3
    s = np.cos(2 * np.pi * f * t)
    r = 0.5 * np.cos(2 * np.pi * f * (t - tau)) + 0.3
    x = np.linspace(-1, 1, 400)
    T = np.linspace(-20, 20, 40001)
    C = [np.mean((0.5 * np.cos(2 * np.pi * f * (T - tau)) + 0.3) * np.cos(2 * np.pi * f * (T + xx)))
         for xx in x]
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 1.9))
    axs[0].plot(t, s, "k", lw=1.2, label=r"emitted $s(t)=a\cos(2\pi f t)$")
    axs[0].plot(t, r, "C3", lw=1.2, label=r"received $r(t)=A\cos(2\pi f(t-\tau))+B$")
    axs[0].axvline(0, color="0.6", lw=0.6); axs[0].axvline(tau, color="C3", lw=0.6, ls="--")
    axs[0].annotate("", xy=(tau, 1.05), xytext=(0, 1.05), arrowprops=dict(arrowstyle="<->", lw=0.8))
    axs[0].text(tau + 0.05, 1.0, r"$\tau$", ha="left")
    axs[0].set_ylim(-1.2, 2.3); axs[0].set_xlabel("$t$ (in periods $1/f$)")
    axs[0].legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
    axs[1].plot(x, C, "C0", lw=1.2)
    axs[1].axvline(-tau, color="C3", lw=0.6, ls="--")
    axs[1].set_xlabel("lag $x$"); axs[1].set_ylabel("$C(x)$")
    axs[1].set_title(r"$C(x)=\frac{aA}{2}\cos(2\pi f(x+\tau))$: phase $\varphi=2\pi f\tau$")
    gp.savefig(fig, "ch06_phase.pdf")


def graycode_figure():
    nb = 4
    cols = np.arange(2 ** nb)
    binary = ((cols[None, :] >> np.arange(nb - 1, -1, -1)[:, None]) & 1)
    gray_c = cols ^ (cols >> 1)
    gray = ((gray_c[None, :] >> np.arange(nb - 1, -1, -1)[:, None]) & 1)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 1.5))
    for ax, B, t in [(axs[0], binary, "binary code"), (axs[1], gray, "Gray code")]:
        ax.imshow(np.repeat(B, 6, axis=0), cmap="gray_r", aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(nb) * 6 + 2.5)
        ax.set_yticklabels([f"pattern {k + 1}" for k in range(nb)] if ax is axs[0] else [], fontsize=7)
        ax.set_xticks(cols); ax.set_xticklabels(cols, fontsize=6)
        ax.set_xlabel("projector column (decoded index)")
        ax.set_title(t + rf": $\log_2 {2 ** nb}={nb}$ patterns")
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    gp.savefig(fig, "ch06_graycode.pdf")


def stereo_figure():
    f_px, B = 580.0, 0.075          # Kinect-like focal length [px] and baseline [m]
    Z = np.linspace(0.4, 4.0, 300)
    d = f_px * B / Z
    dZ = Z ** 2 / (f_px * B) * (1 / 8)  # 1/8 pixel disparity resolution
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 1.9))
    axs[0].plot(Z, d, "C0")
    axs[0].set_xlabel("depth $Z$ [m]"); axs[0].set_ylabel("disparity $d$ [px]")
    axs[0].set_title(r"$d = fB/Z$")
    axs[1].plot(Z, dZ * 1000, "C3")
    axs[1].set_xlabel("depth $Z$ [m]"); axs[1].set_ylabel(r"$\Delta Z$ [mm]")
    axs[1].set_title(r"$\Delta Z \approx Z^2\,\Delta d/(fB)$, $\Delta d=1/8$ px")
    for a in axs:
        a.grid(alpha=0.3)
    gp.savefig(fig, "ch06_stereo.pdf")


def scans_figure():
    V, F = gp.model("bunny", faces=40000)
    P1, N1 = raycast_scan(V, F, [0.3, 1.0, -0.15], n=150, noise=0.002, seed=1)
    P2, N2 = raycast_scan(V, F, [-1.0, 0.25, -0.15], n=150, noise=0.002, seed=2)
    view = (12, -70)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 1.9))
    gp.render_points(axs[0], P1, N=N1, view=view, s=0.5, color=(0.15, 0.35, 0.75), title="scan 1")
    gp.render_points(axs[1], P2, N=N2, view=view, s=0.5, color=(0.85, 0.45, 0.1), title="scan 2")
    gp.render_points(axs[2], P1, N=N1, view=view, s=0.4, color=(0.15, 0.35, 0.75))
    gp.render_points(axs[2], P2, N=N2, view=view, s=0.4, color=(0.85, 0.45, 0.1),
                     title="both, in world coordinates")
    gp.savefig(fig, "ch06_scans.png")


# =============================================================== chapter 7
def scan_pair(angle=25, axis=(0.3, 1, 0.2), trans=(0.05, -0.03, 0.04), noise=0.002):
    V, F = gp.model("bunny", faces=40000)
    P1, N1 = raycast_scan(V, F, [0.3, 1.0, -0.15], n=130, noise=noise, seed=1)
    P2, N2 = raycast_scan(V, F, [-0.6, 1.0, -0.1], n=130, noise=noise, seed=2)
    R = rot(axis, angle)
    Tgt = make_T(R, np.array(trans))          # maps source (moved) back to target
    src = apply(np.linalg.inv(Tgt), P2)       # source = scan 2 displaced
    return src, P1, N1, Tgt


def icp_iterations():
    from icp_demo.icp import icp, ICPParams
    src, tgt, Nt, Tgt = scan_pair()
    res = icp(src, tgt, ICPParams(variant="point2point", max_iter=60, auto_reject_factor=3.0),
              target_normals=Nt, T_gt=Tgt)
    hist = res.history
    show = [0, 1, 4, len(hist) - 1]
    view = (12, -70)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.75))
    for ax, k in zip(axs, show):
        h = hist[k]
        S = apply(h.T, src)
        gp.render_points(ax, tgt, view=view, s=0.35, color=(0.15, 0.35, 0.75))
        gp.render_points(ax, S, view=view, s=0.35, color=(0.85, 0.45, 0.1))
        ax.set_title(f"iteration {h.it}\nrot. error {h.rot_err_deg:.1f}°", fontsize=7.5)
    gp.savefig(fig, "ch07_icp_iterations.png")


def icp_convergence():
    from icp_demo.icp import icp, ICPParams
    from icp_demo.scenario import build_scenario
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.1))
    sc = build_scenario("bunny", n_points=6000, angle_deg=20, translation=0.08, noise=0.001,
                        overlap=1.0, seed=3)
    for var, col in [("point2point", "C0"), ("point2plane", "C3")]:
        res = icp(sc.source, sc.target, ICPParams(variant=var, max_iter=50, auto_reject_factor=None),
                  target_normals=sc.target_normals, T_gt=sc.T_gt)
        e = np.array([h.rot_err_deg for h in res.history])
        g = res.gt_curve()
        axs[0].semilogy(np.maximum(g, 1e-6), color=col, label=var.replace("2", "-to-"))
        axs[1].semilogy(np.maximum(e, 1e-4), color=col, label=var.replace("2", "-to-"))
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("RMSE to ground truth")
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("rotation error [deg]")
    for a in axs:
        a.legend(frameon=False); a.grid(alpha=0.3)
    gp.savefig(fig, "ch07_convergence.pdf")


def icp_basin():
    from icp_demo.icp import icp, ICPParams
    from icp_demo.scenario import build_scenario
    angles = [10, 20, 30, 45, 60, 75, 90, 120]
    out = {"point2point": [], "point2plane": []}
    trials = 8
    for a in angles:
        for var in out:
            ok = 0
            for s in range(trials):
                sc = build_scenario("bunny", n_points=2500, angle_deg=a, translation=0.05,
                                    noise=0.001, seed=100 + s)
                res = icp(sc.source, sc.target, ICPParams(variant=var, max_iter=60),
                          target_normals=sc.target_normals, T_gt=sc.T_gt)
                ok += res.history[-1].rot_err_deg < 2.0
            out[var].append(ok / trials)
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    for var, mk in [("point2point", "o-"), ("point2plane", "s-")]:
        ax.plot(angles, out[var], mk, ms=4, label=var.replace("2", "-to-"))
    ax.set_xlabel("initial rotation [deg]"); ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05); ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax.set_title("ICP converges only from a good start")
    gp.savefig(fig, "ch07_basin.pdf")
    print(out)


# =============================================================== chapter 8
def o3d_mesh(V, F):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.compute_vertex_normals()
    return m


def harris_figure():
    from gr.harris3d import harris_response_mesh, select_keypoints
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.2))
    for ax, (nm, view, faces) in zip(axs, [("armadillo", (10, 200), 30000), ("cow", (15, -50), None),
                                         ("fandisk", (25, -40), None)]):
        V, F = gp.model(nm, faces=faces)
        m = o3d_mesh(V, F)
        h = harris_response_mesh(m, delta_frac=0.015)
        diag = np.linalg.norm(V.max(0) - V.min(0))
        kp = select_keypoints(V, h, n_keypoints=60, nms_radius=0.04 * diag)
        lim = np.percentile(np.abs(h), 97)
        gp.render(ax, V, F, c=h, cmap="RdBu_r", clim=(-lim, lim), view=view)
        R = gp.view_matrix(*view)
        N = gp.vertex_normals(V, F) @ R.T
        vis = kp[N[kp, 1] < 0]
        Q = V[vis] @ R.T
        ax.scatter(Q[:, 0], Q[:, 2], s=8, c="gold", edgecolors="k", linewidths=0.4, zorder=6)
        ax.set_title(nm, fontsize=8)
    gp.savefig(fig, "ch08_harris.png")


def spin_figure():
    from gr.spin_images import compute_spin_images, SpinImageParams
    V, F = gp.model("bunny", faces=40000)
    P, N = raycast_scan(V, F, [0.3, 1.0, -0.15], n=220)
    picks = [np.argmin(np.linalg.norm(P - q, axis=1)) for q in
             [np.array([-0.18, 0.0, 0.25]), np.array([0.05, -0.1, 0.0]), np.array([0.2, -0.05, -0.15])]]
    prm = SpinImageParams(bin_size=0.012, image_width=16, support_angle_deg=60)
    S = compute_spin_images(P, N, np.array(picks), prm)
    view = (5, -75)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.6), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    gp.render_points(axs[0], P, N=N, view=view, s=0.3, color=(0.6, 0.62, 0.7))
    R = gp.view_matrix(*view)
    for k, i in enumerate(picks):
        q = P[i] @ R.T
        axs[0].plot(q[0], q[2], "o", color=f"C{k + 1}", ms=5, mec="k", mew=0.4, zorder=6)
        axs[0].text(q[0] + 0.02, q[2] + 0.02, str(k + 1), color=f"C{k + 1}", fontsize=8, zorder=7)
    axs[0].set_title("scan and three points", fontsize=8)
    for k in range(3):
        img = S[k].reshape(16, 16)
        axs[k + 1].imshow(img, cmap="gray_r", interpolation="nearest")
        axs[k + 1].set_title(f"spin image {k + 1}", fontsize=8, color=f"C{k + 1}")
        axs[k + 1].set_xticks([]); axs[k + 1].set_yticks([])
    gp.savefig(fig, "ch08_spin.png")


def fourpcs_invariants():
    r = np.random.default_rng(7)
    a, b = np.array([0.15, 0.85]), np.array([0.8, 0.12])
    c, d = np.array([0.12, 0.3]), np.array([0.92, 0.62])
    # intersection of ab and cd
    A = np.c_[b - a, -(d - c)]
    s, t = np.linalg.solve(A, c - a)
    e = a + s * (b - a)
    r1, r2 = s, t
    Q = r.random((45, 2)) * 1.6
    th = 0.9
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base = np.array([a, b, c, d]) @ Rm.T + np.array([0.75, 0.15])
    Q = np.vstack([Q, base])
    d1, d2 = np.linalg.norm(b - a), np.linalg.norm(d - c)
    delta = 0.02
    D = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    i1, j1 = np.nonzero(np.abs(D - d1) < delta)
    i2, j2 = np.nonzero(np.abs(D - d2) < delta)
    E1 = Q[i1] + r1 * (Q[j1] - Q[i1])
    E2 = Q[i2] + r2 * (Q[j2] - Q[i2])
    tree = cKDTree(E2)
    hits = [(k, l) for k, ls in enumerate(tree.query_ball_point(E1, 0.012)) for l in ls]
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.6))
    ax = axs[0]
    for p, q, col in [(a, b, "C0"), (c, d, "C1")]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=1.5)
    for nm, p in zip("abcd", [a, b, c, d]):
        ax.plot(*p, "ko", ms=5)
        ax.text(p[0] + 0.02, p[1] + 0.02, f"$\\mathbf{{{nm}}}$", fontsize=10)
    ax.plot(*e, "o", color="C3", ms=6)
    ax.text(e[0] + 0.03, e[1] - 0.06, r"$\mathbf{e}$", color="C3", fontsize=10)
    ax.set_title(rf"base $B$: $r_1={r1:.2f}$, $r_2={r2:.2f}$", fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
    ax = axs[1]
    ax.plot(Q[:, 0], Q[:, 1], ".", color="0.5", ms=3)
    ax.plot(E1[:, 0], E1[:, 1], ".", color="C0", ms=1.5, alpha=0.5, label=r"$\mathbf{e}_1=\mathbf{q}_i+r_1(\mathbf{q}_j-\mathbf{q}_i)$, pairs at $d_1$")
    ax.plot(E2[:, 0], E2[:, 1], ".", color="C1", ms=1.5, alpha=0.5, label=r"$\mathbf{e}_2=\mathbf{q}_i+r_2(\mathbf{q}_j-\mathbf{q}_i)$, pairs at $d_2$")
    nq = len(Q) - 4
    for k, l in hits:
        p, q = Q[i1[k]], Q[j1[k]]
        u, v = Q[i2[l]], Q[j2[l]]
        planted = min(i1[k], j1[k], i2[l], j2[l]) >= nq
        lw, al = (1.6, 1.0) if planted else (0.5, 0.35)
        ax.plot([p[0], q[0]], [p[1], q[1]], color="C0", lw=lw, alpha=al)
        ax.plot([u[0], v[0]], [u[1], v[1]], color="C1", lw=lw, alpha=al)
        ax.plot(*E1[k], "o", color="C3", ms=4 if planted else 2.5, alpha=al)
    ax.legend(frameon=False, fontsize=6, loc="upper left", markerscale=4)
    ax.set_title(f"point set $Q$: {len(E1)}+{len(E2)} intermediate points", fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
    gp.savefig(fig, "ch08_4pcs_invariants.pdf")


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dm @ U.T
    return R, cb - R @ ca


def base_width_and_iterations():
    widths = np.array([0.25, 0.5, 1, 2, 4, 8])
    eps = 0.01
    errs = []
    r = np.random.default_rng(0)
    for D in widths:
        e = []
        for _ in range(400):
            X = r.standard_normal((3, 3))
            X = X / np.linalg.norm(X, axis=1, keepdims=True) * r.random((3, 1)) ** (1 / 3) * D / 2
            Rt = rot(r.standard_normal(3), r.uniform(0, 180))
            Y = X @ Rt.T + eps * r.standard_normal((3, 3))
            Re, _ = kabsch(X, Y)
            e.append(rot_err(Re, Rt))
        errs.append(np.median(e))
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.1))
    axs[0].loglog(widths, errs, "o-", ms=4, label="median, measured")
    axs[0].loglog(widths, errs[2] * widths[2] / widths, "k--", lw=0.8, label=r"$\propto 1/D$")
    axs[0].set_xlabel("base diameter $D$"); axs[0].set_ylabel("rotation error [deg]")
    axs[0].set_title(r"wide bases are stable: $\Delta\theta\sim\varepsilon/D$")
    axs[0].legend(frameon=False); axs[0].grid(alpha=0.3, which="both")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    axs[0].xaxis.set_major_locator(FixedLocator(list(widths)))
    axs[0].xaxis.set_minor_locator(NullLocator())
    axs[0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    f = np.linspace(0.2, 0.95, 100)
    for N, mk in [(3, "C0"), (4, "C3")]:
        L = np.log(1 - 0.99) / np.log(1 - f ** N)
        axs[1].semilogy(f, L, color=mk, label=f"$N={N}$")
    axs[1].set_xlabel("overlap fraction $f$"); axs[1].set_ylabel("trials $L$ for $p_s=0.99$")
    axs[1].set_title(r"$L=\log(1-p_s)/\log(1-f^N)$")
    axs[1].legend(frameon=False); axs[1].grid(alpha=0.3, which="both")
    gp.savefig(fig, "ch08_base_iterations.pdf")


def fourpcs_result():
    import open3d as o3d
    from gr.fourpcs import register, FourPCSConfig
    src, tgt, Nt, Tgt = scan_pair(angle=130, axis=(0.2, 0.3, 1.0), trans=(0.3, -0.2, 0.1), noise=0.002)

    def pc(P):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        p.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        return p
    t0 = time.time()
    res = register(pc(src), pc(tgt), FourPCSConfig(delta=0.01, overlap=0.5, n_bases=200,
                                                   variant="super4pcs", seed=1))
    print("4PCS", time.time() - t0, "LCP", res.lcp)
    from icp_demo.icp import icp, ICPParams
    ref = icp(src, tgt, ICPParams(variant="point2plane", max_iter=40), target_normals=Nt,
              T_init=res.T, T_gt=Tgt)
    view = (12, -70)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 1.9))
    for ax, T, t in [(axs[0], np.eye(4), "initial pose"),
                     (axs[1], res.T, f"Super4PCS, LCP={res.lcp:.2f}\nrot. err. {rot_err(res.T[:3, :3], Tgt[:3, :3]):.1f}°"),
                     (axs[2], ref.T, f"+ ICP refinement\nrot. err. {rot_err(ref.T[:3, :3], Tgt[:3, :3]):.2f}°")]:
        allp = np.vstack([tgt, apply(T, src)])
        sc = 0.5 * (allp.max(0) - allp.min(0)).max() * 1.0
        gp.render_points(ax, tgt, view=view, s=0.3, color=(0.15, 0.35, 0.75))
        gp.render_points(ax, apply(T, src), view=view, s=0.3, color=(0.85, 0.45, 0.1))
        ax.set_title(t, fontsize=7.5)
    gp.savefig(fig, "ch08_4pcs_result.png")


# ----------------------------------------------------------------- RANSAC for primitives
def primitive_scene(n=6000, noise=0.004, outliers=0.05, seed=0):
    r = np.random.default_rng(seed)
    n1, n2, n3 = int(0.45 * n), int(0.25 * n), int(0.30 * n)
    Pp = np.c_[r.uniform(-1, 1, n1), r.uniform(-1, 1, n1), np.zeros(n1)]
    u = r.standard_normal((n2, 3)); u[:, 2] = np.abs(u[:, 2]); u /= np.linalg.norm(u, axis=1, keepdims=True)
    Ps = np.array([-0.4, 0.35, 0.3]) + 0.3 * u
    th = r.uniform(0, 2 * np.pi, n3); hz = r.uniform(0, 0.9, n3)
    Pc = np.c_[0.45 + 0.22 * np.cos(th), -0.4 + 0.22 * np.sin(th), hz]
    X = np.vstack([Pp, Ps, Pc])
    lab = np.r_[np.zeros(n1), np.ones(n2), 2 * np.ones(n3)].astype(int)
    X = X + noise * r.standard_normal(X.shape)
    no = int(outliers * len(X))
    X = np.vstack([X, r.uniform([-1, -1, 0], [1, 1, 1], (no, 3))])
    lab = np.r_[lab, -np.ones(no, int)]
    return X, lab


def pca_normals(P, k=20):
    _, idx = cKDTree(P).query(P, k)
    Y = P[idx] - P[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Y, Y)
    w, V = np.linalg.eigh(C)
    return V[:, :, 0]


def fit_plane(p, n):
    nn = np.cross(p[1] - p[0], p[2] - p[0])
    if np.linalg.norm(nn) < 1e-9:
        return None
    nn /= np.linalg.norm(nn)
    return ("plane", p[0], nn)


def fit_sphere(p, n):
    A = 2 * (p[1:] - p[0])
    b = (p[1:] ** 2).sum(1) - (p[0] ** 2).sum()
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    c = np.linalg.solve(A, b)
    return ("sphere", c, np.linalg.norm(p[0] - c))


def fit_cylinder(p, n):
    a = np.cross(n[0], n[1])
    if np.linalg.norm(a) < 1e-6:
        return None
    a /= np.linalg.norm(a)
    Pm = np.eye(3) - np.outer(a, a)
    q0, q1 = Pm @ p[0], Pm @ p[1]
    m0, m1 = Pm @ n[0], Pm @ n[1]
    A = np.c_[m0, -m1]
    st = np.linalg.lstsq(A, q1 - q0, rcond=None)[0]
    c = q0 + st[0] * m0
    return ("cylinder", c, a, np.linalg.norm(q0 - c))


def residual(model, X, N):
    if model[0] == "plane":
        _, p0, nn = model
        d = np.abs((X - p0) @ nn)
        nt = np.tile(nn, (len(X), 1))
    elif model[0] == "sphere":
        _, c, rr = model
        v = X - c
        dist = np.linalg.norm(v, axis=1)
        d = np.abs(dist - rr)
        nt = v / np.maximum(dist[:, None], 1e-12)
    else:
        _, c, a, rr = model
        v = X - c
        v = v - np.outer(v @ a, a)
        dist = np.linalg.norm(v, axis=1)
        d = np.abs(dist - rr)
        nt = v / np.maximum(dist[:, None], 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs((nt * N).sum(1)), 0, 1)))
    return d, ang


def ransac_extract(X, N, eps=0.012, alpha=20, iters=3000, min_in=300, seed=0):
    r = np.random.default_rng(seed)
    labels = -np.ones(len(X), int)
    remaining = np.ones(len(X), bool)
    found = []
    kinds = [("plane", 3, fit_plane), ("sphere", 4, fit_sphere), ("cylinder", 2, fit_cylinder)]
    for _ in range(5):
        idx = np.nonzero(remaining)[0]
        best, best_in = None, None
        for kind, k, fit in kinds:
            for _ in range(iters):
                s = r.choice(idx, k, replace=False)
                m = fit(X[s], N[s])
                if m is None:
                    continue
                if m[0] in ("sphere", "cylinder") and m[-1] > 1.0:
                    continue
                d, ang = residual(m, X[idx], N[idx])
                inl = idx[(d < eps) & (ang < alpha)]
                if best_in is None or len(inl) > len(best_in):
                    best, best_in = m, inl
        if best_in is None or len(best_in) < min_in:
            break
        labels[best_in] = len(found)
        remaining[best_in] = False
        found.append(best)
    return labels, found


def ransac_figure():
    X, lab = primitive_scene()
    N = pca_normals(X)
    t0 = time.time()
    labels, found = ransac_extract(X, N, iters=600)
    print("RANSAC", time.time() - t0, [m[0] for m in found])
    view = (30, -55)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.8, 2.3))
    gp.render_points(axs[0], X, view=view, s=0.6, color=(0.3, 0.3, 0.35), title="input (6000 pts, 5% outliers)")
    cols = np.array([[0.15, 0.4, 0.8], [0.85, 0.3, 0.2], [0.2, 0.65, 0.3], [0.6, 0.3, 0.7], [0.9, 0.7, 0.1]])
    c = np.where(labels[:, None] >= 0, cols[np.maximum(labels, 0) % 5], np.array([[0.75, 0.75, 0.75]]))
    Q = X @ gp.view_matrix(*view).T
    o = np.argsort(-Q[:, 1])
    axs[1].scatter(Q[o, 0], Q[o, 2], c=c[o], s=0.6, linewidths=0, rasterized=True)
    axs[1].set_aspect("equal"); axs[1].axis("off")
    names = ", ".join(m[0] for m in found)
    axs[1].set_title(f"detected: {names}", fontsize=8)
    gp.savefig(fig, "ch08_ransac.png")


if __name__ == "__main__":
    todo = [phase_figure, graycode_figure, stereo_figure, scans_figure, icp_iterations,
            icp_convergence, icp_basin, harris_figure, spin_figure, fourpcs_invariants,
            base_width_and_iterations, fourpcs_result, ransac_figure]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
