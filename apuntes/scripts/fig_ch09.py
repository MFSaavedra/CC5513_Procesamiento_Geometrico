"""Figures for chapter 9 (normal estimation and surface reconstruction).

The algorithms are taken verbatim from the course notebook
notebooks/Tema07_Normales_y_Reconstruccion.ipynb (definition cells only).
"""
import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import gp

NB = os.path.join(gp.REPO, "notebooks", "Tema07_Normales_y_Reconstruccion.ipynb")
CELLS = [2, 3, 8, 12, 16, 20, 24, 29, 31, 37, 39, 43, 47, 50, 52]
ns = {}
_nb = json.load(open(NB))
for i in CELLS:
    src = "".join(_nb["cells"][i]["source"])
    src = "\n".join(l for l in src.splitlines() if not l.startswith("print("))
    exec(compile(src, f"nb-cell-{i}", "exec"), ns)
ns["TET_TABLE"] = ns["tet_case_table"]()
globals().update({k: v for k, v in ns.items() if not k.startswith("__")})
plt.rcParams.update({"figure.dpi": 150})

VIEW = (25, -50)
P_gt, N_gt = sample_surface(8000, seed=1)
DIAG = np.linalg.norm(P_gt.max(0) - P_gt.min(0))


def normals_k():
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.04]
    ks = [6, 8, 10, 14, 20, 28, 40, 60, 90, 130]
    E = sweep_k(P_gt, N_gt, sigmas, ks)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.2), gridspec_kw={"width_ratios": [1, 1.3]})
    Pn = add_noise(P_gt, N_gt, 0.01, 7)
    gp.render_points(axs[0], Pn, N=N_gt, view=VIEW, s=0.8, title=r"test surface ($g=1$), $\sigma=0.01$")
    for i, s in enumerate(sigmas):
        axs[1].loglog(ks, E[i], "o-", ms=3, label=rf"$\sigma={s}$")
    kk = np.array(ks, float)
    axs[1].loglog(kk, E[2, 0] * (kk / kk[0]) ** -1.0, "k--", lw=0.8, label=r"$k^{-1}$")
    axs[1].loglog(kk, E[0, -1] * (kk / kk[-1]) ** 0.5, "k:", lw=0.8, label=r"$k^{1/2}$")
    axs[1].set_xlabel("$k$ (neighbours)"); axs[1].set_ylabel("median angle error [deg]")
    axs[1].legend(frameon=False, fontsize=6.5, ncol=2); axs[1].grid(alpha=0.3, which="both")
    gp.savefig(fig, "ch09_normals_k.pdf")
    print("k* per sigma:", [ks[int(np.argmin(r))] for r in E])


def orientation():
    P_n = add_noise(P_gt, N_gt, 0.005, seed=3)
    N_raw, _ = normals_pca(P_n, k=30)
    r = np.random.default_rng(0)
    N_raw = N_raw * r.choice([-1.0, 1.0], len(N_raw))[:, None]
    N_or = orient_normals(P_n, N_raw, k=12)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.75, 2.1))
    for ax, N, t in [(axs[0], N_raw, "random signs"),
                     (axs[1], N_or, "after MST propagation")]:
        ok = (N * N_gt).sum(1) > 0
        print(t, "outward fraction", ok.mean())
        gp.render_points(ax, P_n, c=ok.astype(float), cmap="coolwarm_r", clim=(0, 1), view=VIEW, s=0.8,
                         title=t)
        ax.title.set_fontsize(8)
    gp.savefig(fig, "ch09_orientation.png")


def outliers():
    P_out, is_out = add_outliers(P_gt, 0.06, seed=5)
    keep = statistical_outlier_mask(P_out, k=16, alpha=1.0)
    _, W = normals_pca(P_gt, k=20)
    sv = surface_variation(W)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0))
    gp.render_points(axs[0], P_out, c=is_out.astype(float), cmap="coolwarm", clim=(0, 1), view=VIEW, s=0.7,
                     title="outliers (red)")
    gp.render_points(axs[1], P_out, c=(~keep).astype(float), cmap="coolwarm", clim=(0, 1), view=VIEW, s=0.7,
                     title="flagged by the filter")
    gp.render_points(axs[2], P_gt, c=sv, cmap="viridis", clim=(0, np.percentile(sv, 99)), view=VIEW, s=0.8,
                     title=r"surface variation $\sigma_\lambda$")
    for a in axs:
        a.title.set_fontsize(8)
    gp.savefig(fig, "ch09_outliers.png")


def marching_squares_fig():
    xs2 = np.linspace(-1.6, 1.6, 200)
    XX, YY = np.meshgrid(xs2, xs2, indexing="ij")
    G0 = F(np.c_[XX.ravel(), YY.ravel(), np.zeros(XX.size)]).reshape(XX.shape)
    S0 = marching_squares(G0, xs2, xs2)
    xc = np.linspace(-1.5, 1.5, 9)
    Gc = np.hypot(*np.meshgrid(xc, xc, indexing="ij")) - 1.0
    Sc = marching_squares(Gc, xc, xc)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.7), gridspec_kw={"width_ratios": [1, 1, 0.75, 0.75]})
    lim = np.abs(G0).max()
    axs[0].contourf(XX, YY, G0, levels=21, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[0].add_collection(LineCollection(S0, colors="k", lw=1.0))
    axs[0].set_title("$F$ and $F=0$ on $z=0$", fontsize=7.5)
    X9, Y9 = np.meshgrid(xc, xc, indexing="ij")
    for x in xc:
        axs[1].plot([x, x], [xc[0], xc[-1]], color="0.8", lw=0.5)
        axs[1].plot([xc[0], xc[-1]], [x, x], color="0.8", lw=0.5)
    axs[1].scatter(X9.ravel(), Y9.ravel(), c=np.where(Gc.ravel() < 0, "C0", "w").tolist(), edgecolors="k", s=10, linewidths=0.4, zorder=3)
    axs[1].add_collection(LineCollection(Sc, colors="C3", lw=1.2))
    th = np.linspace(0, 2 * np.pi, 200)
    axs[1].plot(np.cos(th), np.sin(th), "k:", lw=0.6)
    L = np.linalg.norm(Sc[:, 1] - Sc[:, 0], axis=1).sum()
    print("circle 9x9 length", L)
    axs[1].set_title("circle, $9\\times 9$ grid", fontsize=7.5)
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    val = np.array([-1.0, 1.0, -1.0, 1.0])
    X = {e: corners[a] + val[a] / (val[a] - val[b]) * (corners[b] - corners[a]) for e, (a, b) in MS_EDGE.items()}
    for ax, pares, t in [(axs[2], MS_TABLE[5], "case 5: A"), (axs[3], [(0, 1), (2, 3)], "case 5: B")]:
        ax.add_collection(LineCollection([corners[[0, 1, 2, 3, 0]]], colors="0.6", lw=1))
        ax.add_collection(LineCollection([[X[a], X[b]] for a, b in pares], colors="C3", lw=2))
        for cc, v in zip(corners, val):
            ax.plot(*cc, "o", ms=8, color="C0" if v < 0 else "w", mec="k", zorder=3)
        ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.25, 1.25)
        ax.set_title(t, fontsize=7.5)
    for a in axs:
        a.set_aspect("equal"); a.axis("off")
    gp.savefig(fig, "ch09_marching_squares.pdf")


def recon_all(P, N, res=64):
    out = {}
    axes, Qg = make_grid(P_gt, res=res)
    out["Hoppe"] = marching_tets(hoppe_sdf(Qg, P, N).reshape(res, res, res), axes)
    s = np.random.default_rng(0).permutation(len(P))[:700]
    Xr, w, c = rbf_fit(P[s], N[s], eps=0.02)
    out["RBF"] = marching_tets(rbf_eval(Qg, Xr, w, c).reshape(res, res, res), axes)
    ax_p, chi, iso = poisson_reconstruct(P, N, res=res)
    out["Poisson"] = marching_tets(chi, ax_p, iso=iso)
    return out, (ax_p, chi, iso)


def reconstruction():
    P_in = add_noise(P_gt, N_gt, 0.005, seed=3)
    keep = statistical_outlier_mask(P_in, k=16, alpha=2.0)
    P_in = P_in[keep]
    N_in = orient_normals(P_in, normals_pca(P_in, k=30)[0], k=12)
    out, (ax_p, chi, iso) = recon_all(P_in, N_in)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.75))
    res = chi.shape[0]
    sl = chi[:, :, res // 2]
    lim = np.abs(sl).max()
    axs[0].imshow(sl.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[0].contour(sl.T, levels=[iso], colors="k", linewidths=1.0)
    axs[0].set_title(r"$\chi$ on a slice", fontsize=7.5)
    axs[0].axis("off")
    for ax, k in zip(axs[1:], ["Hoppe", "RBF", "Poisson"]):
        V, T = out[k]
        e = dist_to_surface(V) / DIAG
        gp.render(ax, V, T, c=e, cmap="inferno", clim=(0, 0.004), view=VIEW)
        chi_e = mesh_stats(V, T)[3]
        print(k, "median err % diag", 100 * np.median(e), "p95", 100 * np.percentile(e, 95), "chi", chi_e)
        ax.set_title(k, fontsize=7.5)
    gp.savefig(fig, "ch09_reconstruction.png")


def regimes():
    cases = {}
    Pa = add_noise(P_gt, N_gt, 0.02, seed=9)
    Na = orient_normals(Pa, normals_pca(Pa, k=60)[0], k=12)
    cases["noise $\\sigma=0.02$"] = (Pa, Na)
    m = np.linalg.norm(P_gt - np.array([0.9, 0.0, 0.35]), axis=1) > 0.8
    cases["hole"] = (P_gt[m], N_gt[m])
    idx = np.random.default_rng(0).permutation(len(P_gt))[:1200]
    cases["1200 points"] = (P_gt[idx], N_gt[idx])
    fig, axs = plt.subplots(3, 4, figsize=(gp.TEXTWIDTH, 4.3))
    for i, (tag, (Pc, Nc)) in enumerate(cases.items()):
        out, _ = recon_all(Pc, Nc)
        gp.render_points(axs[i, 0], Pc, N=Nc, view=VIEW, s=0.8)
        axs[i, 0].set_title("input: " + tag, fontsize=7.5)
        for j, k in enumerate(["Hoppe", "RBF", "Poisson"]):
            V, T = out[k]
            st = mesh_stats(V, T)
            gp.render(axs[i, j + 1], V, T, view=VIEW)
            axs[i, j + 1].set_title(f"{k} (genus {st[4]})", fontsize=7.5)
    plt.subplots_adjust(hspace=0.15, wspace=0.02)
    gp.savefig(fig, "ch09_regimes.png")


def bunny_screened_poisson():
    import open3d as o3d
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fig_ch06_08 import raycast_scan
    V, F_ = gp.model("bunny", faces=40000)
    P = []
    Nn = []
    for d in [[0, 1, -0.2], [1, 0, -0.2], [0, -1, -0.2], [-1, 0, -0.2], [0, 0.3, 1], [0.2, 0.2, -1]]:
        p, n = raycast_scan(V, F_, d, n=110, noise=0.0015, seed=len(P))
        P.append(p); Nn.append(n)
    P = np.vstack(P); Nn = np.vstack(Nn)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(Nn)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.65))
    gp.render_points(axs[0], P, N=Nn, view=(12, -70), s=0.3, title="6 merged scans")
    for ax, depth in zip(axs[1:], [5, 7, 9]):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
        Vm, Fm = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        gp.render(ax, Vm, Fm, view=(12, -70), title=f"depth {depth}")
    for a in axs:
        a.title.set_fontsize(7.5)
    gp.savefig(fig, "ch09_bunny_poisson.png")


if __name__ == "__main__":
    todo = [normals_k, orientation, outliers, marching_squares_fig, reconstruction, regimes,
            bunny_screened_poisson]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
