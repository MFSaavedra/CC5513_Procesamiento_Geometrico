"""Figures for chapter 4 (differential geometry) and chapter 5 (Laplace-Beltrami, smoothing)."""
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay
import gp

rng = np.random.default_rng(1)


# =============================================================== chapter 4
def curve_figure():
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)

    def gamma(t):
        r = 1 + 0.35 * np.cos(3 * t)
        return np.c_[r * np.cos(t), 0.8 * r * np.sin(t)]
    X = gamma(t)
    d1 = (np.roll(X, -1, 0) - np.roll(X, 1, 0)) / (2 * (t[1] - t[0]))
    d2 = (np.roll(X, -1, 0) - 2 * X + np.roll(X, 1, 0)) / (t[1] - t[0]) ** 2
    speed = np.linalg.norm(d1, axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    s = np.r_[0, np.cumsum(speed[:-1] * (t[1] - t[0]))]
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    ax.plot(X[:, 0], X[:, 1], "k", lw=1.2)
    for i, col in zip([0, 520, 1000, 1700], ["C0", "C1", "C3", "C2"]):
        T = d1[i] / speed[i]
        Nn = np.array([-T[1], T[0]])
        R = 1 / kappa[i]
        c = X[i] + R * Nn
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(c[0] + abs(R) * np.cos(th), c[1] + abs(R) * np.sin(th), color=col, lw=0.8, ls="--")
        ax.plot(*X[i], "o", color=col, ms=4)
        ax.annotate("", xy=X[i] + 0.35 * T, xytext=X[i], arrowprops=dict(arrowstyle="->", color=col))
        ax.annotate("", xy=X[i] + 0.35 * Nn, xytext=X[i], arrowprops=dict(arrowstyle="->", color=col))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.5, 1.5)
    ax.set_title(r"osculating circles, radius $1/|\kappa|$")
    ax = axs[1]
    ax.plot(s, kappa, "k", lw=1.2, label=r"exact $\kappa(s)$")
    for n, mk in [(24, "o"), (60, "s")]:
        P = gamma(np.linspace(0, 2 * np.pi, n, endpoint=False))
        e0 = P - np.roll(P, 1, 0)
        e1 = np.roll(P, -1, 0) - P
        ang = np.arctan2(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0], (e0 * e1).sum(1))
        L = 0.5 * (np.linalg.norm(e0, axis=1) + np.linalg.norm(e1, axis=1))
        sp_ = np.r_[0, np.cumsum(np.linalg.norm(e1, axis=1))[:-1]]
        ax.plot(sp_, ang / L, mk, ms=3, label=rf"turning angle / length, $n={n}$")
    ax.set_xlabel("arc length $s$"); ax.set_ylabel("signed curvature")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5); ax.grid(alpha=0.3)
    gp.savefig(fig, "ch04_curve.pdf")


def warp_sphere(S, a=0.85):
    """Move vertices towards the north pole (irregular sampling of the same sphere)."""
    z = S[:, 2]
    th = np.arccos(np.clip(z, -1, 1))
    th2 = np.pi * (th / np.pi) ** (1 + a)
    r = np.sin(th2) / np.maximum(np.sin(th), 1e-12)
    out = np.c_[S[:, 0] * r, S[:, 1] * r, np.cos(th2)]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def normals_figure():
    abc = np.array([1.4, 1.0, 0.6])
    rows = []
    for lv in [1, 2, 3, 4, 5]:
        S, F = gp.icosphere(lv)
        S = warp_sphere(S)
        V = S * abc
        Ntrue = V / abc ** 2
        Ntrue /= np.linalg.norm(Ntrue, axis=1, keepdims=True)
        h = gp.edge_lengths(V, F).mean()
        e = {}
        for w in ["uniform", "area", "angle"]:
            Nv = gp.vertex_normals(V, F, w)
            e[w] = np.degrees(np.arccos(np.clip((Nv * Ntrue).sum(1), -1, 1)))
        rows.append((h, e))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for w, mk in [("uniform", "o-"), ("area", "s-"), ("angle", "^-")]:
        ax.loglog([r[0] for r in rows], [np.percentile(r[1][w], 95) for r in rows], mk, ms=4,
                  label=f"{w} weights")
    hh = np.array([r[0] for r in rows])
    ax.loglog(hh, 40 * hh, "k:", lw=0.8, label=r"$O(h)$")
    ax.loglog(hh, 150 * hh ** 2, "k--", lw=0.8, label=r"$O(h^2)$")
    ax.set_xlabel("mean edge length $h$"); ax.set_ylabel("95th pct. angle error [deg]")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    ax.set_title("vertex normals on an irregular ellipsoid")
    gp.savefig(fig, "ch04_normals.pdf")


def torus_curvatures():
    R, r = 1.0, 0.45
    nu, nv = 90, 40
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    v = v + 0.25 * np.sin(v)  # non-uniform sampling in v
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([(R + r * np.cos(W)) * np.cos(U), (R + r * np.cos(W)) * np.sin(U),
                  r * np.sin(W)], -1).reshape(-1, 3)
    F = gp.grid_faces(nu, nv, True, True)
    w = W.ravel()
    K_true = np.cos(w) / (r * (R + r * np.cos(w)))
    H_true = (R + 2 * r * np.cos(w)) / (2 * r * (R + r * np.cos(w)))
    K = gp.angle_defect(V, F)
    H = gp.mean_curvature(V, F)
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 2.2))
    a1 = fig.add_axes([0.0, 0.05, 0.25, 0.85])
    a2 = fig.add_axes([0.25, 0.05, 0.25, 0.85])
    a3 = fig.add_axes([0.58, 0.2, 0.18, 0.65])
    a4 = fig.add_axes([0.82, 0.2, 0.18, 0.65])
    lim = np.abs(K_true).max()
    gp.render(a1, V, F, c=K, cmap="coolwarm", clim=(-lim, lim), view=(40, -60),
              title="$K$ (angle defect)")
    gp.render(a2, V, F, c=H, cmap="viridis", view=(40, -60), title="$H$ (cotangent)")
    a3.plot(K_true, K, ".", ms=1.5)
    a3.plot([K_true.min(), K_true.max()], [K_true.min(), K_true.max()], "k--", lw=0.7)
    a3.set_xlabel("exact $K$"); a3.set_ylabel("discrete $K$")
    a4.plot(H_true, H, ".", ms=1.5, color="C2")
    a4.plot([H_true.min(), H_true.max()], [H_true.min(), H_true.max()], "k--", lw=0.7)
    a4.set_xlabel("exact $H$"); a4.set_ylabel("discrete $H$")
    for a in (a1, a2):
        a.title.set_fontsize(8)
    gp.savefig(fig, "ch04_torus_curvature.png")
    print("Gauss-Bonnet torus:", (K * gp.mass_matrix(V, F).diagonal()).sum())


def model_curvatures():
    V, F = gp.model("bunny", faces=40000)
    C, M = gp.cotan_matrix(V, F), gp.mass_matrix(V, F)
    h = gp.edge_lengths(V, F).mean() ** 2
    Vs = spla.spsolve((M - 2.0 * h * C).tocsc(), M @ V)   # gentle implicit smoothing
    H = gp.mean_curvature(Vs, F)
    K = gp.angle_defect(Vs, F)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.75, 2.3))
    hl = np.percentile(np.abs(H), 97)
    kl = np.percentile(np.abs(K), 95)
    gp.render(axs[0], Vs, F, c=H, cmap="RdBu_r", clim=(-hl, hl), view=(12, -70), title="mean curvature $H$")
    gp.render(axs[1], Vs, F, c=K, cmap="RdBu_r", clim=(-kl, kl), view=(12, -70),
              title="Gaussian curvature $K$")
    gp.colorbar(fig, axs[0], cmap="RdBu_r", clim=(-hl, hl), shrink=0.6)
    gp.colorbar(fig, axs[1], cmap="RdBu_r", clim=(-kl, kl), shrink=0.6)
    gp.savefig(fig, "ch04_model_curvature.png")


def principal_directions(V, F, rings=2):
    """Principal curvatures/directions by local quadric fitting (outward normals)."""
    n = len(V)
    A = gp.adjacency(F, n)
    A2 = A.copy()
    for _ in range(rings - 1):
        A2 = A2 + A2 @ A
    A2 = A2.tocsr()
    N = gp.vertex_normals(V, F)
    k1 = np.zeros(n); k2 = np.zeros(n)
    d1 = np.zeros((n, 3)); d2 = np.zeros((n, 3))
    for i in range(n):
        nb = A2.indices[A2.indptr[i]:A2.indptr[i + 1]]
        nb = nb[nb != i]
        nrm = N[i]
        t1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        D = V[nb] - V[i]
        x, y, z = D @ t1, D @ t2, D @ nrm
        Am = np.c_[x * x, x * y, y * y, x, y]
        coef = np.linalg.lstsq(Am, z, rcond=None)[0]
        a, b, c, d, e = coef
        # shape operator of z = a x^2 + b xy + c y^2 + d x + e y at the origin (gradient correction)
        E, Fq, G = 1 + d * d, d * e, 1 + e * e
        w = np.sqrt(1 + d * d + e * e)
        L, Mq, Nq = 2 * a / w, b / w, 2 * c / w
        I = np.array([[E, Fq], [Fq, G]])
        II = np.array([[L, Mq], [Mq, Nq]])
        Sop = -np.linalg.solve(I, II)          # minus: outward normal, convex => positive
        ev, evec = np.linalg.eig(Sop)
        ev, evec = ev.real, evec.real
        o = np.argsort(-ev)
        k1[i], k2[i] = ev[o]
        for kk, out in zip(o, (d1, d2)):
            vdir = evec[0, kk] * t1 + evec[1, kk] * t2
            out[i] = vdir / np.linalg.norm(vdir)
    return k1, k2, d1, d2


def draw_directions(ax, V, F, D, view, every=1, scale=0.02, color="k", lw=0.5, mask=None):
    R = gp.view_matrix(*view)
    N = gp.vertex_normals(V, F) @ R.T
    vis = N[:, 1] < -0.15
    if mask is not None:
        vis &= mask
    idx = np.nonzero(vis)[0][::every]
    a = (V[idx] - scale * D[idx]) @ R.T
    b = (V[idx] + scale * D[idx]) @ R.T
    segs = np.stack([a[:, [0, 2]], b[:, [0, 2]]], 1)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=5))


def principal_figure():
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.2))
    # ellipsoid
    S, F = gp.icosphere(4)
    V = S * np.array([1.3, 1.0, 0.7])
    k1, k2, d1, d2 = principal_directions(V, F)
    gp.render(axs[0], V, F, c=k1 * k2, cmap="viridis", view=(30, -50), title=r"ellipsoid: $K=\kappa_1\kappa_2$, $\mathbf{t}_1$")
    draw_directions(axs[0], V, F, d1, (30, -50), every=2, scale=0.035, color="w", lw=0.6)
    # torus
    Vt, Ft = gp.torus(1.0, 0.4, 70, 30)
    k1, k2, d1, d2 = principal_directions(Vt, Ft)
    gp.render(axs[1], Vt, Ft, c=k2, cmap="coolwarm", view=(40, -60), title=r"torus: $\kappa_2$ and $\mathbf{t}_1$")
    draw_directions(axs[1], Vt, Ft, d1, (40, -60), every=3, scale=0.05, color="k", lw=0.5)
    # spot
    Vs, Fs = gp.model("spot")
    C, M = gp.cotan_matrix(Vs, Fs), gp.mass_matrix(Vs, Fs)
    h = gp.edge_lengths(Vs, Fs).mean() ** 2
    Vs = spla.spsolve((M - 1.0 * h * C).tocsc(), M @ Vs)
    k1, k2, d1, d2 = principal_directions(Vs, Fs)
    aniso = (k1 - k2) / (np.abs(k1) + np.abs(k2) + 1e-9)
    gp.render(axs[2], Vs, Fs, c=aniso, cmap="Blues", clim=(0, 1.3), view=(15, -40), shade=0.5,
              title=r"spot: anisotropy and $\mathbf{t}_1$")
    draw_directions(axs[2], Vs, Fs, d1, (15, -40), every=1, scale=0.009, color="k", lw=0.45,
                    mask=aniso > 0.4)
    for a in axs:
        a.title.set_fontsize(8)
    gp.savefig(fig, "ch04_principal.png")


# =============================================================== chapter 5
def heat1d():
    n1 = 400
    x = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    dx = x[1] - x[0]
    f = np.sin(3 * x) + 0.3 * np.sin(25 * x)

    def lap(g):
        return (np.roll(g, -1) - 2 * g + np.roll(g, 1)) / dx ** 2
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 1.9))
    axs[0].plot(x, f, "k", lw=1, label="$f$")
    axs[0].plot(x, lap(f) / 400, "C3", lw=0.6, label=r"$\Delta f/400$")
    axs[0].legend(frameon=False, loc="lower left", ncol=2, fontsize=6.5); axs[0].set_xlabel("$x$"); axs[0].set_ylim(-1.9, 1.5)
    axs[0].set_title(r"$\Delta$ amplifies high frequencies ($\times 625$ vs $\times 9$)")
    for tt, c in [(0, "k"), (0.002, "C0"), (0.01, "C1"), (0.05, "C2")]:
        g = f.copy()
        for _ in range(int(round(tt / 1e-5))):
            g = g + 1e-5 * lap(g)
        axs[1].plot(x, g, color=c, lw=1, label=f"$t={tt}$")
    axs[1].legend(frameon=False, ncol=2, loc="upper right"); axs[1].set_xlabel("$x$")
    axs[1].set_title(r"heat equation: modes decay as $e^{-k^2 t}$")
    axs[1].set_ylim(-1.6, 1.9)
    gp.savefig(fig, "ch05_heat1d.pdf")


def flat_mesh(n=400, seed=0):
    r = np.random.default_rng(seed)
    P = r.random((n, 2))
    bnd = np.r_[np.linspace(0, 1, 15, endpoint=False)]
    B = np.vstack([np.c_[bnd, 0 * bnd], np.c_[1 + 0 * bnd, bnd], np.c_[1 - bnd, 1 + 0 * bnd],
                   np.c_[0 * bnd, 1 - bnd]])
    P = np.vstack([B, P[(P.min(1) > 0.03) & (P.max(1) < 0.97)]])
    tri = Delaunay(P)
    V = np.c_[P, np.zeros(len(P))]
    isb = np.zeros(len(P), bool)
    isb[:len(B)] = True
    return V, tri.simplices, isb


def flat_uniform():
    V, F, bnd = flat_mesh()
    Lu = gp.uniform_laplacian(F, len(V))
    C = gp.cotan_matrix(V, F)
    du = Lu @ V
    dc = C @ V
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.85, 2.6))
    for ax, d, t in [(axs[0], du, r"uniform: $\mathbf{L}_u\mathbf{x}\neq 0$"),
                     (axs[1], dc, r"cotangent: $\mathbf{C}\mathbf{x}=0$ (interior)")]:
        ax.triplot(V[:, 0], V[:, 1], F, lw=0.35, color="0.6")
        q = ~bnd
        ax.quiver(V[q, 0], V[q, 1], d[q, 0], d[q, 1], color="C3", angles="xy", scale_units="xy",
                  scale=0.5, width=0.004)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t)
        print(t, np.abs(d[~bnd]).max())
    gp.savefig(fig, "ch05_flat_uniform.pdf")


def noisy_bunny(nf=5000, sigma=0.25, seed=2):
    V, F = gp.model("bunny", faces=nf)
    r = np.random.default_rng(seed)
    N = gp.vertex_normals(V, F)
    h = gp.edge_lengths(V, F).mean()
    Vn = V + sigma * h * r.standard_normal((len(V), 1)) * N
    return V, Vn, F


def spectrum_figure():
    S, F = gp.icosphere(4)
    S = warp_sphere(S)
    C, M = gp.cotan_matrix(S, F), gp.mass_matrix(S, F)
    lam, _ = gp.eigs(-C, M, 64)
    fig, axs = plt.subplots(1, 1, figsize=(3.2, 2.2))
    axs.plot(lam, ".", ms=4)
    for l in range(8):
        axs.axhline(l * (l + 1), color="0.8", lw=0.7, zorder=0)
    axs.set_xlabel("$k$"); axs.set_ylabel(r"$\lambda_k$")
    axs.set_title(r"irregular unit sphere: plateaus at $\ell(\ell+1)$")
    gp.savefig(fig, "ch05_sphere_spectrum.pdf")
    V, F = gp.model("bunny", faces=12000)
    C, M = gp.cotan_matrix(V, F), gp.mass_matrix(V, F)
    lam, Phi = gp.eigs(-C, M, 400)
    ks = [1, 2, 5, 20, 80, 300]
    fig, axs = plt.subplots(1, 6, figsize=(gp.TEXTWIDTH, 1.4))
    for ax, k in zip(axs, ks):
        f = Phi[:, k]
        lim = np.abs(f).max()
        gp.render(ax, V, F, c=f, cmap="RdBu_r", clim=(-lim, lim), view=(12, -70))
        ax.set_title(rf"$\phi_{{{k + 1}}}$, $\lambda={lam[k]:.0f}$", fontsize=8)
    gp.savefig(fig, "ch05_eigenvectors.png")


def lowpass_figure():
    V0, V1, F = noisy_bunny(5000, 0.3)
    C, M = gp.cotan_matrix(V1, F), gp.mass_matrix(V1, F)
    t0 = time.time()
    lam, E = sla.eigh(-C.toarray(), M.toarray())
    print("dense eigh", len(V1), time.time() - t0)
    coef = E.T @ (M @ V1)
    coef0 = E.T @ (M @ V0)
    ms = [10, 40, 150, 600, len(V1)]
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 1.9))
    for j, m in enumerate(ms):
        ax = fig.add_axes([j * 0.14, 0.0, 0.14, 0.95])
        X = E[:, :m] @ coef[:m]
        gp.render(ax, X, F, view=(12, -70), title=f"$m={m}$" if m < len(V1) else "noisy ($m=n$)")
    ax = fig.add_axes([0.74, 0.2, 0.25, 0.7])
    ax.loglog(lam[1:], np.linalg.norm(coef[1:], axis=1), ".", ms=1.5, alpha=0.5, label="noisy")
    ax.loglog(lam[1:], np.linalg.norm(coef0[1:], axis=1), ".", ms=1.5, alpha=0.5, label="clean")
    ax.set_xlabel(r"$\lambda_k$"); ax.set_ylabel(r"$\|\mathbf{c}_k\|$")
    ax.legend(frameon=False, markerscale=4)
    ax.set_title("signal decays, noise is flat", fontsize=8)
    gp.savefig(fig, "ch05_lowpass.png")


def transfer_figure():
    lam_max = 4000.0
    lam_ax = np.linspace(0, lam_max, 2000)
    t = 2e-3
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    ax.plot(lam_ax, (lam_ax < 1 / t).astype(float), "k:", lw=1, label="ideal cut at $1/t$")
    ax.plot(lam_ax, np.exp(-t * lam_ax), "k", lw=1.8, label=r"exact $e^{-t\lambda}$")
    ax.plot(lam_ax, 1 / (1 + t * lam_ax), "C0", label="implicit, 1 step")
    ax.plot(lam_ax, 1 / (1 + t / 10 * lam_ax) ** 10, "C2", label="implicit, 10 steps")
    h = 1.9 / lam_max
    N = int(round(t / h))
    ax.plot(lam_ax, (1 - h * lam_ax) ** N, "C3", lw=0.8, label=rf"explicit, $h=1.9/\lambda_{{\max}}$")
    ax.set_ylim(-0.3, 1.1); ax.set_xlabel(r"$\lambda$"); ax.legend(frameon=False, fontsize=6)
    ax.set_title(f"transfer, $t={t}$")
    ax = axs[1]
    N50 = 50
    ax.semilogy(lam_ax, np.exp(-N50 * h * lam_ax), "k", lw=1.8, label="exact")
    ax.semilogy(lam_ax, np.abs(1 - 1.9 / lam_max * lam_ax) ** N50, "C1", lw=0.9, label=r"$h=1.9/\lambda_{\max}$")
    ax.semilogy(lam_ax, np.abs(1 - 2.1 / lam_max * lam_ax) ** N50, "C3", lw=0.9, label=r"$h=2.1/\lambda_{\max}$")
    ax.axhline(1, color="0.5", lw=0.7)
    ax.set_ylim(1e-12, 1e4); ax.set_xlabel(r"$\lambda$"); ax.legend(frameon=False, fontsize=6)
    ax.set_title(f"explicit, {N50} steps")
    ax = axs[2]
    lam_n = np.linspace(0, 2, 400)          # uniform Laplacian spectrum lies in [0, 2]
    lmb, mu = 0.33, -0.34
    ax.plot(lam_n, (1 - lmb * lam_n) ** 10, "C0", label=r"Laplacian, $\lambda=0.33$")
    ax.plot(lam_n, ((1 - lmb * lam_n) * (1 - mu * lam_n)) ** 10, "C3", label=r"Taubin $\lambda|\mu$")
    ax.axhline(1, color="0.5", lw=0.7)
    ax.set_xlabel(r"eigenvalue of $-\mathbf{L}_u$"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("Laplacian vs Taubin")
    gp.savefig(fig, "ch05_transfer.pdf")


def irregular_sphere():
    S, F = gp.icosphere(4)
    S = warp_sphere(S)
    A0 = gp.face_areas(S, F).sum()
    Lu = gp.uniform_laplacian(F, len(S))
    C, M = gp.cotan_matrix(S, F), gp.mass_matrix(S, F)
    L = sp.diags(1 / M.diagonal()) @ C
    lmax = spla.eigsh(-C, k=1, M=M, which="LM", return_eigenvectors=False)[0]
    hs = 0.9 * 2 / lmax

    def until_half(Lop, h):
        X = S.copy()
        while gp.face_areas(X, F).sum() > 0.5 * A0:
            X = X + h * (Lop @ X)
        return X
    Xu = until_half(Lu, 0.5)
    Xc = until_half(L, hs)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH * 0.85, 2.0))
    for ax, X, t in [(axs[0], S, "irregular sphere"), (axs[1], Xu, "uniform"),
                     (axs[2], Xc, "cotangent")]:
        gp.render(ax, X, F, edges=True, lw=0.1, view=(10, -60), smooth=False,
                  scale=1.05, center=np.zeros(3))
        A = np.c_[2 * X, np.ones(len(X))]
        sol = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)[0]
        c = sol[:3]
        r = np.sqrt(sol[3] + c @ c)
        dev = np.abs(np.linalg.norm(X - c, axis=1) - r).max() / r
        ax.set_title(t + f"\nshift {np.linalg.norm(c):.3f}, dev. {100 * dev:.1f}%", fontsize=7.5)
    gp.savefig(fig, "ch05_irregular_sphere.png")


def implicit_figure():
    V0, V1, F = noisy_bunny(30000, 0.4)
    C, M = gp.cotan_matrix(V1, F), gp.mass_matrix(V1, F)
    l2 = gp.edge_lengths(V1, F).mean() ** 2
    vol0 = gp.signed_volume(V0, F)
    fig, axs = plt.subplots(1, 6, figsize=(gp.TEXTWIDTH, 1.45))
    gp.render(axs[0], V1, F, view=(12, -70), title="noisy")
    for ax, a in zip(axs[1:5], [1, 10, 100, 1000]):
        X = spla.spsolve((M - a * l2 * C).tocsc(), M @ V1)
        gp.render(ax, X, F, view=(12, -70), title=rf"$h={a}\,\bar\ell^2$", scale=0.5, center=np.zeros(3))
        if a == 1000:
            Xbig = X
    m = M.diagonal()
    c = m @ Xbig / m.sum()
    s = (vol0 / gp.signed_volume(Xbig, F)) ** (1 / 3)
    gp.render(axs[5], c + s * (Xbig - c), F, view=(12, -70), title="1000, rescaled", scale=0.55,
              center=np.zeros(3))
    for a in axs:
        a.title.set_fontsize(8)
    gp.savefig(fig, "ch05_implicit.png")


def explicit_vs_implicit_curves():
    """Error curves on a synthetic star-shaped surface with known radius function."""
    S, F = gp.icosphere(5)
    S = warp_sphere(S, 0.6)

    def Rfun(u):
        x, y, z = u.T
        base = 1.0 / np.sqrt((x / 1.25) ** 2 + (y / 1.0) ** 2 + (z / 0.8) ** 2)
        lobes = 1 + 0.12 * np.cos(4 * np.arctan2(y, x)) * (1 - z ** 2)
        r2 = np.random.default_rng(5)
        cen = r2.standard_normal((30, 3)); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        dents = 1 - 0.04 * np.exp(-((u[:, None, :] - cen[None]) ** 2).sum(-1) / 0.02).sum(1)
        return base * lobes * dents
    V0 = S * Rfun(S)[:, None]
    N = gp.vertex_normals(V0, F)
    V1 = V0 + 0.01 * np.random.default_rng(3).standard_normal((len(S), 1)) * N

    def err(X):
        u = X / np.linalg.norm(X, axis=1, keepdims=True)
        return np.median(np.abs(np.linalg.norm(X, axis=1) - Rfun(u)))
    C, M = gp.cotan_matrix(V1, F), gp.mass_matrix(V1, F)
    vol0 = gp.signed_volume(V1, F)
    Lu = gp.uniform_laplacian(F, len(S))
    XA, eA = V1.copy(), [err(V1)]
    for _ in range(40):
        XA = XA + 0.5 * (Lu @ XA); eA.append(err(XA))
    hB = 1e-4
    solve = spla.factorized((M - hB * C).tocsc())
    XB, XC, eB, eC = V1.copy(), V1.copy(), [err(V1)], [err(V1)]

    def step(X):
        B = M @ X
        return np.column_stack([solve(B[:, d]) for d in range(3)])

    def rescale(X):
        m = M.diagonal()
        c = m @ X / m.sum()
        return c + (vol0 / gp.signed_volume(X, F)) ** (1 / 3) * (X - c)
    for _ in range(80):
        XB = step(XB); XC = rescale(step(XC)); eB.append(err(XB)); eC.append(err(XC))
    tB = hB * np.arange(81)
    eD = []
    for tt in tB[1::4]:
        X = spla.spsolve((M - tt * C).tocsc(), M @ V1)
        eD.append(err(X))
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0), gridspec_kw={"width_ratios": [0.8, 1, 1.5], "wspace": 0.45})
    gp.render(axs[0], V1, F, view=(20, -50), title="noisy test surface")
    axs[1].plot(eA, "C1.-", ms=3)
    axs[1].set_xlabel("iterations"); axs[1].set_ylabel("median radial error")
    axs[1].set_title(r"uniform explicit, $h=0.5$", fontsize=8)
    axs[1].axhline(err(V1), color="0.6", ls=":")
    axs[2].plot(tB, eB, "C0", label="implicit, $h=10^{-4}$")
    axs[2].plot(tB, eC, "C2", label="+ volume rescaling")
    axs[2].plot(tB[1::4], eD, "C3--", label="one step of size $t$")
    axs[2].axhline(err(V1), color="0.6", ls=":")
    axs[2].set_xlabel("$t$"); axs[2].legend(frameon=False, fontsize=6.5)
    axs[2].set_title("cotangent Laplacian", fontsize=8)
    gp.savefig(fig, "ch05_error_curves.pdf")


if __name__ == "__main__":
    todo = [curve_figure, normals_figure, torus_curvatures, model_curvatures, principal_figure,
            heat1d, flat_uniform, spectrum_figure, lowpass_figure, transfer_figure,
            irregular_sphere, implicit_figure, explicit_vs_implicit_curves]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
