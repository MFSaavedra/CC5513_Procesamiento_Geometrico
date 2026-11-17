"""Figures for chapters 14-16: shape analysis, spectral shape analysis, functional maps.

The TOSCA cats and horses in modelos/ share vertex ordering across poses, so the vertex
index gives the ground-truth correspondence between poses.
"""
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import gp

CACHE = "/tmp/cc5513_spectra"
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(16)


def raw(name):
    """TOSCA mesh at full resolution (vertex order preserved), unit-area scaling."""
    V, F = gp.load_mesh(os.path.join(gp.MODELS, name + ".off"))
    if name.startswith("elephant"):
        V = gp.zup(V)
    V = V - V.mean(0)
    V = V / np.sqrt(gp.face_areas(V, F).sum())
    return V, F


def spectrum(name, k=200, V=None, F=None):
    """First k eigenpairs of L phi = lambda M phi (unit-area mesh), cached on disk."""
    if V is None:
        for kk in (k, 200, 300):
            path = os.path.join(CACHE, f"{name}_{kk}.npz")
            if kk >= k and os.path.exists(path):
                d = np.load(path)
                return d["lam"][:k], d["Phi"][:, :k], d["m"]
        V, F = raw(name)
    L, M = gp.laplacian(V, F), gp.mass_matrix(V, F)
    lam, Phi = gp.eigs(L, M, k)
    if name:
        np.savez(os.path.join(CACHE, f"{name}_{k}.npz"), lam=lam, Phi=Phi, m=M.diagonal())
    return lam, Phi, M.diagonal()


def nn_search(X, Q, chunk=1500):
    """Nearest row of X for every row of Q (KD-tree in low dimension, BLAS brute force otherwise)."""
    if X.shape[1] <= 12:
        return cKDTree(X).query(Q)[1]
    x2 = (X ** 2).sum(1)
    out = np.empty(len(Q), int)
    for i in range(0, len(Q), chunk):
        D = x2[None, :] - 2 * Q[i:i + chunk] @ X.T
        out[i:i + chunk] = D.argmin(1)
    return out


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


VIEW_CAT = (12, -50)


def tight(ax, V, view, pad=0.03):
    """Crop a rendered panel to the projected extent of the shape (keeps equal aspect)."""
    P = V @ gp.view_matrix(*view).T
    lo, hi = P[:, [0, 2]].min(0), P[:, [0, 2]].max(0)
    d = pad * (hi - lo).max()
    ax.set_xlim(lo[0] - d, hi[0] + d)
    ax.set_ylim(lo[1] - d, hi[1] + d)


# =============================================================== chapter 14
def depth_images(V, F, res=64):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    ext = np.abs(V).max() * 1.05
    s = np.linspace(-ext, ext, res)
    A, B = np.meshgrid(s, s)
    imgs = []
    for ax in range(3):
        for sgn in (1, -1):
            d = np.zeros(3); d[ax] = sgn
            o_ax = [i for i in range(3) if i != ax]
            O = np.zeros((res * res, 3))
            O[:, o_ax[0]] = A.ravel(); O[:, o_ax[1]] = B.ravel(); O[:, ax] = -sgn * 2 * ext
            rays = np.c_[O, np.tile(d, (len(O), 1))].astype(np.float32)
            t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            depth = np.where(np.isfinite(t), 1 - (t - ext) / (2 * ext), 0.0)
            imgs.append(depth.reshape(res, res))
    return imgs


def pca_normalize(V, F):
    a = gp.face_areas(V, F)
    c = (V[F].mean(1) * a[:, None]).sum(0) / a.sum()
    X = V - c
    Cov = (X[F].mean(1) * a[:, None]).T @ X[F].mean(1) / a.sum()
    w, U = np.linalg.eigh(Cov)
    U = U[:, ::-1]
    Y = X @ U
    # sign disambiguation by the third moment
    sk = np.sign((Y ** 3).mean(0))
    sk[sk == 0] = 1
    Y = Y * sk
    return Y / np.sqrt(w[::-1][0]) / 3, U


def load_plane(name):
    V, F = gp.load_mesh(os.path.join(gp.MODELS, name + ".off"))
    if len(F) > 40000:
        V, F = gp.decimate(V, F, 30000)
    return gp.normalize(V), F


def pose_normalization_figure():
    V, F = load_plane("airplane_0629")
    Vr = V @ rot([0.3, 0.8, 0.5], 65).T
    Y, U = pca_normalize(Vr, F)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.7, 2.1))
    gp.render(axs[0], Vr, F, smooth=False, cull=False, view=(20, -60), title="arbitrary pose")
    c = Vr.mean(0)
    R = gp.view_matrix(20, -60)
    for k, col in enumerate(["C3", "C2", "C0"]):
        p0 = c @ R.T
        p1 = (c + 0.35 * U[:, k] * (1 - 0.25 * k)) @ R.T
        axs[0].annotate("", xy=p1[[0, 2]], xytext=p0[[0, 2]], arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    gp.render(axs[1], Y, F, smooth=False, cull=False, view=(20, -60), title="after PCA")
    gp.savefig(fig, "ch14_pose_normalization.png")
    imgs = depth_images(Y, F, 64)
    fig, axs = plt.subplots(1, 7, figsize=(gp.TEXTWIDTH, 1.2))
    labels = ["$+x$", "$-x$", "$+y$", "$-y$", "$+z$", "$-z$"]
    for ax, im, lab in zip(axs[:6], imgs, labels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lab, fontsize=8); ax.axis("off")
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(imgs[4])))
    axs[6].imshow(np.log1p(Fm), cmap="magma")
    axs[6].set_title(r"$\log|\mathcal{F}|$ of $+z$", fontsize=8); axs[6].axis("off")
    gp.savefig(fig, "ch14_depth_buffer.png")


def panorama_image(V, F, nth=128, nz=64):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    R = np.hypot(V[:, 0], V[:, 1]).max() * 1.05
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    zs = np.linspace(V[:, 2].min(), V[:, 2].max(), nz)
    TH, Z = np.meshgrid(th, zs)
    O = np.c_[R * np.cos(TH.ravel()), R * np.sin(TH.ravel()), Z.ravel()]
    d = -np.c_[np.cos(TH.ravel()), np.sin(TH.ravel()), np.zeros(TH.size)]
    t = scene.cast_rays(o3d.core.Tensor(np.c_[O, d].astype(np.float32)))["t_hit"].numpy()
    r = np.where(np.isfinite(t) & (t < R), (R - t) / R, 0.0)
    return r.reshape(nz, nth)


def panorama_figure():
    V, F = raw("horse0")
    Y, U = pca_normalize(V, F)
    Yc = Y[:, [1, 2, 0]]                     # cylinder axis (new z) = first principal axis
    img = panorama_image(Yc, F, nth=256, nz=96)
    z0, z1 = Yc[:, 2].min(), Yc[:, 2].max()
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 1.75))
    a0 = fig.add_axes([0.0, 0.02, 0.25, 0.86])
    view = (12, -60)
    gp.render(a0, V, F, view=view)
    R = gp.view_matrix(*view)
    a_ = gp.face_areas(V, F)
    c = (V[F].mean(1) * a_[:, None]).sum(0) / a_.sum()
    s1 = np.sign((((V - c) @ U[:, 0]) ** 3).mean()) or 1.0
    ext = np.abs((V - c) @ U[:, 0]).max() * 1.1
    p = np.array([c - ext * s1 * U[:, 0], c + ext * s1 * U[:, 0]]) @ R.T
    a0.plot(p[:, 0], p[:, 2], "C3--", lw=1)
    a0.annotate("", xy=p[1, [0, 2]], xytext=p[0, [0, 2]], arrowprops=dict(arrowstyle="->", color="C3", lw=1))
    a0.set_title("cylinder axis = 1st PCA axis", fontsize=8)
    a1 = fig.add_axes([0.30, 0.2, 0.42, 0.68])
    a1.imshow(img, cmap="viridis", aspect="auto", origin="lower", extent=[0, 360, z0, z1])
    a1.set_xticks([0, 90, 180, 270, 360]); a1.set_yticks([])
    a1.set_xlabel(r"angle $\theta$ [deg]", fontsize=8); a1.set_ylabel("position along axis", fontsize=8)
    a1.set_title("panoramic view $s(\\theta, z)$", fontsize=8)
    a2 = fig.add_axes([0.77, 0.2, 0.22, 0.68])
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    cy, cx = np.array(Fm.shape) // 2
    a2.imshow(np.log1p(Fm[cy - 24:cy + 25, cx - 24:cx + 25]), cmap="magma", origin="lower",
              extent=[-24, 24, -24, 24])
    a2.set_title(r"$\log|\mathcal{F}(s)|$ (low frequencies)", fontsize=8)
    a2.tick_params(labelsize=6)
    gp.savefig(fig, "ch14_panorama.png")


def d2_descriptor(V, F, n=20000, bins=48, seed=0):
    r = np.random.default_rng(seed)
    a = gp.face_areas(V, F)
    f = r.choice(len(F), 2 * n, p=a / a.sum())
    u, v = r.random(2 * n), r.random(2 * n)
    s = np.sqrt(u)
    P = (V[F[f, 0]] * (1 - s)[:, None] + V[F[f, 1]] * (s * (1 - v))[:, None] + V[F[f, 2]] * (s * v)[:, None])
    d = np.linalg.norm(P[:n] - P[n:], axis=1)
    d = d / d.mean()
    h, _ = np.histogram(d, bins=bins, range=(0, 3), density=True)
    return h


COLLECTION = ["cat0", "cat1", "cat2", "cat5", "cat8", "horse0", "horse5", "horse6", "horse10", "horse15"]


def bof_figure():
    """Bag of features with HKS words on a cat and a horse."""
    from sklearn.cluster import KMeans
    shapes = ["cat0", "horse0"]
    H, data = [], {}
    for nm in shapes:
        lam, Phi, m = spectrum(nm)
        ts = np.geomspace(4 * np.log(10) / lam[-1], 4 * np.log(10) / lam[1], 12)
        hks = (Phi ** 2) @ np.exp(-np.outer(lam, ts))
        hks = hks / (m @ hks)
        data[nm] = (np.log(hks), m)
        H.append(np.log(hks))
    km = KMeans(n_clusters=12, n_init=4, random_state=0).fit(np.vstack(H)[::7])
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 1.9))
    cmap = plt.get_cmap("tab20")
    hists = []
    for j, nm in enumerate(shapes):
        V, F = raw(nm)
        X, m = data[nm]
        w = km.predict(X)
        ax = fig.add_axes([j * 0.26, 0.0, 0.26, 0.92])
        gp.render(ax, V, F, face_colors=cmap(w[F[:, 0]] % 20)[:, :3], view=VIEW_CAT if j == 0 else (10, -60),
                  title=f"{nm[:-1]}: visual words")
        ax.title.set_fontsize(8)
        hist = np.bincount(w, weights=m, minlength=12)
        hists.append(hist / hist.sum())
    top = max(h.max() for h in hists) * 1.08
    for j, (nm, h) in enumerate(zip(shapes, hists)):
        ax2 = fig.add_axes([0.58 + j * 0.215, 0.2, 0.18, 0.6])
        ax2.bar(range(12), h, color=[cmap(i % 20) for i in range(12)])
        ax2.set_xticks([]); ax2.set_ylim(0, top)
        ax2.set_title(f"{nm[:-1]}: word histogram", fontsize=8)
        ax2.set_xlabel("word", fontsize=7)
        if j:
            ax2.set_yticklabels([])
    print("BoF histogram L1 distance cat-horse:", np.abs(hists[0] - hists[1]).sum().round(3))
    gp.savefig(fig, "ch14_bof.png")


# =============================================================== chapter 15
def sign_align(PhiA, PhiB, mA):
    c = np.einsum("ij,ij->j", PhiA * mA[:, None], PhiB)
    return np.where(c >= 0, 1.0, -1.0)


def eigenfunction_figure():
    lamA, PhiA, mA = spectrum("cat0")
    lamB, PhiB, mB = spectrum("cat2")
    VA, FA = raw("cat0")
    VB, FB = raw("cat2")
    s = sign_align(PhiA, PhiB, mA)
    fig, axs = plt.subplots(2, 5, figsize=(gp.TEXTWIDTH, 2.6))
    for j in range(5):
        k = j + 1
        for r, (V, F, Phi, sg) in enumerate([(VA, FA, PhiA, 1.0), (VB, FB, PhiB, s[k])]):
            f = sg * Phi[:, k]
            lim = np.percentile(np.abs(f), 99)
            gp.render(axs[r, j], V, F, c=f, cmap="RdBu_r", clim=(-lim, lim), view=VIEW_CAT)
        axs[0, j].set_title(rf"$\phi_{{{k + 1}}}$: $\lambda={lamA[k]:.1f}$ / {lamB[k]:.1f}", fontsize=7.5)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    gp.savefig(fig, "ch15_eigenfunctions.png")


def shapedna_figure():
    names = COLLECTION
    dna = {}
    for nm in names:
        lam, Phi, m = spectrum(nm, k=100)
        dna[nm] = lam                       # area already normalised to 1
    d2 = {}
    for nm in names:
        V, F = raw(nm)
        d2[nm] = d2_descriptor(V, F)
    k = np.arange(1, 101)
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 2.3))
    a0 = fig.add_axes([0.06, 0.18, 0.26, 0.72])
    for nm in names:
        col = "C3" if nm.startswith("cat") else "C0"
        a0.plot(k[1:40], dna[nm][1:40] / (4 * np.pi * k[1:40]), color=col, lw=1)
    a0.plot([], [], "C3", label="cats"); a0.plot([], [], "C0", label="horses")
    a0.axhline(1, color="k", ls=":", lw=0.8)
    a0.set_xlabel("$k$"); a0.set_ylabel(r"$A\lambda_k/(4\pi k)$"); a0.legend(frameon=False)
    a0.set_title("ShapeDNA vs. Weyl's law", fontsize=8)

    def dmat(desc, kind):
        X = np.array([desc[n] for n in names])
        if kind == "dna":
            kk = np.arange(1, X.shape[1] + 1)
            diff = (X[:, None, 1:] - X[None, :, 1:]) / kk[1:]
        else:
            diff = X[:, None] - X[None, :]
        return np.sqrt((diff ** 2).sum(-1))
    for j, (desc, kind, t) in enumerate([(d2, "d2", "extrinsic: D2 distribution"),
                                          (dna, "dna", "intrinsic: ShapeDNA")]):
        ax = fig.add_axes([0.40 + 0.3 * j, 0.12, 0.27, 0.8])
        D = dmat(desc, kind)
        ax.imshow(D / D.max(), cmap="viridis")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels([n.replace("horse", "h").replace("cat", "c") for n in names], fontsize=5.5, rotation=90)
        ax.set_yticklabels([n.replace("horse", "h").replace("cat", "c") for n in names], fontsize=5.5)
        ax.set_title(t, fontsize=8)
        Dn = D.copy(); np.fill_diagonal(Dn, np.inf)
        lab = np.array([n[:3] for n in names])
        print(t, "1-NN accuracy", np.mean(lab[Dn.argmin(1)] == lab))
    gp.savefig(fig, "ch15_shapedna.png")


def heat_figure():
    lam, Phi, m = spectrum("cat0")
    V, F = raw("cat0")
    x_tail = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))      # tip of the tail
    f0 = (np.linalg.norm(V - V[x_tail], axis=1) < 0.1).astype(float)
    c = Phi.T @ (m * f0)
    ts = [0.0, 0.002, 0.01, 0.05, 0.25]
    fig, axs = plt.subplots(1, 5, figsize=(gp.TEXTWIDTH, 1.25))
    for ax, t in zip(axs, ts):
        f = f0 if t == 0 else Phi @ (np.exp(-lam * t) * c)
        gp.render(ax, V, F, c=f, cmap="hot_r", clim=(f.min(), f.max()), view=VIEW_CAT, title=f"$t={t}$")
    gp.savefig(fig, "ch15_heat.png")
    # heat kernels at two points and three scales
    x_ear = int(np.argmax(V[:, 2]))
    x_back = int(np.argmin(np.linalg.norm(V - (V.mean(0) + np.array([0, 0, 0.3])), axis=1)))
    fig, axs = plt.subplots(2, 3, figsize=(gp.TEXTWIDTH * 0.75, 2.0))
    for r, x in enumerate([x_ear, x_back]):
        for j, t in enumerate([0.005, 0.03, 0.2]):
            h = Phi @ (np.exp(-t * lam) * Phi[x])
            gp.render(axs[r, j], V, F, c=h, cmap="hot_r", clim=(h.min(), h.max()), view=VIEW_CAT)
            tight(axs[r, j], V, VIEW_CAT)
            if r == 0:
                axs[r, j].set_title(f"$h_t(x,\\cdot)$, $t={t}$", fontsize=8)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    gp.savefig(fig, "ch15_heat_kernel.png")


def distances_figure():
    lam, Phi, m = spectrum("cat0")
    V, F = raw("cat0")
    x = int(np.argmax(V[:, 2]))
    geo = gp.heat_geodesic(V, F, x)
    emb = {
        r"diffusion, $t=0.01$": lambda l: np.exp(-0.01 * l),
        r"diffusion, $t=0.1$": lambda l: np.exp(-0.1 * l),
        r"GPS $1/\sqrt{\lambda}$": lambda l: 1 / np.sqrt(l),
        r"biharmonic $1/\lambda$": lambda l: 1 / l,
    }
    fig, axs = plt.subplots(1, 5, figsize=(gp.TEXTWIDTH, 1.3))
    gp.render(axs[0], V, F, c=geo, cmap="viridis", view=VIEW_CAT, title="geodesic")
    for ax, (t, G) in zip(axs[1:], emb.items()):
        Psi = Phi[:, 1:] * G(lam[1:])
        d = np.linalg.norm(Psi - Psi[x], axis=1)
        gp.render(ax, V, F, c=d, cmap="viridis", view=VIEW_CAT, title=t)
        print(t, "Spearman with geodesic", np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(geo)))[0, 1])
    for a in axs:
        a.title.set_fontsize(7.5)
    gp.savefig(fig, "ch15_distances.png")


def hks_figure():
    data = {}
    for nm in ["cat0", "cat2"]:
        lam, Phi, m = spectrum(nm)
        data[nm] = (lam, Phi, m)
    lam = data["cat0"][0]
    ts = np.geomspace(4 * np.log(10) / lam[-1], 4 * np.log(10) / lam[1], 16)
    fig, axs = plt.subplots(2, 4, figsize=(gp.TEXTWIDTH, 2.5))
    H = {nm: (Phi ** 2) @ np.exp(-np.outer(l, ts)) for nm, (l, Phi, m) in data.items()}
    for j, q in enumerate([0, 5, 10, 15]):
        lo, hi = np.percentile(np.log(np.r_[H["cat0"][:, q], H["cat2"][:, q]]), [1, 99])
        for r, nm in enumerate(["cat0", "cat2"]):
            V, F = raw(nm)
            gp.render(axs[r, j], V, F, c=np.log(H[nm][:, q]), cmap="turbo", clim=(lo, hi), view=VIEW_CAT)
        axs[0, j].set_title(f"$\\log$ HKS, $t={ts[q]:.3g}$", fontsize=8)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    gp.savefig(fig, "ch15_hks.png")
    rel = np.median(np.abs(H["cat2"] / H["cat0"] - 1), axis=0)
    print("HKS relative change between poses (median per t):", rel.round(3))


def hks_curvature_figure():
    abc = np.array([1.3, 1.0, 0.6])
    V, F = gp.icosphere(4)
    V = V * abc
    L, M = gp.laplacian(V, F), gp.mass_matrix(V, F)
    lam, Phi = sla.eigh(L.toarray(), M.toarray())
    x, y, z = V.T
    a, b, c = abc
    K = 1 / (a * b * c) ** 2 / ((x / a ** 2) ** 2 + (y / b ** 2) ** 2 + (z / c ** 2) ** 2) ** 2
    h = gp.edge_lengths(V, F).mean()
    tgrid = np.geomspace(2e-3, 0.5, 30)
    Hh = (Phi ** 2) @ np.exp(-np.outer(lam, tgrid))
    corr = [pearsonr(Hh[:, j], K)[0] for j in range(len(tgrid))]
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0), gridspec_kw={"width_ratios": [0.75, 1, 1]})
    plt.subplots_adjust(wspace=0.6, left=0.02, right=0.98, bottom=0.2)
    gp.render(axs[0], V, F, c=K, cmap="viridis", view=(25, -50), title="ellipsoid: exact $K$")
    axs[0].title.set_fontsize(8)
    axs[1].semilogx(tgrid / h ** 2, corr, "o-", ms=2.5)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_xlabel("$t/h^2$"); axs[1].set_ylabel("Pearson(HKS, $K$)")
    j = int(np.argmax(corr))
    y = Hh[:, j] - 1 / (4 * np.pi * tgrid[j])
    slope = np.polyfit(K, y, 1)[0]
    axs[2].plot(K, y, ".", ms=1.5)
    kk = np.linspace(K.min(), K.max(), 2)
    off = np.mean(y - K / (12 * np.pi))
    axs[2].plot(kk, kk / (12 * np.pi) + off, "k--", lw=0.8, label=r"slope $1/12\pi$")
    axs[2].set_xlabel("$K$"); axs[2].set_ylabel(r"HKS$\,-\,1/4\pi t$"); axs[2].legend(frameon=False, fontsize=7)
    axs[2].set_title(f"$t={tgrid[j]:.3f}$ ($t/h^2={tgrid[j] / h ** 2:.0f}$)", fontsize=8)
    print("HKS-curvature: best corr", max(corr), "at t/h^2", tgrid[j] / h ** 2, "fitted slope", slope,
          "theory", 1 / (12 * np.pi), "h", h)
    gp.savefig(fig, "ch15_hks_curvature.pdf")


def wks(lam, Phi, E=100, wmult=7):
    loglam = np.log(lam[1:])
    P2 = Phi[:, 1:] ** 2
    e = np.linspace(loglam[0], loglam[-1] / 1.02, E)
    sigma = wmult * (e[1] - e[0])
    W = np.exp(-(e[None, :] - loglam[:, None]) ** 2 / (2 * sigma ** 2))
    return (P2 @ W) / W.sum(0), e, sigma


def wks_figure():
    lam, Phi, m = spectrum("cat0")
    V, F = raw("cat0")
    W, e, sig = wks(lam, Phi)
    ts = np.geomspace(4 * np.log(10) / lam[-1], 4 * np.log(10) / lam[1], 16)
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 3.4))
    a0 = fig.add_axes([0.07, 0.68, 0.4, 0.27])
    a1 = fig.add_axes([0.57, 0.68, 0.4, 0.27])
    lg = np.geomspace(lam[1], lam[-1], 400)
    for t in ts[::5]:
        a0.semilogx(lg, np.exp(-t * lg), lw=1)
    a0.set_title(r"HKS filters $e^{-t\lambda}$ (low-pass)", fontsize=8); a0.set_xlabel(r"$\lambda$")
    for ej in e[::20]:
        a1.semilogx(lg, np.exp(-(ej - np.log(lg)) ** 2 / (2 * sig ** 2)), lw=1)
    a1.set_title(r"WKS filters $f_e^2(\lambda)$ (band-pass in $\log\lambda$)", fontsize=8); a1.set_xlabel(r"$\lambda$")
    for a in (a0, a1):
        a.plot(lam[1:], np.zeros(len(lam) - 1), "|k", ms=5)
    for j, q in enumerate([0, 33, 66, 99]):
        ax = fig.add_axes([j * 0.25, 0.0, 0.25, 0.47])
        w = np.log(W[:, q])
        lo, hi = np.percentile(w, [1, 99])
        gp.render(ax, V, F, c=w, cmap="turbo", clim=(lo, hi), view=VIEW_CAT, title=f"log WKS, $e={e[q]:.2f}$")
        ax.title.set_fontsize(8)
    gp.savefig(fig, "ch15_wks.png")


def gps_figure():
    lamA, PhiA, mA = spectrum("cat0")
    lamB, PhiB, mB = spectrum("cat2")
    VA, FA = raw("cat0")
    VB, FB = raw("cat2")
    s = sign_align(PhiA, PhiB, mA)
    EA = PhiA[:, 1:4] / np.sqrt(lamA[1:4])
    EB = PhiB[:, 1:4] * s[1:4] / np.sqrt(lamB[1:4])
    col = VA[:, 0]
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 1.9))
    for j, (V, F, t) in enumerate([(VA, FA, "pose A (colour = $x$ in A)"), (VB, FB, "pose B, same colours")]):
        ax = fig.add_axes([0.25 * j, 0.0, 0.25, 0.92])
        gp.render(ax, V, F, c=col, cmap="turbo", view=VIEW_CAT, title=t)
        ax.title.set_fontsize(7.5)
    lo = np.percentile(np.r_[EA, EB], 0.5, axis=0)
    hi = np.percentile(np.r_[EA, EB], 99.5, axis=0)
    for j, (E, t) in enumerate([(EA, "GPS embedding of A"), (EB, "GPS embedding of B")]):
        ax = fig.add_axes([0.5 + 0.25 * j, 0.0, 0.25, 0.92], projection="3d")
        ax.scatter(E[::3, 0], E[::3, 1], E[::3, 2], c=col[::3], cmap="turbo", s=0.8, linewidths=0)
        ax.view_init(20, -70)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.25)
        ax.set_axis_off()
        ax.set_title(t, fontsize=7.5, y=0.95)
    gp.savefig(fig, "ch15_gps.png")
    print("GPS relative difference", np.linalg.norm(EA - EB) / np.linalg.norm(EA))


def descriptor_matching_figure():
    lamA, PhiA, mA = spectrum("cat0")
    lamB, PhiB, mB = spectrum("cat2")
    VA, FA = raw("cat0")
    idx = rng.choice(len(VA), 300, replace=False)
    G = gp.dijkstra_geodesic(VA, FA, idx)
    ts = np.geomspace(4 * np.log(10) / lamA[-1], 4 * np.log(10) / lamA[1], 16)
    descs = {}
    HA = (PhiA ** 2) @ np.exp(-np.outer(lamA, ts)); HB = (PhiB ** 2) @ np.exp(-np.outer(lamB, ts))
    descs["HKS"] = (HA / (mA @ HA), HB / (mB @ HB))
    descs["WKS"] = (wks(lamA, PhiA)[0], wks(lamB, PhiB)[0])
    r_grid = np.linspace(0, 0.5, 101)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for name, (DA, DB) in descs.items():
        nn = nn_search(np.log(DB), np.log(DA[idx]))
        err = G[np.arange(len(idx)), nn]          # A and B share vertex indices
        ax.plot(r_grid, [(err <= r).mean() for r in r_grid], label=f"{name} (median {np.median(err):.3f})")
    ax.set_xlabel(r"geodesic error (shapes of unit area)"); ax.set_ylabel("fraction of points")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax.set_title("nearest neighbour in descriptor space", fontsize=8)
    gp.savefig(fig, "ch15_matching.pdf")


# =============================================================== chapter 16
def fmap_setup():
    lamM, PhiM, mM = spectrum("cat0")
    VM, FM = raw("cat0")
    VKfull, FKfull = raw("cat5")
    VN, FN = gp.decimate(VKfull, FKfull, 16000)
    T_gt = cKDTree(VKfull).query(VN)[1]          # N -> M through shared vertex indices of cat5/cat0
    path = os.path.join(CACHE, "catN_200.npz")
    if os.path.exists(path):
        d = np.load(path)
        lamN, PhiN, mN = d["lam"], d["Phi"], d["m"]
    else:
        lamN, PhiN, mN = spectrum("", 200, VN, FN)
        np.savez(path, lam=lamN, Phi=PhiN, m=mN)
    return (VM, FM, lamM, PhiM, mM), (VN, FN, lamN, PhiN, mN), T_gt


def fmap_from_pointwise(T, PhiM, PhiN, mN, k):
    return PhiN[:, :k].T @ (mN[:, None] * PhiM[T, :k])


def pointwise_from_fmap(C, PhiM, PhiN):
    """T(y) = argmin_x || C^T-mapped embedding of x - Phi_N(y) ||, i.e. rows of Phi_M C^T vs rows of Phi_N."""
    k = C.shape[0]
    return nn_search(PhiM[:, :k] @ C.T, PhiN[:, :k])


def solve_fmap(A, B, lamM, lamN, mu):
    k = A.shape[0]
    C = np.zeros((k, k))
    AAt = A @ A.T
    for j in range(k):
        C[j] = np.linalg.solve(AAt + mu * np.diag((lamN[j] - lamM) ** 2), A @ B[j])
    return C


def coord_colors(V):
    X = (V - V.min(0)) / (V.max(0) - V.min(0))
    return np.clip(0.15 + 0.8 * X, 0, 1)


def fmap_figures():
    (VM, FM, lamM, PhiM, mM), (VN, FN, lamN, PhiN, mN), T_gt = fmap_setup()
    k = 30
    C_gt = fmap_from_pointwise(T_gt, PhiM, PhiN, mN, k)
    T_rand = rng.integers(0, len(VM), len(VN))
    C_rand = fmap_from_pointwise(T_rand, PhiM, PhiN, mN, k)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, C, t in [(axs[0], C_gt, r"$\mathbf{C}$ of the true map ($30\times30$)"), (axs[1], C_rand, r"$\mathbf{C}$ of a random map")]:
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    dl = np.abs(lamN[:k, None] - lamM[None, :k])
    axs[2].scatter(dl.ravel() + 1e-2, np.abs(C_gt).ravel(), s=3, alpha=0.6, label="true map")
    axs[2].scatter(dl.ravel() + 1e-2, np.abs(C_rand).ravel(), s=3, alpha=0.4, label="random map")
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"$|\lambda^{\mathcal{N}}_j-\lambda^{\mathcal{M}}_i|$")
    axs[2].set_ylabel(r"$|C_{ji}|$"); axs[2].legend(frameon=False, fontsize=6.5, markerscale=3, loc="center left")
    axs[2].set_title("commutativity with the Laplacian", fontsize=8)
    gp.savefig(fig, "ch16_cmatrix.png")
    # segmentation transfer by spectral clustering parts on M
    from sklearn.cluster import KMeans
    lab = KMeans(8, n_init=4, random_state=0).fit_predict(PhiM[:, 1:8] / np.sqrt(lamM[1:8]))
    onehot = np.eye(8)[lab]
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.35))
    gp.render(axs[0], VM, FM, face_colors=cmap(lab[FM[:, 0]])[:, :3], view=VIEW_CAT, title="parts on $\\mathcal{M}$")
    for ax, kk in zip(axs[1:], [5, 20, 80]):
        Ck = fmap_from_pointwise(T_gt, PhiM, PhiN, mN, kk)
        a = PhiM[:, :kk].T @ (mM[:, None] * onehot)
        g = PhiN[:, :kk] @ (Ck @ a)
        labN = g.argmax(1)
        acc = np.mean(labN == lab[T_gt])
        gp.render(ax, VN, FN, face_colors=cmap(labN[FN[:, 0]])[:, :3], view=VIEW_CAT,
                  title=f"$k={kk}$: {100 * acc:.0f}% correct")
    for a in axs:
        a.title.set_fontsize(7.5)
    gp.savefig(fig, "ch16_transfer.png")
    # bound: pointwise map from the true C of size k
    idx = rng.choice(len(VN), 300, replace=False)
    G = gp.dijkstra_geodesic(VM, FM, T_gt[idx])

    def errors(T):
        return G[np.arange(len(idx)), T[idx]]
    ks = [5, 10, 20, 40, 80, 120, 200]
    bound = [np.median(errors(pointwise_from_fmap(fmap_from_pointwise(T_gt, PhiM, PhiN, mN, kk), PhiM, PhiN))) for kk in ks]
    print("bound median error by k", dict(zip(ks, np.round(bound, 4))))
    # estimation from descriptors + landmarks, then spectral ICP and ZoomOut
    k0 = 20
    lmN = [int(np.argmax(VN[:, 2])), int(np.argmax(np.linalg.norm(VN - VN.mean(0), axis=1))),
           int(np.argmin(VN[:, 2])), int(np.argmax(VN[:, 0])), int(np.argmin(VN[:, 0]))]
    lmM = T_gt[lmN]                              # landmark correspondences given by the user

    def descriptors(lam, Phi, m, lms):
        W = wks(lam, Phi, 50)[0]
        Hk = [np.stack([Phi @ (np.exp(-t * lam) * Phi[l]) for l in lms], 1) for t in (0.05, 0.2)]
        D = np.hstack([W] + Hk)
        return D / np.sqrt(m @ D ** 2)
    A = PhiM[:, :k0].T @ (mM[:, None] * descriptors(lamM, PhiM, mM, lmM))
    B = PhiN[:, :k0].T @ (mN[:, None] * descriptors(lamN, PhiN, mN, lmN))
    C0 = solve_fmap(A, B, lamM[:k0], lamN[:k0], 1e-3)
    T0 = pointwise_from_fmap(C0, PhiM, PhiN)
    C = C0.copy()
    for _ in range(10):
        T = pointwise_from_fmap(C, PhiM, PhiN)
        C = fmap_from_pointwise(T, PhiM, PhiN, mN, k0)
        U, _, Vt = np.linalg.svd(C)
        C = U @ Vt
    T_icp = pointwise_from_fmap(C, PhiM, PhiN)
    C = C0.copy()
    kk = k0
    while kk < 120:
        T = pointwise_from_fmap(C, PhiM, PhiN)
        kk = min(kk + 2, 120)
        C = fmap_from_pointwise(T, PhiM, PhiN, mN, kk)
    T_zo = pointwise_from_fmap(C, PhiM, PhiN)
    T_bound = pointwise_from_fmap(fmap_from_pointwise(T_gt, PhiM, PhiN, mN, 120), PhiM, PhiN)
    colM = coord_colors(VM)
    maps = {"initial estimate": T0, "spectral ICP": T_icp, "ZoomOut": T_zo, "ground truth": T_gt}
    fig, axs = plt.subplots(1, 5, figsize=(gp.TEXTWIDTH, 1.35))
    gp.render(axs[0], VM, FM, face_colors=colM[FM[:, 0]], view=VIEW_CAT, title="source $\\mathcal{M}$")
    for ax, (t, T) in zip(axs[1:], maps.items()):
        gp.render(ax, VN, FN, face_colors=colM[T][FN[:, 0]], view=VIEW_CAT, title=t)
    for a in axs:
        a.title.set_fontsize(7)
    gp.savefig(fig, "ch16_pipeline.png")
    r_grid = np.linspace(0, 0.3, 101)
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.0))
    plt.subplots_adjust(wspace=0.35)
    for t, T in list(maps.items())[:3] + [("bound: true $\\mathbf{C}$, $k=120$", T_bound)]:
        e = errors(T)
        axs[0].plot(r_grid, [(e <= r).mean() for r in r_grid], label=f"{t} ({np.median(e):.3f})")
        print("fmap", t, "median", np.median(e).round(4), "mean", e.mean().round(4))
    axs[0].set_xlabel("geodesic error"); axs[0].set_ylabel("fraction of points")
    axs[0].legend(frameon=False, fontsize=6); axs[0].grid(alpha=0.3)
    axs[1].plot(ks, bound, "o-", ms=4)
    axs[1].set_xlabel("$k$"); axs[1].set_ylabel("median geodesic error")
    axs[1].set_title(r"map recovered from the true $\mathbf{C}$", fontsize=8); axs[1].grid(alpha=0.3)
    gp.savefig(fig, "ch16_errors.pdf")


if __name__ == "__main__":
    todo = [pose_normalization_figure, panorama_figure, bof_figure, eigenfunction_figure,
            shapedna_figure, heat_figure, distances_figure, hks_figure, hks_curvature_figure,
            wks_figure, gps_figure, descriptor_matching_figure, fmap_figures]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
