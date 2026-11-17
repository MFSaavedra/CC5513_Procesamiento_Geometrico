"""Figures for chapters 1-3: gallery, 3D representations, mesh data structures."""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.spatial import cKDTree, Delaunay
from skimage.measure import marching_cubes
import gp

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- gallery
def gallery():
    names = [("bunny", (12, -70)), ("armadillo", (10, 200)), ("fandisk", (25, -40)),
             ("planck", (5, -70)), ("cat0", (15, -40)), ("horse0", (10, -60)),
             ("elephant", (10, -50)), ("spot", (15, -40))]
    fig, axs = plt.subplots(2, 4, figsize=(gp.TEXTWIDTH, 3.1))
    for ax, (nm, view) in zip(axs.ravel(), names):
        V, F = gp.model(nm, faces=40000)
        gp.render(ax, V, F, view=view, title=nm.replace("0", ""))
    plt.subplots_adjust(wspace=0.02, hspace=0.12)
    gp.savefig(fig, "ch01_gallery.png")


# ---------------------------------------------------------------- representations
def voxelize_points(P, n):
    lo, hi = P.min(0), P.max(0)
    h = (hi - lo).max() / n
    idx = np.floor((P - lo) / h).astype(int)
    idx = np.unique(np.clip(idx, 0, n - 1), axis=0)
    return idx, lo, h


def cube_mesh(idx, lo, h):
    """Boundary faces of a set of occupied voxels as quads -> triangles."""
    occ = set(map(tuple, idx))
    Vs, Fs = [], []
    dirs = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
    for c in idx:
        for d, quad in dirs:
            if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) in occ:
                continue
            base = len(Vs)
            for q in quad:
                Vs.append(lo + h * (c + np.array(q)))
            Fs += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(Vs), np.array(Fs)


def sample_surface(V, F, n, rng):
    a = gp.face_areas(V, F)
    f = rng.choice(len(F), n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.c_[1 - s, s * (1 - r2), s * r2]
    P = (V[F[f]] * w[:, :, None]).sum(1)
    return P, gp.face_normals(V, F)[f]


def poisson_disk(P, N, r):
    """Greedy dart throwing on a dense candidate set: keep points at distance >= r."""
    order = rng.permutation(len(P))
    tree = cKDTree(P)
    keep = np.zeros(len(P), bool)
    blocked = np.zeros(len(P), bool)
    for i in order:
        if blocked[i]:
            continue
        keep[i] = True
        blocked[tree.query_ball_point(P[i], r)] = True
    return P[keep], N[keep]


def representations():
    V, F = gp.model("bunny", faces=30000)
    view = (12, -70)
    fig, axs = plt.subplots(1, 5, figsize=(gp.TEXTWIDTH, 1.6))
    P, N = sample_surface(V, F, 4000, rng)
    gp.render_points(axs[0], P, N=N, view=view, s=0.6, title="points")
    Vs, Fs = gp.model("bunny", faces=1500)
    gp.render(axs[1], Vs, Fs, edges=True, view=view, title="mesh")
    Pd, _ = sample_surface(V, F, 200000, rng)
    idx, lo, h = voxelize_points(Pd, 24)
    Vc, Fc = cube_mesh(idx, lo, h)
    gp.render(axs[2], Vc, Fc, view=view, smooth=False, edges=True, lw=0.1,
              title=r"voxels")
    # implicit: signed distance of the hole-filled mesh
    n = 72
    ax_, S = gp.closed_sdf_grid(V, F, n=n)
    lo = np.array([a[0] for a in ax_]); hi = np.array([a[-1] for a in ax_])
    sl = S[:, n // 2, :].T
    axs[3].imshow(sl, origin="lower", cmap="RdBu_r", vmin=-0.15, vmax=0.15,
                  extent=[lo[0], hi[0], lo[2], hi[2]])
    axs[3].contour(ax_[0], ax_[2], sl, levels=[0], colors="k", linewidths=0.8)
    axs[3].set_aspect("equal"); axs[3].axis("off"); axs[3].set_title("slice of $f$")
    Vm, Fm = gp.iso_surface(S, ax_)
    Vm, Fm = gp.decimate(Vm, Fm, 30000)
    gp.render(axs[4], Vm, Fm, view=view, title=r"$\{f=0\}$")
    plt.subplots_adjust(wspace=0.03)
    gp.savefig(fig, "ch02_representations.png")


def voxel_resolution():
    V, F = gp.model("bunny", faces=30000)
    Pd, _ = sample_surface(V, F, 400000, rng)
    view = (12, -70)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH * 0.8, 1.8))
    for ax, n in zip(axs, [12, 24, 48]):
        idx, lo, h = voxelize_points(Pd, n)
        Vc, Fc = cube_mesh(idx, lo, h)
        gp.render(ax, Vc, Fc, view=view, smooth=False, edges=n < 40, lw=0.08)
        ax.set_title(f"$n={n}$: {len(idx)} of ${n}^3$ cells", fontsize=8)
    gp.savefig(fig, "ch02_voxel_resolution.png")


def sampling():
    V, F = gp.model("bunny", faces=30000)
    Pu, Nu = sample_surface(V, F, 3000, rng)
    Pc, Nc = sample_surface(V, F, 60000, rng)
    Pp, Np = poisson_disk(Pc, Nc, 0.0128)
    Pp, Np = Pp[:3000], Np[:3000]
    view = (12, -70)
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 2.0))
    ax1 = fig.add_axes([0.0, 0.0, 0.3, 1.0])
    ax2 = fig.add_axes([0.3, 0.0, 0.3, 1.0])
    ax3 = fig.add_axes([0.68, 0.18, 0.31, 0.72])
    gp.render_points(ax1, Pu, N=Nu, view=view, s=1.2, title=f"uniform random ({len(Pu)} pts)")
    gp.render_points(ax2, Pp, N=Np, view=view, s=1.2, title=f"Poisson disk ({len(Pp)} pts)")
    du = cKDTree(Pu).query(Pu, 2)[0][:, 1]
    dp = cKDTree(Pp).query(Pp, 2)[0][:, 1]
    bins = np.linspace(0, 0.05, 40)
    ax3.hist(du, bins, alpha=0.6, label="uniform")
    ax3.hist(dp, bins, alpha=0.6, label="Poisson disk")
    ax3.set_xlabel("distance to nearest neighbour")
    ax3.set_ylabel("count")
    ax3.legend(frameon=False)
    gp.savefig(fig, "ch02_sampling.png")


class KD:
    def __init__(self, P, depth=0, lo=None, hi=None):
        self.lo = P.min(0) if lo is None else lo
        self.hi = P.max(0) if hi is None else hi
        self.axis = depth % 2
        if len(P) <= 1:
            self.leaf, self.P = True, P
            return
        self.leaf = False
        o = np.argsort(P[:, self.axis])
        m = len(P) // 2
        self.split = P[o[m], self.axis]
        self.pt = P[o[m]]
        hl, lh = self.hi.copy(), self.lo.copy()
        hl[self.axis] = self.split
        lh[self.axis] = self.split
        self.left = KD(P[o[:m]], depth + 1, self.lo, hl)
        self.right = KD(P[o[m:]], depth + 1, lh, self.hi)

    def draw(self, ax, lw=1.2, level=0):
        if self.leaf:
            return
        if self.axis == 0:
            ax.plot([self.split] * 2, [self.lo[1], self.hi[1]], color=plt.cm.viridis(level / 7),
                    lw=max(lw * 0.8 ** level, 0.4))
        else:
            ax.plot([self.lo[0], self.hi[0]], [self.split] * 2, color=plt.cm.viridis(level / 7),
                    lw=max(lw * 0.8 ** level, 0.4))
        self.left.draw(ax, lw, level + 1)
        self.right.draw(ax, lw, level + 1)

    def range_visit(self, box_lo, box_hi, visited):
        """Collect cells visited by a range query (for the figure)."""
        if (self.hi < box_lo).any() or (self.lo > box_hi).any():
            return
        visited.append((self.lo, self.hi, self.leaf))
        if not self.leaf:
            self.left.range_visit(box_lo, box_hi, visited)
            self.right.range_visit(box_lo, box_hi, visited)


def kdtree_figure():
    P = rng.random((60, 2))
    T = KD(P, lo=np.zeros(2), hi=np.ones(2))
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.1))
    ax = axs[0]
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1))
    T.draw(ax)
    ax.plot(P[:, 0], P[:, 1], "k.", ms=3)
    ax.set_title("kd-tree cells")
    ax = axs[1]
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1))
    T.draw(ax, lw=0.8)
    blo, bhi = np.array([0.35, 0.3]), np.array([0.7, 0.62])
    vis = []
    T.range_visit(blo, bhi, vis)
    for lo, hi, leaf in vis:
        if leaf:
            continue
        ax.add_patch(Rectangle(lo, *(hi - lo), color="C1", alpha=0.06, lw=0))
    ax.add_patch(Rectangle(blo, *(bhi - blo), fill=False, color="C3", lw=1.4))
    inside = np.all((P >= blo) & (P <= bhi), 1)
    ax.plot(P[~inside, 0], P[~inside, 1], "k.", ms=3)
    ax.plot(P[inside, 0], P[inside, 1], "o", color="C3", ms=3.5)
    ax.set_title("range query")
    ax = axs[2]
    q = np.array([0.52, 0.45])
    d, j = cKDTree(P).query(q, 6)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1))
    ax.plot(P[:, 0], P[:, 1], "k.", ms=3)
    for jj in j:
        ax.plot([q[0], P[jj, 0]], [q[1], P[jj, 1]], color="C0", lw=0.8)
    ax.add_patch(Circle(q, d[-1], fill=False, ls="--", color="C0", lw=0.8))
    ax.plot(*q, "*", color="C3", ms=8)
    ax.set_title(r"$k$-NN query ($k=6$)")
    for a in axs:
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02); a.set_aspect("equal"); a.axis("off")
    gp.savefig(fig, "ch02_kdtree.png")


def kdtree_benchmark():
    ns = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5]).astype(int)
    tb, tk, tq = [], [], []
    nq = 2000
    for n in ns:
        P = rng.random((n, 3))
        Q = rng.random((nq, 3))
        t0 = time.perf_counter()
        if n <= 1e5:
            for s in range(0, nq, 200):
                d = ((Q[s:s + 200, None, :] - P[None, :, :]) ** 2).sum(-1)
                d.argmin(1)
            tb.append((time.perf_counter() - t0) / nq)
        else:
            tb.append(np.nan)
        t0 = time.perf_counter()
        tree = cKDTree(P)
        tk.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tree.query(Q, 1)
        tq.append((time.perf_counter() - t0) / nq)
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.loglog(ns, np.array(tb) * 1e6, "o-", label="brute force, per query")
    ax.loglog(ns, np.array(tq) * 1e6, "s-", label="kd-tree, per query")
    ax.loglog(ns, np.array(tk) * 1e6 / ns, "^-", label=r"kd-tree build / $n$")
    ax.set_xlabel("number of points $n$")
    ax.set_ylabel(r"time [$\mu$s]")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    gp.savefig(fig, "ch02_kdtree_bench.pdf")


# ---------------------------------------------------------------- Euler / genus
def sdf_genus2(P):
    def torus_sdf(P, c, R=0.75, r=0.28):
        Q = P - np.array(c)
        q = np.c_[np.hypot(Q[:, 0], Q[:, 1]) - R, Q[:, 2]]
        return np.linalg.norm(q, axis=1) - r
    a, b = torus_sdf(P, (-0.72, 0, 0)), torus_sdf(P, (0.72, 0, 0))
    k = 0.15
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def euler_figure():
    meshes = []
    S, FS = gp.icosphere(2)
    meshes.append(("sphere", S, FS))
    T, FT = gp.torus(1.0, 0.4, 36, 18)
    meshes.append(("torus", T, FT))
    n = 64
    ax_ = np.linspace(-1.8, 1.8, n)
    G = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
    Sd = sdf_genus2(G).reshape(n, n, n)
    Vm, Fm = gp.iso_surface(Sd, [ax_] * 3)
    Vm, Fm = gp.decimate(Vm, Fm, 3000)
    meshes.append(("double torus", Vm, Fm))
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 1.9))
    for ax, (nm, V, F) in zip(axs, meshes):
        gp.render(ax, V, F, edges=True, lw=0.12, view=(35, -60), smooth=False)
        nv, ne, nf = len(np.unique(F)), len(gp.edges(F)), len(F)
        chi = nv - ne + nf
        ax.set_title(f"{nm}: $\\chi={chi}$, $g={(2 - chi) // 2}$\n$V={nv}$, $E={ne}$, $F={nf}$", fontsize=8)
    gp.savefig(fig, "ch02_euler.png")


# ---------------------------------------------------------------- SDF and CSG
def sdf_sphere(P, c=(0, 0, 0), r=1.0):
    return np.linalg.norm(P - np.asarray(c), axis=-1) - r


def sdf_box(P, s=(0.75, 0.75, 0.75)):
    q = np.abs(P) - np.asarray(s)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def smooth_union(a, b, k=0.25):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def csg_figure():
    n = 200
    x = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    P = np.stack([X, Y, np.zeros_like(X)], -1)
    a = sdf_sphere(P, (-0.35, 0, 0), 0.8)
    b = sdf_box(P, (0.6, 0.6, 0.6)) + 0 * X
    b = sdf_box(P - np.array([0.45, 0, 0]), (0.55, 0.55, 0.55))
    ops = [("union\n$\\min(f_A,f_B)$", np.minimum(a, b)),
           ("intersection\n$\\max(f_A,f_B)$", np.maximum(a, b)),
           ("difference\n$\\max(f_A,-f_B)$", np.maximum(a, -b)),
           ("smooth union\n($k=0.3$)", smooth_union(a, b, 0.3))]
    fig, axs = plt.subplots(1, 5, figsize=(gp.TEXTWIDTH, 1.55),
                            gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for ax, (t, f) in zip(axs[:4], ops):
        ax.contourf(X, Y, f, levels=np.linspace(-1, 1, 21), cmap="RdBu_r", extend="both")
        ax.contour(X, Y, f, levels=[0], colors="k", linewidths=1.0)
        ax.contour(X, Y, a, levels=[0], colors="0.3", linewidths=0.5, linestyles=":")
        ax.contour(X, Y, b, levels=[0], colors="0.3", linewidths=0.5, linestyles=":")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=7.5)
    # 3D: (box intersect sphere) minus three cylinders, extracted by marching cubes
    m = 96
    g = np.linspace(-1.2, 1.2, m)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
    cyl = np.minimum(np.minimum(np.hypot(G[..., 0], G[..., 1]), np.hypot(G[..., 1], G[..., 2])),
                     np.hypot(G[..., 0], G[..., 2])) - 0.35
    f3 = np.maximum(np.maximum(sdf_box(G), sdf_sphere(G, r=1.0)), -cyl)
    Vm, Fm = gp.iso_surface(f3, [g] * 3)
    gp.render(axs[4], Vm, Fm, view=(25, -50), color=(0.85, 0.72, 0.45),
              title="CSG solid")
    axs[4].title.set_fontsize(7.5)
    gp.savefig(fig, "ch02_csg.png")


# ---------------------------------------------------------------- parametric
def bernstein(n, i, t):
    from math import comb
    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def parametric_figure():
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 2.1))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    xs = np.linspace(0, 3, 4)
    Cx, Cy = np.meshgrid(xs, xs, indexing="ij")
    Cz = np.array([[0, 0.5, 0.5, 0], [0.3, 1.6, 1.2, 0.2], [0.2, 1.0, 1.8, 0.4], [0, 0.3, 0.6, 0]])
    u = np.linspace(0, 1, 30)
    Bu = np.array([bernstein(3, i, u) for i in range(4)])  # (4, nu)
    S = [np.einsum("iu,ij,jv->uv", Bu, C, Bu) for C in (Cx, Cy, Cz)]
    ax.plot_surface(*S, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True, rstride=1,
                    cstride=1)
    ax.plot_wireframe(Cx, Cy, Cz, color="crimson", lw=0.8)
    ax.scatter(Cx, Cy, Cz, color="crimson", s=8)
    ax.set_axis_off(); ax.view_init(28, -60)
    ax.set_title("bicubic B\\'ezier patch and control net".replace("\\'e", "é"), fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    V, F = gp.torus(1.0, 0.4, 40, 20)
    gp.render(ax2, V, F, edges=True, lw=0.15, view=(35, -60), smooth=False,
              title=r"torus from a $(u,v)$ grid: $\chi=0$")
    gp.savefig(fig, "ch02_parametric.png")


# ---------------------------------------------------------------- data structures
def halfedge_onering():
    """Planar triangulation with the half-edge one-ring traversal of a vertex."""
    rr = np.random.default_rng(3)
    ang1 = np.arange(6) * np.pi / 3 + rr.normal(0, 0.12, 6)
    ang2 = np.arange(12) * np.pi / 6 + np.pi / 12 + rr.normal(0, 0.08, 12)
    P = np.vstack([[0, 0], np.c_[np.cos(ang1), np.sin(ang1)] * (1 + rr.normal(0, 0.08, (6, 1))),
                   np.c_[np.cos(ang2), np.sin(ang2)] * (2.0 + rr.normal(0, 0.1, (12, 1)))])
    tri = Delaunay(P)
    F = tri.simplices.copy()
    c = 0
    # build half-edges
    he_from, he_to, he_face = [], [], []
    for f, t in enumerate(F):
        a, b, cc = t
        u, v = P[b] - P[a], P[cc] - P[a]
        if u[0] * v[1] - u[1] * v[0] < 0:
            t = [a, cc, b]
            F[f] = t
        for k in range(3):
            he_from.append(F[f][k]); he_to.append(F[f][(k + 1) % 3]); he_face.append(f)
    he_from, he_to = np.array(he_from), np.array(he_to)
    nxt = np.array([3 * (h // 3) + (h % 3 + 1) % 3 for h in range(len(he_from))])
    key = {(a, b): h for h, (a, b) in enumerate(zip(he_from, he_to))}
    opp = np.array([key.get((b, a), -1) for a, b in zip(he_from, he_to)])
    # circulate around c: h = opp(next(next(h))) style (CCW): start with an outgoing half-edge
    h0 = int(np.nonzero(he_from == c)[0][0])
    ring, h = [], h0
    for _ in range(20):
        ring.append(h)
        h = opp[nxt[nxt[h]]]           # previous half-edge of the face, then its twin
        if h == h0 or h < 0:
            break
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.6))
    ax = axs[0]
    ax.triplot(P[:, 0], P[:, 1], F, color="0.75", lw=0.6)

    def arrow(h, color, lw=1.4, shrink=0.12, label=None, off=0.0):
        a, b = P[he_from[h]], P[he_to[h]]
        d = b - a
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        a2, b2 = a + shrink * d + off * nrm, b - shrink * d + off * nrm
        ax.annotate("", xy=b2, xytext=a2, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        if label:
            m = 0.5 * (a2 + b2) + 0.16 * nrm
            ax.text(*m, label, fontsize=7, color=color, ha="center", va="center")
    for k, h in enumerate(ring):
        arrow(h, "0.25", lw=1.0, label=str(k), off=0.05)
    for h in ring:
        f0 = F[h // 3]
        ax.fill(P[f0, 0], P[f0, 1], color="C0", alpha=0.10, lw=0)
    ax.plot(*P[c], "o", color="k", ms=4)
    ax.set_title(r"one-ring circulation: $h\leftarrow\mathrm{opp}(\mathrm{prev}(h))$", fontsize=8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    # valence histogram on the bunny
    ax = axs[1]
    for nm, col in [("bunny", "C0"), ("armadillo", "C1")]:
        V, Fm = gp.model(nm)
        val = np.bincount(gp.edges(Fm).ravel(), minlength=len(V))
        ax.hist(val, bins=np.arange(2.5, 13.5), alpha=0.55, color=col,
                label=f"{nm} (mean {val.mean():.2f})", density=True)
    ax.set_xlabel("vertex valence"); ax.set_ylabel("fraction of vertices")
    ax.legend(frameon=False, loc="upper left"); ax.set_xticks(range(3, 13)); ax.set_ylim(0, 0.95)
    gp.savefig(fig, "ch03_halfedge_valence.png")


if __name__ == "__main__":
    import sys
    todo = [gallery, representations, voxel_resolution, sampling, kdtree_figure,
            kdtree_benchmark, euler_figure, csg_figure, parametric_figure, halfedge_onering]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
