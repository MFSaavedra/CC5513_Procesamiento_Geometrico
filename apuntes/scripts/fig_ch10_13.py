"""Figures for chapters 10-13: simplification and subdivision, parameterization,
remeshing, deformation."""
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from scipy.spatial import Voronoi, cKDTree
import gp

gp.add_code_path("qem_demo")
gp.add_code_path("subdiv_demo")
rng = np.random.default_rng(10)


# =============================================================== chapter 10
def qem_levels():
    from qem_demo.methods import collapse_history
    V, F = gp.model("bunny", faces=20000)
    t0 = time.time()
    h = collapse_history(V, F, dict(cost="qem", placement="optimal"))
    print("QEM history", time.time() - t0, "s")
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.6))
    for ax, target in zip(axs, [len(F), 2000, 500, 150]):
        k = h.step_for_faces(target)
        Vs, Fs, _, _ = h.state(k)
        gp.render(ax, Vs, Fs, edges=len(Fs) < 3000, lw=0.15, smooth=False, view=(12, -70),
                  title=f"{len(Fs)} faces")
    gp.savefig(fig, "ch10_qem_levels.png")


def qem_compare():
    from qem_demo.methods import run_at_targets
    from qem_demo.metrics import geometric_error, SurfaceDistance
    V, F = gp.model("cow")
    targets = [3000, 1500, 800, 400, 200, 100]
    od = SurfaceDistance(V, F)
    res = {}
    for name, lab in [("qem", "QEM (Garland-Heckbert)"), ("length", "shortest edge first"),
                      ("clustering", "vertex clustering")]:
        out = run_at_targets(name, V, F, targets)
        res[lab] = ([len(f) for _, f, _ in out],
                    [geometric_error((V, F), v, f, n=15000, orig_dist=od)["mean"] for v, f, _ in out],
                    out)
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 1.9))
    ax = fig.add_axes([0.07, 0.2, 0.33, 0.7])
    for lab, (nf, err, _) in res.items():
        ax.loglog(nf, err, "o-", ms=3.5, label=lab)
    ax.set_xlabel("faces"); ax.set_ylabel("mean error [‰ of diagonal]")
    ax.legend(frameon=False, fontsize=6, loc="lower left"); ax.grid(alpha=0.3, which="both")
    for j, lab in enumerate(res):
        a = fig.add_axes([0.43 + 0.19 * j, 0.0, 0.19, 0.95])
        v, f, _ = res[lab][2][4]
        gp.render(a, v, f, edges=True, lw=0.12, smooth=False, view=(15, -50))
        print(lab, len(f), "faces")
        a.set_title(["QEM", "shortest edge", "clustering"][j], fontsize=7.5)
    gp.savefig(fig, "ch10_qem_compare.png")


def curve_subdivision():
    P0 = np.array([[0.1, 0.1], [0.25, 0.85], [0.5, 0.5], [0.85, 0.9], [0.95, 0.4], [0.6, 0.1]])

    def chaikin(P):
        Q = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        return np.array(Q)

    def cubic(P):
        Q = []
        n = len(P)
        for i in range(n):
            Q.append(0.125 * P[i - 1] + 0.75 * P[i] + 0.125 * P[(i + 1) % n])
            Q.append(0.5 * P[i] + 0.5 * P[(i + 1) % n])
        return np.array(Q)

    def four_point(P, w=1 / 16):
        Q = []
        n = len(P)
        for i in range(n):
            Q.append(P[i])
            Q.append((0.5 + w) * (P[i] + P[(i + 1) % n]) - w * (P[i - 1] + P[(i + 2) % n]))
        return np.array(Q)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.0))
    for ax, rule, t in [(axs[0], chaikin, "Chaikin (quadratic B-spline)"),
                        (axs[1], cubic, "cubic B-spline"),
                        (axs[2], four_point, "4-point (interpolating)")]:
        P = P0.copy()
        ax.plot(*np.vstack([P, P[:1]]).T, "o--", color="0.5", ms=3, lw=0.8)
        for lev, col in zip(range(1, 6), plt.cm.viridis(np.linspace(0.15, 0.85, 5))):
            P = rule(P)
            if lev in (1, 2, 5):
                ax.plot(*np.vstack([P, P[:1]]).T, "-", color=col, lw=1.0 if lev < 5 else 1.4,
                        label=f"level {lev}")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=8)
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    gp.savefig(fig, "ch10_curves.pdf")


def surface_subdivision():
    from subdiv_demo.mesh import PolyMesh
    from subdiv_demo.models import cube
    from subdiv_demo.schemes import subdivide
    V, Fq = cube()
    V = V - 0.5
    m = PolyMesh(V, Fq, name="cube")
    rows = [("catmull-clark", "Catmull-Clark"), ("doo-sabin", "Doo-Sabin"), ("loop", "Loop (triangulated cube)")]
    fig, axs = plt.subplots(3, 4, figsize=(gp.TEXTWIDTH, 4.6))
    for r, (scheme, name) in enumerate(rows):
        meshes, _ = subdivide(m, scheme, 4)
        for c, lev in enumerate([0, 1, 2, 4]):
            mm = meshes[lev]
            T = mm.triangles()
            faces = mm.face_list()
            ax = axs[r, c]
            # draw polygon outlines of the subdivided mesh on top of the shaded triangles
            gp.render(ax, mm.V, T, smooth=False, view=(25, -55), color=(0.75, 0.8, 0.9), scale=0.9,
                      center=np.zeros(3))
            if lev <= 2:
                R = gp.view_matrix(25, -55)
                Q = mm.V @ R.T
                Nf = []
                segs = []
                for f in faces:
                    pts = mm.V[f]
                    nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                    if (R @ nrm)[1] < 0:
                        q = Q[f][:, [0, 2]]
                        segs += [np.vstack([q[k], q[(k + 1) % len(q)]]) for k in range(len(q))]
                ax.add_collection(LineCollection(segs, colors="k", linewidths=0.35, zorder=5))
            ax.set_title(f"{name.split(' (')[0]}, level {lev}" if c else name, fontsize=7.5)
    plt.subplots_adjust(wspace=0.02, hspace=0.18)
    gp.savefig(fig, "ch10_subdivision.png")


def loop_on_model():
    from subdiv_demo.mesh import PolyMesh
    from subdiv_demo.schemes import subdivide
    V, F = gp.model("bunny", faces=400)
    m = PolyMesh(V, F.tolist(), name="bunny")
    meshes, _ = subdivide(m, "loop", 3)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.6))
    for ax, lev in zip(axs, range(4)):
        mm = meshes[lev]
        T = mm.triangles()
        gp.render(ax, mm.V, T, edges=lev < 2, lw=0.15, smooth=lev >= 2, view=(12, -70),
                  title=f"level {lev} ({len(T)} faces)")
        ax.title.set_fontsize(7.5)
    gp.savefig(fig, "ch10_loop_bunny.png")


# =============================================================== chapter 11
def projections():
    lam = np.radians(np.linspace(-180, 180, 13))
    phis = np.radians(np.array([-60, -30, 0, 30, 60]))
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH, 2.6))

    def merc(l, p):
        return l, np.log(np.tan(np.pi / 4 + p / 2))

    def lamb(l, p):
        return l, np.sin(p)
    for ax, proj, t in [(axs[0], merc, "Mercator (conformal)"), (axs[1], lamb, "Lambert cylindrical (equal-area)")]:
        for l in np.radians(np.arange(-180, 181, 30)):
            p = np.radians(np.linspace(-75, 75, 100))
            ax.plot(*proj(np.full_like(p, l), p), color="0.75", lw=0.5)
        for p in np.radians(np.arange(-75, 76, 15)):
            l = np.radians(np.linspace(-180, 180, 100))
            ax.plot(*proj(l, np.full_like(l, p)), color="0.75", lw=0.5)
        # Tissot indicatrices: small circles of angular radius rho on the sphere
        rho = np.radians(7)
        s = np.linspace(0, 2 * np.pi, 80)
        for l0 in np.radians(np.arange(-150, 151, 60)):
            for p0 in phis:
                c = np.array([np.cos(p0) * np.cos(l0), np.cos(p0) * np.sin(l0), np.sin(p0)])
                e1 = np.array([-np.sin(l0), np.cos(l0), 0])
                e2 = np.cross(c, e1)
                X = np.cos(rho) * c[None] + np.sin(rho) * (np.cos(s)[:, None] * e1 + np.sin(s)[:, None] * e2)
                pl = np.arcsin(np.clip(X[:, 2], -1, 1))
                ll = np.arctan2(X[:, 1], X[:, 0])
                x, y = proj(ll, pl)
                ax.fill(x, y, color="C3", alpha=0.6, lw=0)
        ax.set_title(t, fontsize=8); ax.set_aspect("equal"); ax.axis("off")
    gp.savefig(fig, "ch11_projections.pdf")


def disk_mesh(faces=24000):
    """Face patch of the Max Planck bust: a topological disk (chi = 1, one boundary loop)."""
    from scipy.sparse.csgraph import connected_components
    V, F = gp.model("planck", faces=faces)
    c = np.array([0.0, 0.12, 0.04])
    Fk = F[np.linalg.norm(V[F].mean(1) - c, axis=1) < 0.2]
    Vk, Fk = gp.remove_unreferenced(V, Fk)
    nc, lab = connected_components(gp.adjacency(Fk, len(Vk)))
    Fk = Fk[lab[Fk[:, 0]] == np.argmax(np.bincount(lab))]
    return gp.remove_unreferenced(Vk, Fk)


def circle_boundary(V, F):
    b = gp.boundary_loop(F)
    seg = np.linalg.norm(V[np.roll(b, -1)] - V[b], axis=1)
    t = np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    return b, np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]


def weights_matrix(V, F, kind):
    n = len(V)
    if kind == "uniform":
        W = gp.adjacency(F, n).astype(float)
    elif kind == "cotangent":
        C = gp.cotan_matrix(V, F)
        W = C - sp.diags(C.diagonal())
    else:  # mean value: w_ij = (tan(g/2) + tan(d/2)) / |v_i - v_j|
        rows, cols, vals = [], [], []
        A = gp.corner_angles(V, F)
        for k in range(3):
            i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
            ai = A[:, k]                       # angle at i in this triangle
            t = np.tan(ai / 2)
            for jj in (j, l):
                rows.append(i); cols.append(jj)
                vals.append(t / np.linalg.norm(V[jj] - V[i], axis=1))
        W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
    return W.tocsr()


def tutte(V, F, kind):
    n = len(V)
    b, ub = circle_boundary(V, F)
    W = weights_matrix(V, F, kind)
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    inner = np.setdiff1d(np.arange(n), b)
    U = np.zeros((n, 2))
    U[b] = ub
    Lii = L[inner][:, inner].tocsc()
    rhs = -L[inner][:, b] @ ub
    solve = spla.factorized(Lii)
    U[inner] = np.column_stack([solve(rhs[:, 0]), solve(rhs[:, 1])])
    return U


def jacobian_singular_values(V, F, U):
    """Singular values of the per-triangle affine map from the 3D triangle to the plane."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    nrm = np.cross(e1, e2)
    y = np.cross(nrm / np.linalg.norm(nrm, axis=1, keepdims=True), x)
    P = np.stack([np.c_[(e1 * x).sum(1), (e1 * y).sum(1)], np.c_[(e2 * x).sum(1), (e2 * y).sum(1)]], 2)
    Q = np.stack([U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]], 2)
    J = Q @ np.linalg.inv(P)
    s = np.linalg.svd(J, compute_uv=False)
    det = np.linalg.det(J)
    return s[:, 0], s[:, 1], det


def lscm(V, F):
    """Free-boundary conformal map: minimise sum_T A_T |grad v - n x grad u|^2, two pins."""
    n = len(V)
    G = gp.grad_matrix(V, F)
    N = gp.face_normals(V, F)
    a = gp.face_areas(V, F)
    m = len(F)
    # J = block-diagonal cross-product matrices [n]_x, so that (J g)_T = n_T x g_T
    entries = [(0, 1, -N[:, 2]), (0, 2, N[:, 1]), (1, 0, N[:, 2]), (1, 2, -N[:, 0]),
               (2, 0, -N[:, 1]), (2, 1, N[:, 0])]
    rows = [3 * np.arange(m) + r_ for r_, c_, v_ in entries]
    cols = [3 * np.arange(m) + c_ for r_, c_, v_ in entries]
    vals = [v_ for r_, c_, v_ in entries]
    J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * m, 3 * m)).tocsr()
    Wsq = sp.diags(np.repeat(np.sqrt(a), 3))
    A = sp.hstack([-Wsq @ J @ G, Wsq @ G]).tocsr()          # residual for x = [u; v]
    b = gp.boundary_loop(F)
    i0 = b[0]
    i1 = b[np.argmax(np.linalg.norm(V[b] - V[i0], axis=1))]
    pins = np.array([i0, i1, n + i0, n + i1])
    vals_p = np.array([0.0, 1.0, 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * n), pins)
    Af, Ap = A[:, free], A[:, pins]
    rhs = -Ap @ vals_p
    x = np.zeros(2 * n)
    x[pins] = vals_p
    x[free] = spla.spsolve((Af.T @ Af).tocsc(), Af.T @ rhs)
    return np.c_[x[:n], x[n:]]


def tutte_figure():
    V, F = disk_mesh()
    view = (10, 155)
    Us = {k: tutte(V, F, k) for k in ["uniform", "cotangent", "mean value"]}
    fig, axs = plt.subplots(2, 3, figsize=(gp.TEXTWIDTH, 3.9))
    Vs, Fs = disk_mesh(faces=5000)
    for c, (k, U) in enumerate(Us.items()):
        ax = axs[0, c]
        Uc = tutte(Vs, Fs, k)
        ax.triplot(Uc[:, 0], Uc[:, 1], Fs, lw=0.25, color="0.3")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{k} weights", fontsize=8)
        gp.render(axs[1, c], V, F, face_colors=gp.checker_colors(U, F, n=14), view=view)
        s1, s2, det = jacobian_singular_values(V, F, U)
        flips = min((det <= 0).sum(), (det > 0).sum())
        print(k, "flipped", flips, "median angle distortion", np.median(s1 / s2))
    plt.subplots_adjust(hspace=0.05, wspace=0.02)
    gp.savefig(fig, "ch11_tutte.png")


def free_boundary_figure():
    V, F = disk_mesh()
    U_fix = tutte(V, F, "cotangent")
    U_free = lscm(V, F)
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.95), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    Vs, Fs = disk_mesh(faces=5000)
    for ax, U, t in [(axs[0], tutte(Vs, Fs, "cotangent"), "fixed circle"),
                     (axs[1], lscm(Vs, Fs), "LSCM (free)")]:
        ax.triplot(U[:, 0], U[:, 1], Fs, lw=0.25, color="0.3")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=8)
    gp.render(axs[2], V, F, face_colors=gp.checker_colors(U_free, F, n=14), view=(10, 155),
              title="LSCM texture")
    axs[2].title.set_fontsize(8)
    ax = axs[3]
    for U, lab, col in [(U_fix, "fixed circle", "C0"), (U_free, "LSCM", "C3")]:
        s1, s2, det = jacobian_singular_values(V, F, U)
        a = gp.face_areas(V, F)
        area_d = np.log((s1 * s2) / np.average(s1 * s2, weights=a))
        ax.hist(np.log2(s1 / s2), bins=np.linspace(0, 1.0, 40), alpha=0.55, color=col, density=True,
                label=lab)
        print(lab, "median log2(s1/s2)", np.median(np.log2(s1 / s2)), "area spread (std log)", area_d.std())
    ax.set_xlabel(r"$\log_2(\sigma_1/\sigma_2)$"); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("angle distortion", fontsize=8)
    gp.savefig(fig, "ch11_free_boundary.png")


# =============================================================== chapter 12
def remesh_figure():
    import pymeshlab
    V, F = gp.model("planck", faces=4000)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F))
    L = gp.edge_lengths(V, F).mean()
    ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=pymeshlab.PureValue(0.8 * L))
    m = ms.current_mesh()
    V2, F2 = m.vertex_matrix(), m.face_matrix()
    ms2 = pymeshlab.MeshSet()
    ms2.add_mesh(pymeshlab.Mesh(V, F))
    ms2.meshing_isotropic_explicit_remeshing(iterations=10, adaptive=True,
                                             targetlen=pymeshlab.PureValue(0.7 * L))
    m3 = ms2.current_mesh()
    V3, F3 = m3.vertex_matrix(), m3.face_matrix()
    fig = plt.figure(figsize=(gp.TEXTWIDTH, 2.3))
    for j, (Vx, Fx, t) in enumerate([(V, F, "input (QEM-decimated)"), (V2, F2, "isotropic remeshing"),
                                     (V3, F3, "adaptive remeshing")]):
        ax = fig.add_axes([j * 0.215, 0.0, 0.215, 0.95])
        gp.render(ax, Vx, Fx, edges=True, lw=0.1, smooth=False, view=(5, -70), title=t)
        ax.title.set_fontsize(8)
    ax = fig.add_axes([0.70, 0.62, 0.29, 0.3])
    ax2 = fig.add_axes([0.70, 0.13, 0.29, 0.3])
    for Vx, Fx, lab, col in [(V, F, "input", "0.5"), (V2, F2, "isotropic", "C0")]:
        A = np.degrees(gp.corner_angles(Vx, Fx)).min(1)
        ax.hist(A, bins=np.linspace(0, 60, 31), alpha=0.6, color=col, density=True, label=lab)
        val = np.bincount(gp.edges(Fx).ravel())
        ax2.hist(val, bins=np.arange(2.5, 12.5), alpha=0.6, color=col, density=True, label=lab)
    ax.set_xlabel("min. angle per triangle [deg]", fontsize=7); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6)
    ax2.set_xlabel("valence", fontsize=7); ax2.set_yticks([])
    gp.savefig(fig, "ch12_remeshing.png")


def lloyd_figure():
    r = np.random.default_rng(3)
    P = r.random((60, 2))

    def cells(P):
        # mirror the sites across the four sides to bound the Voronoi cells to the unit square
        M = [P, np.c_[-P[:, 0], P[:, 1]], np.c_[2 - P[:, 0], P[:, 1]],
             np.c_[P[:, 0], -P[:, 1]], np.c_[P[:, 0], 2 - P[:, 1]]]
        vor = Voronoi(np.vstack(M))
        polys = []
        for i in range(len(P)):
            reg = vor.regions[vor.point_region[i]]
            polys.append(vor.vertices[reg])
        return polys

    def centroid(poly):
        x, y = poly[:, 0], poly[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        A = cr.sum() / 2
        return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A)

    def energy(P, polys, n=60000):
        Xs = np.random.default_rng(0).random((n, 2))
        d, _ = cKDTree(P).query(Xs)
        return (d ** 2).mean()
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    E = []
    snaps = {0: None, 3: None, 40: None}
    Q = P.copy()
    for it in range(41):
        polys = cells(Q)
        E.append(energy(Q, polys))
        if it in snaps:
            snaps[it] = (Q.copy(), polys)
        Q = np.array([centroid(p) for p in polys])
    for ax, (it, (Qs, polys)) in zip(axs, snaps.items()):
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="C0", linewidths=0.6))
        cs = np.array([centroid(p) for p in polys])
        ax.plot(Qs[:, 0], Qs[:, 1], "k.", ms=3)
        ax.plot(cs[:, 0], cs[:, 1], "+", color="C3", ms=3, mew=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Lloyd iteration {it}", fontsize=8)
    axs[3].plot(np.arange(len(E)), 1e3 * np.array(E), "C0.-", ms=3)
    axs[3].set_xlabel("iteration"); axs[3].set_ylabel(r"CVT energy ($\times 10^{-3}$)")
    axs[3].grid(alpha=0.3)
    gp.savefig(fig, "ch12_lloyd.pdf")


# =============================================================== chapter 13
def bernstein(n, i, t):
    from math import comb
    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def ffd_figure():
    th = np.linspace(0, 2 * np.pi, 400)
    r = 0.32 + 0.08 * np.cos(5 * th)
    shape = np.c_[0.5 + r * np.cos(th), 0.5 + r * np.sin(th)]
    eye = np.c_[0.5 + 0.06 * np.cos(th), 0.5 + 0.06 * np.sin(th)]
    n = 3
    g = np.linspace(0, 1, n + 1)
    Pc = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    Pd = Pc.copy()
    Pd[3, 3] += [0.25, 0.2]
    Pd[3, 2] += [0.18, 0.05]
    Pd[0, 3] += [-0.1, 0.12]
    Pd[1, 1] += [0.08, -0.08]

    def deform(X, P):
        s, t = X[:, 0], X[:, 1]
        out = np.zeros_like(X)
        for i in range(n + 1):
            for j in range(n + 1):
                out += (bernstein(n, i, s) * bernstein(n, j, t))[:, None] * P[i, j]
        return out
    fig, axs = plt.subplots(1, 2, figsize=(gp.TEXTWIDTH * 0.75, 2.4))
    for ax, P, t in [(axs[0], Pc, "lattice and object"), (axs[1], Pd, "moved control points")]:
        for i in range(n + 1):
            ax.plot(P[i, :, 0], P[i, :, 1], color="0.6", lw=0.7)
            ax.plot(P[:, i, 0], P[:, i, 1], color="0.6", lw=0.7)
        ax.plot(P[..., 0].ravel(), P[..., 1].ravel(), "o", color="C3", ms=3.5)
        S = deform(shape, P)
        E = deform(eye, P)
        ax.fill(S[:, 0], S[:, 1], color="C0", alpha=0.35)
        ax.plot(S[:, 0], S[:, 1], color="C0", lw=1)
        ax.fill(E[:, 0], E[:, 1], color="w")
        ax.plot(E[:, 0], E[:, 1], color="C0", lw=0.8)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=8)
    gp.savefig(fig, "ch13_ffd.pdf")


def mvc_weights(poly, X):
    """Mean value coordinates of points X w.r.t. polygon vertices (Floater 2003)."""
    n = len(poly)
    D = poly[None, :, :] - X[:, None, :]           # v_i - x
    r = np.linalg.norm(D, axis=2)
    Dn = np.roll(D, -1, axis=1)
    cross = D[..., 0] * Dn[..., 1] - D[..., 1] * Dn[..., 0]
    dot = (D * Dn).sum(2)
    alpha = np.arctan2(cross, dot)                  # angle at x between v_i and v_{i+1}
    t = np.tan(alpha / 2)
    w = (np.roll(t, 1, axis=1) + t) / r
    return w / w.sum(1, keepdims=True)


def wachspress_weights(poly, X):
    n = len(poly)
    W = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        A_abc = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def tri(p, q):
            return 0.5 * ((p[0] - X[:, 0]) * (q[1] - X[:, 1]) - (p[1] - X[:, 1]) * (q[0] - X[:, 0]))
        W.append(A_abc / (tri(a, b) * tri(b, c)))
    W = np.array(W).T
    return W / W.sum(1, keepdims=True)


def coords_figure():
    from matplotlib.path import Path
    convex = np.array([[0.1, 0.2], [0.6, 0.05], [0.95, 0.45], [0.7, 0.9], [0.2, 0.8]])
    nonconvex = np.array([[0.05, 0.1], [0.95, 0.1], [0.95, 0.9], [0.55, 0.9], [0.5, 0.35],
                          [0.4, 0.9], [0.05, 0.9]])
    x = np.linspace(0, 1, 300)
    XX, YY = np.meshgrid(x, x)
    X = np.c_[XX.ravel(), YY.ravel()]
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.1))
    for ax, poly, fun, vi, t in [(axs[0], convex, wachspress_weights, 2, "Wachspress, convex"),
                                 (axs[1], convex, mvc_weights, 2, "mean value, convex"),
                                 (axs[2], nonconvex, mvc_weights, 3, "mean value, non-convex")]:
        inside = Path(poly).contains_points(X)
        w = np.full(len(X), np.nan)
        w[inside] = fun(poly, X[inside])[:, vi]
        lim = np.nanmax(np.abs(w))
        im = ax.contourf(XX, YY, w.reshape(XX.shape), levels=np.linspace(-0.2, 1, 25), cmap="RdBu_r",
                         vmin=-1, vmax=1, extend="both")
        ax.contour(XX, YY, w.reshape(XX.shape), levels=[0], colors="k", linewidths=0.8)
        ax.plot(*np.vstack([poly, poly[:1]]).T, "k-", lw=1)
        ax.plot(*poly[vi], "o", color="C3", ms=5)
        print(t, "min weight", np.nanmin(w))
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=7.5)
    gp.savefig(fig, "ch13_coordinates.pdf")


def cylinder(n_around=40, n_len=80, r=0.12, length=1.0):
    th = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    z = np.linspace(-length / 2, length / 2, n_len)
    T, Z = np.meshgrid(th, z, indexing="ij")
    V = np.stack([r * np.cos(T), r * np.sin(T), Z], -1).reshape(-1, 3)
    F = gp.grid_faces(n_around, n_len, wrap_u=True)
    return V, F


def lbs_figure():
    V, F = cylinder()
    z = V[:, 2]
    w1 = np.clip(0.5 - z / 0.25, 0, 1)                # weight of bone 1 (z < 0), blend in |z|<0.125
    w2 = 1 - w1

    def lbs(R2, pivot=np.zeros(3)):
        X2 = (V - pivot) @ R2.T + pivot
        return w1[:, None] * V + w2[:, None] * X2
    from fig_ch06_08 import rot
    bend = lbs(rot([1, 0, 0], 100))
    twist = lbs(rot([0, 0, 1], 170))
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH * 0.8, 2.1))
    gp.render(axs[0], V, F, c=w2, cmap="coolwarm", clim=(0, 1), view=(10, -60), title="rest pose: $w_2$")
    gp.render(axs[1], bend, F, view=(0, -90), title="bend $100^\\circ$")
    gp.render(axs[2], twist, F, view=(10, -60), title="twist $170^\\circ$")
    for a in axs:
        a.title.set_fontsize(7.5)
    gp.savefig(fig, "ch13_lbs.png")


def attenuation_figure():
    V, F = gp.model("horse0", faces=16000)
    feet = np.argsort(V[:, 2])[:400]
    front = feet[np.argmax(V[feet, 0])] if False else None
    # handle: lowest vertex of the front-left leg (largest x among the low vertices)
    low = np.nonzero(V[:, 2] < V[:, 2].min() + 0.05)[0]
    h = low[np.argmin(V[low, 1])]                    # a front hoof (the head is towards -y)
    d_euc = np.linalg.norm(V - V[h], axis=1)
    d_geo = gp.heat_geodesic(V, F, h)
    R = 0.3
    disp = np.array([0.0, -0.14, 0.16])

    def falloff(d):
        s = np.clip(d / R, 0, 1)
        return (1 - s ** 2) ** 2
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 2.1))
    gp.render(axs[0], V, F, c=d_geo, cmap="viridis", view=(5, -75), title="geodesic distance")
    for ax, d, t in [(axs[1], d_euc, "Euclidean falloff"), (axs[2], d_geo, "geodesic falloff")]:
        W = falloff(d)
        gp.render(ax, V + W[:, None] * disp, F, c=W, cmap="YlOrRd", clim=(-0.05, 1.0), view=(5, -75), title=t)
    for a in axs:
        a.title.set_fontsize(8)
    gp.savefig(fig, "ch13_attenuation.png")


def bumpy_plane(n=50):
    x = np.linspace(0, 2, 2 * n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.zeros_like(X)
    for cx in np.linspace(0.3, 1.7, 5):
        for cy in [0.25, 0.5, 0.75]:
            Z += 0.12 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.006)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = gp.grid_faces(2 * n, n)
    return V, F, X.ravel()


def laplacian_edit(V, F, fixed, handle, target, w=1000.0):
    n = len(V)
    L = gp.cotan_matrix(V, F)          # uses rest-pose geometry
    delta = L @ V
    idx = np.r_[fixed, handle]
    pos = np.vstack([V[fixed], target])
    S = sp.coo_matrix((np.full(len(idx), w), (np.arange(len(idx)), idx)), shape=(len(idx), n))
    A = sp.vstack([L, S]).tocsr()
    B = np.vstack([delta, w * pos])
    solve = spla.factorized((A.T @ A).tocsc())
    Atb = A.T @ B
    return np.column_stack([solve(Atb[:, d]) for d in range(3)])


def arap(V, F, fixed, handle, target, iters=30, X0=None):
    n = len(V)
    C = gp.cotan_matrix(V, F)
    W = (C - sp.diags(C.diagonal())).tocoo()
    i, j, wij = W.row, W.col, W.data
    cons = np.r_[fixed, handle]
    cpos = np.vstack([V[fixed], target])
    free = np.setdiff1d(np.arange(n), cons)
    Lpos = -C                                  # positive semidefinite
    Lff = Lpos[free][:, free].tocsc()
    Lfc = Lpos[free][:, cons]
    solve = spla.factorized(Lff)
    X = V.copy() if X0 is None else X0.copy()
    X[cons] = cpos
    E0 = V[i] - V[j]
    for _ in range(iters):
        E1 = X[i] - X[j]
        S = np.zeros((n, 3, 3))
        np.add.at(S, i, wij[:, None, None] * E0[:, :, None] * E1[:, None, :])
        U, sig, Vt = np.linalg.svd(S)
        Rm = np.transpose(Vt, (0, 2, 1)) @ np.transpose(U, (0, 2, 1))
        bad = np.linalg.det(Rm) < 0
        if bad.any():
            U[bad, :, 2] *= -1
            Rm[bad] = np.transpose(Vt[bad], (0, 2, 1)) @ np.transpose(U[bad], (0, 2, 1))
        rhs = np.zeros((n, 3))
        np.add.at(rhs, i, 0.5 * wij[:, None] * ((Rm[i] + Rm[j]) @ E0[:, :, None])[:, :, 0])
        b = rhs[free] - Lfc @ cpos
        X[free] = np.column_stack([solve(b[:, d]) for d in range(3)])
    return X


def deformation_figure():
    V, F, xs = bumpy_plane()
    fixed = np.nonzero(xs < 0.12)[0]
    handle = np.nonzero(xs > 1.88)[0]
    target = V[handle] + np.array([-0.2, 0, 1.0])
    XL = laplacian_edit(V, F, fixed, handle, target)
    XA = arap(V, F, fixed, handle, target, iters=40, X0=XL)
    fig, axs = plt.subplots(1, 3, figsize=(gp.TEXTWIDTH, 1.9))
    col = np.zeros(len(V))
    col[fixed] = -1
    col[handle] = 1
    for ax, X, t in [(axs[0], V, "input"), (axs[1], XL, "Laplacian editing"),
                     (axs[2], XA, "as-rigid-as-possible")]:
        gp.render(ax, X, F, c=col if X is V else None, cmap="coolwarm", clim=(-1, 1),
                  view=(25, -60) if X is V else (6, 0), smooth=X is not V, color=(0.8, 0.82, 0.86))
        ax.set_title(t, fontsize=8)
    gp.savefig(fig, "ch13_laplacian_arap.png")
    # soft constraint weights on a horse leg
    Vh, Fh = gp.model("cat0", faces=12000)
    z = Vh[:, 2]
    tail = Vh[:, 0] - 0.25 * Vh[:, 2]
    handle = np.nonzero(tail > np.percentile(tail, 99.4))[0]
    fixed = np.nonzero(Vh[:, 0] < np.percentile(Vh[:, 0], 55))[0]
    target = Vh[handle] + np.array([0.0, 0.0, 0.35])
    fig, axs = plt.subplots(1, 4, figsize=(gp.TEXTWIDTH, 1.6))
    gp.render(axs[0], Vh, Fh, view=(10, -80), title="input")
    for ax, w in zip(axs[1:], [0.003, 0.03, 10.0]):
        X = laplacian_edit(Vh, Fh, fixed, handle, target, w=w)
        err = np.linalg.norm(X[handle] - target, axis=1).mean()
        print("soft weight", w, "handle error", err)
        gp.render(ax, X, Fh, view=(10, -80), title=f"$w={w:g}$")
    for a in axs:
        a.title.set_fontsize(7.5)
    gp.savefig(fig, "ch13_soft_weights.png")


if __name__ == "__main__":
    todo = [qem_levels, qem_compare, curve_subdivision, surface_subdivision, loop_on_model,
            projections, tutte_figure, free_boundary_figure, remesh_figure, lloyd_figure,
            ffd_figure, coords_figure, lbs_figure, attenuation_figure, deformation_figure]
    if len(sys.argv) > 1:
        todo = [f for f in todo if f.__name__ in sys.argv[1:]]
    for f in todo:
        t0 = time.time()
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
