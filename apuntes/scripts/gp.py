"""Shared utilities for the figures of the CC5513 lecture notes.

Everything here is plain NumPy/SciPy/Matplotlib so the figure scripts run
anywhere. The course demos under ``codigo/`` are imported by the chapter
scripts that need them (ICP, global registration, QEM, subdivision).

Sign conventions used in the notes
----------------------------------
* ``cotan_matrix`` returns the cotangent stiffness matrix C, which is
  negative semidefinite: C_ij = (cot a_ij + cot b_ij)/2, C_ii = -sum_j C_ij.
  With the lumped mass matrix M, the Laplace-Beltrami operator is
  Delta = M^{-1} C (analyst's convention, Delta = div grad).
* ``laplacian`` returns L = -C (positive semidefinite), the convention of the
  spectral chapters: L phi = lambda M phi with lambda >= 0.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.dirname(HERE)
REPO = os.path.dirname(NOTES)
FIGDIR = os.path.join(NOTES, "figs")
CODE = os.path.join(REPO, "codigo")
MODELS = os.path.join(REPO, "modelos")
os.makedirs(FIGDIR, exist_ok=True)
sys.dont_write_bytecode = True

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TEXTWIDTH = 6.1  # inches, matches \textwidth in the LaTeX preamble


def savefig(fig, name, dpi=300):
    """Save a figure into apuntes/figs and close it."""
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", os.path.relpath(path, REPO))


def add_code_path(sub):
    """Make a course demo package importable (e.g. 'qem_demo')."""
    p = os.path.join(CODE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------------
# I/O and normalisation
# ----------------------------------------------------------------------------
def load_mesh(path):
    """Read a triangle mesh from OBJ or OFF. Polygons are fan-triangulated."""
    ext = os.path.splitext(path)[1].lower()
    V, F = [], []
    if ext == ".obj":
        with open(path) as fh:
            for line in fh:
                if line.startswith("v "):
                    V.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                    idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                    for k in range(1, len(idx) - 1):
                        F.append([idx[0], idx[k], idx[k + 1]])
    elif ext == ".off":
        with open(path) as fh:
            tok = fh.read().split()
        assert tok[0].upper().endswith("OFF")
        nv, nf = int(tok[1]), int(tok[2])
        p = 4
        V = np.array(tok[p:p + 3 * nv], float).reshape(nv, 3)
        p += 3 * nv
        F = []
        for _ in range(nf):
            k = int(tok[p])
            idx = [int(x) for x in tok[p + 1:p + 1 + k]]
            p += 1 + k
            for j in range(1, k - 1):
                F.append([idx[0], idx[j], idx[j + 1]])
    else:
        raise ValueError(ext)
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    return remove_unreferenced(V, F)


def remove_unreferenced(V, F):
    used = np.unique(F)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def normalize(V, mode="diag"):
    """Centre at the bounding-box centre and scale to unit diagonal (or area)."""
    c = 0.5 * (V.max(0) + V.min(0))
    V = V - c
    if mode == "diag":
        V = V / np.linalg.norm(V.max(0) - V.min(0))
    return V


def zup(V):
    """Rotate a y-up model so that its up axis becomes +z."""
    return np.c_[V[:, 0], -V[:, 2], V[:, 1]]


def model(name, faces=None):
    """Load one of the repository meshes by short name, z-up, unit diagonal.

    faces: optional target face count (QEM decimation with Open3D) to keep the
    software renderer fast.
    """
    paths = {
        "bunny": "codigo/icp_demo/data/stanford-bunny.obj",
        "armadillo": "codigo/icp_demo/data/armadillo.obj",
        "cow": "codigo/icp_demo/data/cow.obj",
        "fandisk": "codigo/icp_demo/data/fandisk.obj",
        "planck": "codigo/icp_demo/data/max-planck.obj",
        "nefertiti": "codigo/icp_demo/data/nefertiti.obj",
        "happy": "codigo/icp_demo/data/happy.obj",
        "spot": "codigo/subdiv_demo/data/spot.obj",
        "suzanne": "codigo/subdiv_demo/data/suzanne.obj",
        "elephant": "modelos/elephant-50kv.off",
    }
    if name.startswith(("cat", "horse")):
        path = os.path.join(MODELS, name + ".off")
    else:
        path = os.path.join(REPO, paths[name])
    V, F = load_mesh(path)
    if name in ("bunny", "armadillo", "happy", "spot", "suzanne", "planck", "elephant", "cow"):
        V = zup(V)
    if faces is not None and len(F) > faces:
        V, F = decimate(V, F, faces)
    return normalize(V), F


def decimate(V, F, faces):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(int(faces))
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


# ----------------------------------------------------------------------------
# Basic mesh quantities
# ----------------------------------------------------------------------------
def face_normals(V, F, unit=True):
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if unit:
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    return N


def face_areas(V, F):
    return 0.5 * np.linalg.norm(face_normals(V, F, unit=False), axis=1)


def corner_angles(V, F):
    """Interior angle at each corner, shape (nF, 3)."""
    A = np.zeros(F.shape)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, v = V[b] - V[a], V[c] - V[a]
        cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        A[:, k] = np.arccos(np.clip(cs, -1, 1))
    return A


def vertex_normals(V, F, weight="area"):
    if weight == "area":
        Nf = face_normals(V, F, unit=False)
        W = np.ones((len(F), 3))
    else:
        Nf = face_normals(V, F)
        W = corner_angles(V, F) if weight == "angle" else np.ones((len(F), 3))
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], Nf * W[:, k:k + 1])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def edges(F):
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    return np.unique(E, axis=0)


def edge_lengths(V, F):
    E = edges(F)
    return np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)


def adjacency(F, n):
    E = edges(F)
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


def boundary_edges(F):
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    return Eu[cnt == 1]


def boundary_loop(F):
    """Ordered vertex indices of the (longest) boundary loop."""
    B = boundary_edges(F)
    nbr = {}
    for a, b in B:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    loops, seen = [], set()
    for s in nbr:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [x for x in nbr[cur] if x != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == s:
                break
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return np.array(max(loops, key=len))


def euler_characteristic(V, F):
    return len(np.unique(F)) - len(edges(F)) + len(F)


# ----------------------------------------------------------------------------
# Synthetic meshes
# ----------------------------------------------------------------------------
def icosphere(level=3):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(level):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        inv = inv.ravel()
        M = V[Eu[:, 0]] + V[Eu[:, 1]]
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        m = len(V) + inv.reshape(3, -1).T
        V = np.vstack([V, M])
        a, b, c = F.T
        ab, bc, ca = m.T
        F = np.vstack([np.c_[a, ab, ca], np.c_[b, bc, ab], np.c_[c, ca, bc], np.c_[ab, bc, ca]])
    return V, F


def grid_faces(nu, nv, wrap_u=False, wrap_v=False):
    idx = np.arange(nu * nv).reshape(nu, nv)
    F = []
    for i in range(nu if wrap_u else nu - 1):
        for j in range(nv if wrap_v else nv - 1):
            a, b = idx[i, j], idx[(i + 1) % nu, j]
            c, d = idx[(i + 1) % nu, (j + 1) % nv], idx[i, (j + 1) % nv]
            F += [[a, b, c], [a, c, d]]
    return np.array(F)


def torus(R=1.0, r=0.35, nu=64, nv=24):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([(R + r * np.cos(W)) * np.cos(U), (R + r * np.cos(W)) * np.sin(U),
                  r * np.sin(W)], -1).reshape(-1, 3)
    return V, grid_faces(nu, nv, True, True)


# ----------------------------------------------------------------------------
# Discrete differential operators
# ----------------------------------------------------------------------------
def cotan_matrix(V, F):
    """Cotangent stiffness matrix C (negative semidefinite, rows sum to 0)."""
    n = len(V)
    rows, cols, vals = [], [], []
    for k in range(3):
        o, i, j = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, v = V[i] - V[o], V[j] - V[o]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-300)
        rows += [i, j]
        cols += [j, i]
        vals += [0.5 * cot, 0.5 * cot]
    C = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return (C - sp.diags(np.asarray(C.sum(1)).ravel())).tocsr()


def laplacian(V, F):
    """Positive semidefinite cotangent Laplacian L = -C."""
    return (-cotan_matrix(V, F)).tocsr()


def mass_matrix(V, F, kind="barycentric"):
    n = len(V)
    m = np.zeros(n)
    if kind == "barycentric":
        a = face_areas(V, F)
        for k in range(3):
            np.add.at(m, F[:, k], a / 3)
    else:  # Voronoi (non-obtuse formula of Meyer et al.)
        for k in range(3):
            i, j, o = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]

            def cot(u, v):
                return (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
            a2 = ((V[j] - V[i]) ** 2).sum(1)
            b2 = ((V[o] - V[i]) ** 2).sum(1)
            np.add.at(m, i, (a2 * cot(V[i] - V[o], V[j] - V[o])
                             + b2 * cot(V[i] - V[j], V[o] - V[j])) / 8)
    return sp.diags(m).tocsr()


def uniform_laplacian(F, n):
    """L_u = D^{-1} A - I (negative semidefinite, rows sum to 0)."""
    A = adjacency(F, n)
    deg = np.asarray(A.sum(1)).ravel()
    return (sp.diags(1.0 / deg) @ A - sp.eye(n)).tocsr()


def eigs(L, M, k, sigma=-1e-8):
    """k smallest eigenpairs of L phi = lambda M phi (shift-invert)."""
    lam, Phi = spla.eigsh(L, k=k, M=M, sigma=sigma, which="LM")
    o = np.argsort(lam)
    lam, Phi = lam[o], Phi[:, o]
    lam[0] = max(lam[0], 0.0)
    return lam, Phi


def angle_defect(V, F, area=True):
    """Discrete Gaussian curvature (2 pi - sum of angles) / A_i."""
    s = np.zeros(len(V))
    A = corner_angles(V, F)
    for k in range(3):
        np.add.at(s, F[:, k], A[:, k])
    K = 2 * np.pi - s
    B = np.unique(boundary_edges(F))
    if len(B):
        K[B] = np.pi - s[B]
    if area:
        K = K / mass_matrix(V, F).diagonal()
    return K


def mean_curvature_normal(V, F):
    """H n = -1/2 Delta x with Delta = M^{-1} C."""
    C, M = cotan_matrix(V, F), mass_matrix(V, F)
    return -0.5 * (C @ V) / M.diagonal()[:, None]


def mean_curvature(V, F):
    Hn = mean_curvature_normal(V, F)
    N = vertex_normals(V, F)
    return (Hn * N).sum(1)


def grad_matrix(V, F):
    """Per-face gradient of piecewise-linear functions: (3 nF) x nV sparse."""
    N = face_normals(V, F, unit=False)
    A2 = np.linalg.norm(N, axis=1)
    N = N / A2[:, None]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        e = V[l] - V[j]                      # edge opposite to vertex i
        g = np.cross(N, e) / A2[:, None]     # grad of hat function phi_i
        for d in range(3):
            rows.append(3 * np.arange(len(F)) + d)
            cols.append(i)
            vals.append(g[:, d])
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(3 * len(F), len(V))).tocsr()


def heat_geodesic(V, F, sources, t=None):
    """Geodesic distance by the heat method (Crane et al. 2013)."""
    sources = np.atleast_1d(sources)
    C, M = cotan_matrix(V, F), mass_matrix(V, F)
    h = edge_lengths(V, F).mean()
    t = 4 * h * h if t is None else t
    d = np.zeros(len(V))
    d[sources] = 1.0
    u = spla.spsolve((M - t * C).tocsc(), d)
    G = grad_matrix(V, F)
    g = (G @ u).reshape(-1, 3)
    X = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
    a = face_areas(V, F)
    div = -(G.T @ (X * a[:, None]).ravel())      # integrated divergence
    # weak Poisson problem C phi = div (C is the negative semidefinite stiffness matrix)
    phi = spla.spsolve((C - 1e-10 * M).tocsc(), div)
    phi = phi - phi[sources].min()
    return phi


def dijkstra_geodesic(V, F, sources):
    E = edges(F)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    n = len(V)
    G = sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                      shape=(n, n)).tocsr()
    return dijkstra(G, indices=np.atleast_1d(sources))


# ----------------------------------------------------------------------------
# Software renderer (orthographic, painter's algorithm, Lambert shading)
# ----------------------------------------------------------------------------
def view_matrix(elev=20, azim=-60):
    """Rotation taking world (z-up) to view space; the camera looks along +y."""
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


LIGHT = np.array([-0.45, -0.85, 0.55])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def render(ax, V, F, c=None, view=(20, -60), cmap="viridis", clim=None, color=(0.78, 0.80, 0.84),
           edges=False, lw=0.15, ec=(0.15, 0.15, 0.2), cull=True, shade=0.8, smooth=True,
           title=None, alpha=1.0, center=None, scale=None, zorder=1, face_colors=None):
    """Render a triangle mesh into a 2D matplotlib axis.

    c: per-vertex (len V) or per-face (len F) scalars, or None for plain shading.
    face_colors: explicit RGB(A) per face (overrides c).
    """
    R = view_matrix(*view) if np.ndim(view) == 1 else np.asarray(view)
    Vc = V if center is None else V - center
    Q = Vc @ R.T
    Nf = face_normals(V, F) @ R.T
    vis = Nf[:, 1] < 0 if cull else np.ones(len(F), bool)
    lam = np.clip(Nf @ LIGHT, 0, 1)
    if not cull:
        lam = np.abs(Nf @ LIGHT)
    inten = (1 - shade) + shade * lam
    depth = Q[F, 1].mean(1)
    order = np.argsort(-depth[vis])
    idx = np.nonzero(vis)[0][order]
    P2 = Q[:, [0, 2]]
    if face_colors is not None:
        fc = np.asarray(face_colors)[idx]
        if fc.shape[1] == 3:
            fc = np.c_[fc, np.ones(len(fc))]
        fc[:, :3] *= inten[idx, None]
    elif c is None and smooth and not edges:
        # Gouraud-shaded grey via per-vertex intensity
        Nv = vertex_normals(V, F) @ R.T
        iv = (1 - shade) + shade * np.clip(Nv @ LIGHT, 0, 1)
        tri = Triangulation(P2[:, 0], P2[:, 1], F[idx])
        base = np.array(color)
        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list("shade", [base * 0.0 + 0.05, base], 256)
        ax.tripcolor(tri, iv, shading="gouraud", cmap=cm, vmin=0, vmax=1, zorder=zorder,
                     rasterized=True)
        _finish(ax, P2, title, scale)
        return None
    else:
        if c is None:
            fc = np.tile(np.r_[color, 1.0], (len(idx), 1))
        else:
            c = np.asarray(c, float)
            cf = c[F].mean(1) if len(c) == len(V) else c
            lo, hi = (np.nanpercentile(cf, 1), np.nanpercentile(cf, 99)) if clim is None else clim
            cm = plt.get_cmap(cmap)
            fc = cm(np.clip((cf[idx] - lo) / max(hi - lo, 1e-300), 0, 1))
        fc = np.array(fc, float)
        fc[:, :3] *= inten[idx, None]
    fc[:, 3] = alpha
    pc = PolyCollection(P2[F[idx]], facecolors=fc, edgecolors=ec if edges else fc,
                        linewidths=lw if edges else 0.05, zorder=zorder, rasterized=True)
    ax.add_collection(pc)
    _finish(ax, P2, title, scale)
    return pc


def _finish(ax, P2, title, scale):
    lo, hi = P2.min(0), P2.max(0)
    c = 0.5 * (lo + hi)
    r = 0.5 * (hi - lo).max() * 1.04 if scale is None else scale
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def render_points(ax, P, c=None, N=None, view=(20, -60), s=1.0, cmap="viridis", clim=None,
                  color=(0.35, 0.40, 0.55), title=None, cull=False, shade=0.75, alpha=1.0,
                  zorder=2, scale=None):
    R = view_matrix(*view)
    Q = P @ R.T
    vis = np.ones(len(P), bool)
    if N is not None and cull:
        vis = (N @ R.T)[:, 1] < 0
    order = np.argsort(-Q[vis, 1])
    idx = np.nonzero(vis)[0][order]
    if c is None:
        if N is not None:
            lam = np.abs((N @ R.T) @ LIGHT)
            col = np.array(color)[None, :] * ((1 - shade) + shade * lam[idx, None])
            col = np.clip(col / max(np.array(color).max(), 1e-9) * np.array(color).max(), 0, 1)
        else:
            col = [color]
        ax.scatter(Q[idx, 0], Q[idx, 2], s=s, c=col, linewidths=0, zorder=zorder, alpha=alpha,
                   rasterized=True)
    else:
        c = np.asarray(c)
        lo, hi = (np.percentile(c, 1), np.percentile(c, 99)) if clim is None else clim
        ax.scatter(Q[idx, 0], Q[idx, 2], s=s, c=c[idx], cmap=cmap, vmin=lo, vmax=hi,
                   linewidths=0, zorder=zorder, alpha=alpha, rasterized=True)
    _finish(ax, Q[:, [0, 2]], title, scale)


def project(P, view):
    R = view_matrix(*view)
    Q = P @ R.T
    return Q[:, [0, 2]]


def checker_colors(UV, F, n=10, c0=(0.95, 0.95, 0.95), c1=(0.75, 0.15, 0.15)):
    """Per-face colours of a checkerboard texture defined in the UV domain."""
    uv = UV[F].mean(1)
    uv = (uv - UV.min(0)) / np.maximum(UV.max(0) - UV.min(0), 1e-12)
    k = (np.floor(uv[:, 0] * n) + np.floor(uv[:, 1] * n)) % 2
    return np.where(k[:, None] > 0, np.array(c1), np.array(c0))


def colorbar(fig, ax, mappable=None, cmap="viridis", clim=(0, 1), label=None, shrink=0.7,
             orientation="vertical"):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    if mappable is None:
        mappable = cm.ScalarMappable(norm=mcolors.Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(mappable, ax=ax, shrink=shrink, orientation=orientation, pad=0.02)
    if label:
        cb.set_label(label)
    cb.ax.tick_params(labelsize=7)
    return cb


def closed_sdf_grid(V, F, n=64, pad=0.05):
    """Signed distance of a (hole-filled) mesh sampled on an n^3 grid (Open3D)."""
    import open3d as o3d
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F))
    ms.meshing_close_holes(maxholesize=1000)
    m = ms.current_mesh()
    Vc, Fc = m.vertex_matrix(), m.face_matrix()
    tm = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(Vc.astype(np.float32)),
                                     o3d.core.Tensor(Fc.astype(np.int32)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    lo, hi = V.min(0) - pad, V.max(0) + pad
    axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    d = scene.compute_signed_distance(o3d.core.Tensor(G)).numpy()
    return axes, d.reshape(n, n, n)


def signed_volume(V, F):
    return np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6


def iso_surface(S, axes, level=0.0):
    """Marching cubes on a grid sampled at `axes`; faces oriented outward."""
    from skimage.measure import marching_cubes
    lo = np.array([a[0] for a in axes])
    h = tuple(a[1] - a[0] for a in axes)
    Vm, Fm, _, _ = marching_cubes(S, level, spacing=h)
    Vm = Vm + lo
    if signed_volume(Vm, Fm) < 0:
        Fm = Fm[:, ::-1]
    return Vm, np.ascontiguousarray(Fm)
