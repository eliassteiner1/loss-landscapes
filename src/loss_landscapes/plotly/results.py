import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.plotly.templates import use
use("mine_dark")

# ---- shared plot style (used by all landscape plots in this notebook) ----
CUTOFF = 10         # loss cutoff: values above are smoothly flattened (soft clip)
SOFT_K = 0.5        # width of the rounded transition at the cutoff (in log-loss units)
EPS = 1e-6          # avoid log(0)
Z_PAD = 0.02        # extra z-axis range (fraction of the span) so the surfaces don't z-fight with the box

CAMERA = dict(eye=dict(x=1.1, y=1.1, z=1.2))
LIGHTING = dict(
    ambient=0.4,      # base brightness everywhere (0–1). Lower = more contrast, deeper shadows
    diffuse=1.0,      # brightness from facing the light (0–1). Brings out the shape
    specular=0.3,     # shiny highlights (0–2)
    roughness=0.5,    # how spread out the highlights are (0–1). Lower = sharper, glossier
)
LIGHTPOSITION = dict(x=1000, y=1000, z=5000)   # light direction, in data-ish coordinates


def soft_clip(z, c, k=SOFT_K):
    """ smooth min(z, c) via log-sum-exp; k = width of the rounded transition """
    return -k * np.logaddexp(-z / k, -c / k)


def surface(r: dict) -> go.Surface:
    """ one landscape surface with the shared color axis + lighting """
    return go.Surface(
        x=r["alphas"], y=r["betas"], z=r["z"].T,
        coloraxis="coloraxis",
        lighting=LIGHTING,
        lightposition=LIGHTPOSITION,
    )


# load all results, keyed by (depth, method, skip) parsed from "resnet_<depth>_<skip>_<method>.npz"
results = {}
for path in sorted((PROJ_ROOT / "results").glob("resnet_*.npz")):
    _, depth, skip, method = path.stem.split("_")
    res = np.load(path)
    results[(depth, method, skip)] = dict(
        alphas=res["alphas"],
        betas=res["betas"],
        z=soft_clip(np.log(np.maximum(res["z"], EPS)), np.log(CUTOFF)),
    )

# shared z range over ALL results (both methods), so every panel in every figure is comparable
zrange = [min(r["z"].min() for r in results.values()), np.log(CUTOFF)]


def plot_landscapes(method: str):
    """ grid of surfaces for one direction method: one row per depth, columns = no skip | skip """
    
    # get all results for one method (all depths, skip and noskip)
    depths = sorted({d for d, m, _ in results if m == method})
    
    # sanity check if results are available
    depths = [d for d in depths if (d, method, "noskip") in results and (d, method, "skip") in results] 

    fig = make_subplots(
        rows=len(depths), cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}] for _ in depths],
        subplot_titles=[f"{d} | {s}" for d in depths for s in ("no skip", "skip")],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for i, depth in enumerate(depths, start=1):
        for j, skip in enumerate(["noskip", "skip"], start=1):
            fig.add_trace(surface(results[(depth, method, skip)]), row=i, col=j)

    pad = Z_PAD * (zrange[1] - zrange[0])
    fig.update_scenes(   # applies to every 3D subplot
        zaxis=dict(range=[zrange[0] - pad, zrange[1] + pad]),
        xaxis_title="α", yaxis_title="β", zaxis_title="log loss",
        camera=CAMERA,
    )
    fig.update_layout(
        title=f"directions: {method}",
        coloraxis=dict(colorscale="dense", cmin=zrange[0], cmax=zrange[1], showscale=False),
        margin=dict(t=60, b=20, l=20, r=20),
        height=450 * len(depths),
    )
    return fig
