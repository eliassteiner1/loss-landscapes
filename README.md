# Abstract

this is a short exploration into visualizing loss landscapes for neural networks. Inspiration for this project:
- [LossLandscape Website](https://losslandscape.com/)
- [Paper "Visualizing the Loss Landscapes of Neural Nets](https://arxiv.org/pdf/1712.09913)

# Setup

1. Install uv package manager from `https://docs.astral.sh/uv/getting-started/installation/`
2. clone github repo
    - `git clone https://github.com/eliassteiner1/loss-landscapes.git`
    - `cd loss-landscapes`
3. sync environment from lockfile: `uv sync`
    > Note: this will install the basic `torch`dependency built for cpu. If your machine supports CUDA, use the torch wheel built against cuda 13.4 already setup in pyproject.toml with `uv sync --group cu134`
    > if you need a different cuda version or pytorch build, replace the index and source configuration for torch and torchvision in pyproject.toml
4. Enjoy the show: open and run the notebook `notebooks/presentation.ipynb`

# TODOS

- asdf

# Ideas

- ASFD