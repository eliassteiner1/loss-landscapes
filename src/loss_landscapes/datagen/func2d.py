import numpy as np


def _composite_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # low frequency: learned first
    smooth = 0.5 * (np.sin(1 * np.pi * x) * np.cos(0.8 * np.pi * y))
    # high frequency: learned late
    r = np.sqrt((x + 0.25)**2 + (y - 0.25)**2)
    ripple = 0.15 * np.sin(50 * r) * np.exp(-5 * r)
    # small, sharp feature        
    bump   = 0.8 * np.exp(-((x - 0.3)**2 + (y + 0.5)**2) / 0.01)
    # nearly a step: sharp edge 
    ridge  = 0.3 * np.abs(np.tanh(20 * (y - 0.8 * x + 0.2)))

    return smooth + ripple + bump + ridge


def generate_artificial_data(n_datapoints: int, added_noise: float = 5e-2) -> tuple[np.ndarray, np.ndarray]:
    
    # data generation
    print("> creating artificial dataset ...")
    xy = np.random.uniform(-1, 1, size=(n_datapoints, 2))
    z = _composite_func(xy[:, 0], xy[:, 1]) + np.random.normal(loc=0, scale=added_noise, size=n_datapoints)

    data = xy.astype(np.float32)
    targets = z.astype(np.float32)
    
    return data, targets


