#! /usr/bin/env python
import argparse
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py


def main(npy_paths: List[Path]):
    data = []
    for npy_path in npy_paths:
        x, y, z = np.split(np.loadtxt(str(npy_path)), 3, axis=1)
        data.append(go.Scatter3d(x=x, y=y, z=z))
    py.iplot(data, filename='pandas-brownian-motion-3d', height=700)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_paths', type=Path, nargs='*')
    main(**vars(parser.parse_args()))
