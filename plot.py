#! /usr/bin/env python
from pathlib import Path
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import argparse


def main(npy_path: Path):
    x, y, z = np.split(np.loadtxt(str(npy_path)), 3, axis=1)
    trace = go.Scatter3d(x=x, y=y, z=z)
    py.iplot([trace], filename='pandas-brownian-motion-3d', height=700)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_path', type=Path)
    main(**vars(parser.parse_args()))
