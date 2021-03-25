import argparse
from typing import Generator, List

from tqdm import tqdm
from wandb.apis.public import Sweep
import pandas as pd

import wandb


def main(sweeps: List[str], run_names: List[str]):
    def get_run_dfs(sweep: Sweep) -> Generator[pd.DataFrame, None, None]:
        for run in tqdm(sweep.runs):
            breakpoint()
            df = (
                run.history()
                .loc[pd.IndexSlice[:, [*keys, step_key]]]
                .rename(columns=columns)
            )
            df = df.loc[(start_step <= df[step_key]) & (df[step_key] <= stop_step)]
            df = df.melt(id_vars=["_step"], value_vars=metrics)
            df["run"] = run.id
            yield df

    def get_sweep_dfs() -> Generator[pd.DataFrame, None, None]:
        api = wandb.Api()
        assert len(sweeps) == len(run_names)
        for sweep_id, name in tqdm([*zip(sweeps, run_names)]):
            sweep = api.sweep(sweep_id)
            df = pd.concat([*get_run_dfs(sweep)])
            df["name"] = name
            yield df

    source = pd.concat([*get_sweep_dfs()])


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--sweeps", nargs="*")
    PARSER.add_argument("--run_names", nargs="*")
    main(**vars(PARSER.parse_args()))
