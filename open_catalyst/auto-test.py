import json
import itertools
import os
from subprocess import run
from typing import Dict, Optional, List
from copy import deepcopy

from tqdm import tqdm


"""

auto-test.py

This script will automatically run through a number of
configurations for the sake of profiling OpenCatalyst
performance as a function of multiinstance training
variables.
"""


default_matrix = {
    "threads": [
        8,
        12,
        16,
        20,
    ],
    "mpi_workers": [1, 4, 8, 16],
    "batch_size": [32, 64, 128, 256, 512],
    "loader_workers": [
        0,
        1,
    ],
}

# quick job to make sure stuff works
test_matrix = {
    "threads": [
        12,
        16,
    ],
    "mpi_workers": [
        8,
    ],
    "batch_size": [
        32,
    ],
    "loader_workers": [
        1,
    ],
}


def load_previous() -> List[Dict[str, int]]:
    with open("timing_result.json", "r") as read_file:
        data = json.load(read_file)
    return data


def list_differences(
    previous_data: List[Dict[str, int]], default_matrix: List[Dict[str, int]]
) -> List[Dict[str, int]]:
    previous = deepcopy(previous_data)
    # get rid of the time key to make sure the dicts are the same
    for subdict in previous:
        del subdict["time"]
    # generate a hash for each subdictionary
    prev_hashes = [hash(frozenset(subdict.items())) for subdict in previous]
    def_hashes = [
        hash(frozenset(subdict.items())) for subdict in default_matrix
    ]
    return_list = []
    # iterate through and for every hash that hasn't been done,
    # append the configuration
    for index, entry in enumerate(def_hashes):
        if entry not in prev_hashes:
            return_list.append(default_matrix[index])
    return return_list


def launch(config: Dict[str, int]):
    threads = config.get("threads")
    mpi_workers = config.get("mpi_workers")
    # set the thread count
    os.environ["OMP_NUM_THREADS"] = str(threads)
    cmd = [
        "torchrun",
        f"--nproc_per_node={mpi_workers}",
        "main.py",
        "--distributed",
        "--distributed-backend",
        "mpi",
        "--cpu",
        "--mode",
        "train",
        "--config",
        "profile-config.yml",
    ]
    result = run(cmd, capture_output=True)
    stdout = result.stdout.decode("utf-8")
    timing = stdout.split()[-1].strip().split()[-1]
    return timing


def product_dict(**kwargs) -> Dict[str, int]:
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def write_config(config, template) -> None:
    # writes the configuration YAML for the run
    relevant = {
        key: config.get(key) for key in ["batch_size", "loader_workers"]
    }
    with open("profile-config.yml", "w+") as write_file:
        write_file.write(template.format_map(relevant))


def read_template() -> str:
    # reads a template configuration file
    with open("profile-template.yml") as read_file:
        template = read_file.read()
    return template


def main(test_matrix: List[Dict[str, int]], data=None):
    # read in the template
    template = read_template()
    if data is None:
        data = []
    for combo in tqdm(iterator):
        write_config(combo, template)
        # this is just to catch failures
        try:
            timing = launch(combo)
            combo["time"] = timing
        except:
            combo["time"] = "nan"
        # update the timings data, and dump it to json
        data.append(combo)
        with open("timing_result.json", "w+") as write_file:
            json.dump(data, write_file)


if __name__ == "__main__":
    restart = True
    # this generates a list of configurations to run
    iterator = list(product_dict(**default_matrix))
    # if we restart, we don't want to rerun the configurations
    # that have already been run
    if restart:
        previous_runs = load_previous()
        iterator = list_differences(previous_runs, iterator)
    else:
        previous_runs = None
    main(iterator, previous_runs)
