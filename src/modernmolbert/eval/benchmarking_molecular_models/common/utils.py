import logging as log

from tqdm.auto import trange


def batch(data, n=128):
    data_len = len(data)
    for ndx in trange(0, data_len, n):
        batch = data[ndx : min(ndx + n, data_len)]
        if type(batch) is not list:
            batch = batch.tolist()
        # log.debug(f"Type: {type(batch)} Batch: {batch}")
        yield batch


try:
    import torch
    import subprocess
    import pandas as pd
    import sys
    from io import StringIO

    def cuda_available() -> bool:
        return torch.cuda.is_available()

    def get_least_utilized_gpu() -> int:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
        ).decode(sys.stdout.encoding)
        gpu_df = pd.read_csv(StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1)
        log.debug(f"GPU usage:\n{gpu_df}")
        gpu_df["memory.free"] = (
            gpu_df["memory.free"].astype("string").str.replace(" [MiB]", "", regex=False)
        )
        free_memory = pd.to_numeric(gpu_df["memory.free"], errors="coerce")
        idx = int(free_memory.idxmax())
        free_mib = float(free_memory.loc[idx])
        log.info(f"Returning GPU{idx} with {free_mib} free MiB")
        return idx

    def get_device(
        device: str | None = None, optimize_gpu_distribution: bool = True
    ) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                if optimize_gpu_distribution:
                    device = f"cuda:{get_least_utilized_gpu()}"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            log.warning(f"Using device: {device}")
        return torch.device(device)
except Exception as e:
    log.info(e)
