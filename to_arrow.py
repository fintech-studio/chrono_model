import math
from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from datetime import datetime
import os

def transform_iso_to_datetime(unix_timestamp: Union[str, int]) -> str:
    """
    Transform a Unix timestamp to '2000-01-01 00:00' format.

    Args:
        unix_timestamp: A Unix timestamp (e.g., 1589290200) as string or integer

    Returns:
        A string in the format '2000-01-01 00:00'
    """
    # Convert to integer if it's a string
    if isinstance(unix_timestamp, str):
        unix_timestamp = int(unix_timestamp)
    # Parse the Unix timestamp to datetime
    dt = datetime.fromtimestamp(unix_timestamp)

    # Format the datetime object to the desired format
    formatted_date = dt.strftime('%Y-%m-%d %H:%M')

    return formatted_date


def convert_to_arrow(
        path: Union[str, Path],
        compression: str = "lz4",
        dataset: List = None
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    # assert isinstance(time_series, list) or (
    #         isinstance(time_series, np.ndarray) and
    #         time_series.ndim == 2
    # )

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    current_file = ''
    files = os.listdir("./datasets_test")
    dataset = []
    validset = []
    try:
        for file in files:
            current_file = file
            df = pd.read_csv(f"./datasets_test/{file}")
            first_timestamp = df["time"][0]
            first_date = transform_iso_to_datetime(first_timestamp)
            start = np.datetime64(first_date, "s")
            # group = Path(file).stem
            # print(group)
            # Generate 20 random time series of length 1024
            time_series = df["close"].dropna().ffill().to_numpy()

            # length_80 = math.ceil(len(time_series) * 0.8)
            # ds_ts = time_series[:length_80]
            # valid_ts = time_series[length_80:]
            #
            # _80_timestamp = df["time"][length_80]
            # _80_date = transform_iso_to_datetime(_80_timestamp)

            dataset.append({"start": start, "target": time_series})
            # validset.append({"start": _80_date, "target": valid_ts})

            # Convert to GluonTS arrow format

        convert_to_arrow("./arrows/dataset.arrow", dataset=dataset)
        convert_to_arrow("./arrows/valid.arrow", dataset=validset)
    except Exception as e:
        print(e)
        print(current_file)

    # Test the transform_iso_to_datetime function


'''



'''
    # Generate 20 random time series of length 1024
    # time_series = [np.random.randn(1024) for i in range(20)]
    #
    # # Convert to GluonTS arrow format
    # convert_to_arrow("./noise-data.arrow", time_series=time_series)

'''
    $env:CUDA_VISIBLE_DEVICES = "0";
    python train.py --config config.yaml `
    --model-id amazon/chronos-t5-small `
    --no-random-init `
    --max-steps 10 `
    --learning-rate 0.001
       
    python train.py --config config.yaml --model-id amazon/chronos-t5-small --no-random-init --max-steps 10 --learning-rate 0.001
    python train.py --config config.yaml --model-id amazon/chronos-t5-small --no-random-init --learning-rate 0.001
'''