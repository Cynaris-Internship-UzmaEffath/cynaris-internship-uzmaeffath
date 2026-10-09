from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Data_dir=Path("data")
raw_data_dir=Data_dir/"raw"
processed_data_dir=Data_dir/"processed"
def check_data_folders():
    """"create the data folders required if they do not exist..."""
    raw_data_dir.mkdir(parents=True,exist_ok=True)
    processed_data_dir.mkdir(parents=True,exist_ok=True)
    print("data folders are ready:\n")
    print(f"raw data folder:{raw_data_dir}")
    print(f"processed data folder:{processed_data_dir}")

if __name__=="__main__":
    check_data_folders()
