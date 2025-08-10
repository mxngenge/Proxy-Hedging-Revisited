# utils.py
import os
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def read_xlsx_file(file_name):  ### For bloomberg data
    data_folder = os.path.join(PROJECT_ROOT, "Clean_Dat") 
    file_path = os.path.join(data_folder, file_name)

    # Read Excel file
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f"Excel file '{file_name}' not found in {data_folder}")

def read_csv_file(file_name):  ### For Yuri's data
    data_folder = os.path.join(PROJECT_ROOT, "Data")
    file_path = os.path.join(data_folder, file_name)

    if os.path.exists(file_path):
        return pd.read_csv(file_path, skiprows=1)
    raise FileNotFoundError(f"{file_name} not found in {data_folder}")

def read_txt_file(file_name, **kwargs): 
    """
    For saved states. Histotrical, from initial code given to us... might be useful later
    """
    data_folder = os.path.join(PROJECT_ROOT, "Saved_States")
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        return np.loadtxt(file_path, **kwargs)
    raise FileNotFoundError(f"{file_name} not found in {data_folder}")

def readData(data, NumProxies, FundNo):
    X = data.iloc[:, 1:(1+NumProxies)]
    Xnames = X.columns
    Dates = data.iloc[:, 0]
    X = X.to_numpy().T
    Y = data.iloc[:, NumProxies+FundNo] * 100_000_000
    FundName = Y.name
    Y = np.array([[y for y in Y.to_numpy().T]])
    return X, Y, Xnames, FundName, Dates

def generate_initial_states(NumProxies, scale=1.0, overwrite=True):
    """
    Generates and (optionally) saves initial states. Will be tweaked as we go for deciding with more informed states.
    The scale param is just there in case there for some easy manual tuning. There might be some hyperparam tuning potentials with it, but not really a focus rn
    """

    state0 = np.zeros((NumProxies, 1))
    P0 = np.identity(NumProxies) * scale
    Q = np.identity(NumProxies) * scale
    R = np.array([[0.01]])

    if overwrite:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(parent_dir, "Saved_States")
        os.makedirs(save_dir, exist_ok=True)

        np.savetxt(os.path.join(save_dir, "state0.txt"), state0)
        np.savetxt(os.path.join(save_dir, "P0.txt"), P0)
        np.savetxt(os.path.join(save_dir, "Q.txt"), Q)
        np.savetxt(os.path.join(save_dir, "R.txt"), R)

    return state0, P0, Q, R

