
import os, json
from tqdm import tqdm
import sys
import pandas as pd
from dotenv import load_dotenv
from src.data_preprocessor import aggregate_employee_data, split_train_data
from src.workflows.employee_cluster import EmployeeCluster
from src.workflows.global_cluster import GlobalCluster

from utils.utils import extract_phrase_set, load_provider_settings, merge_signal_set

# load_dotenv()

def setup_employee_cluster(provider_name: str):
    cfg = load_provider_settings(provider_name)

    return EmployeeCluster(
        provider=cfg["provider"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        api_key=cfg["api_key"]
    )

def setup_global_cluster(provider_name: str):
    cfg = load_provider_settings(provider_name)

    return GlobalCluster(
        provider=cfg["provider"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        api_key=cfg["api_key"]
    )

def main():
    
    employee_cluster = setup_employee_cluster("anthropic")

    print("="*60)
    print("Data Preprocessing")
    print("="*60)
    employee_list = aggregate_employee_data(data_dir="data")
    
    

    for e in tqdm(employee_list):
        emp_id, result, is_vp = employee_cluster.extract_raw_signals(e)
        signal_set = extract_phrase_set(result)
        employee_cluster.clustering_signal(emp_id, signal_set, is_vp)

    
    # Merge Pattern results by vp flag
    non_vp_patterns = merge_signal_set("./output/False")
    vp_patterns = merge_signal_set("./output/True")

    


    
if __name__ == "__main__":
    
    main()
