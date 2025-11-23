
import pandas as pd
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split



def aggregate_employee_data(data_dir: str = "data"):
    """
    Aggregate award and history data into employee-level JSON structure
    
    Returns:
        List of employee dicts with structure:
        {
            'rec_id': int,
            'awards': [
                {'award_id': int, 'title': str, 'message': str, 'award_date': str},
                ...
            ],
            'num_awards': int,
            'is_vp': bool
        }
    """
    data_path = Path(data_dir)
    vp_pattern = r"\bVP\b|\bV\.P\.\b|\bVice\b|\bVice[- ]President\b|\bSVP\b|\bEVP\b|\bAVP\b|\bRVP\b|\bDVP\b"

    # Load raw data
    print("Loading raw data...")
    control_df = pd.read_csv(data_path / "control copy.csv")
    treatment_df = pd.read_csv(data_path / "treatment copy.csv")
    history_df = pd.read_csv(data_path / "wh_history_full.csv")
    
    # 1. Merge award data
    print("1. Merging award data (control + treatment)...")
    award_df = pd.concat([control_df, treatment_df], ignore_index=True)
   
    
    # 2. Create VP labels from history
    print("2. Creating VP labels from history...")
    history_df['effective_start_date'] = pd.to_datetime(history_df['effective_start_date'])
    label_df = (
        history_df
        .assign(is_vp=history_df['job_title'].str.contains(vp_pattern, regex=True, case=False, na=False))
        .groupby("pk_user")['is_vp']
        .max()
        .reset_index()
    )
    
    
    # 3. Merge labels to award data
    print("3. Merging labels to award data...")
    merged_df = award_df.merge(
        label_df,
        left_on='rec_id',
        right_on='pk_user',
        how='left'
    )
    merged_df['is_vp'] = merged_df['is_vp'].fillna(False).astype(bool)
    
    
    # 4. Aggregate by employee
    print("4. Aggregating by employee...")
    employee_list = []
    
    for rec_id in merged_df['rec_id'].unique():
        emp_awards = merged_df[merged_df['rec_id'] == rec_id]
        
        # Create awards list
        awards_list = []
        for _, row in emp_awards.iterrows():
            award_dict = {
                'title': str(row['title']) if pd.notna(row['title']) else '',
                'message': str(row['message']) if pd.notna(row['message']) else '',
            }
            awards_list.append(award_dict)
        
        # Get VP label
        is_vp = emp_awards['is_vp'].iloc[0] if len(emp_awards) > 0 else False
        
        employee_list.append({
            'rec_id': int(rec_id),
            'awards': awards_list,
            'num_awards': len(emp_awards),
            'is_vp': bool(is_vp)
        })
    
    vp_count = sum(1 for emp in employee_list if emp['is_vp'])
    print(f"   Total employees: {len(employee_list)}")
    print(f"   VP employees: {vp_count} ({vp_count/len(employee_list)*100:.1f}%)")
    
    return employee_list

def split_train_data(employee_list):
    labels = [emp["is_vp"] for emp in employee_list]

    train_idx, test_idx = train_test_split(
        range(len(employee_list)),
        test_size=0.2,
        # random_state=43,
        stratify=labels
    )

    train_list = [employee_list[i] for i in train_idx]
    test_list = [employee_list[i] for i in test_idx]

    return train_list, test_list