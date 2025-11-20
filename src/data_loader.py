"""Data loading utilities for control, treatment, and history datasets"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataLoader:
    """Load and preprocess employee award and history data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_control(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load control dataset (non-promoted employees)"""
        if file_path is None:
            file_path = self.data_dir / "control copy.csv"
        df = pd.read_csv(file_path)
        return df
    
    def load_treatment(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load treatment dataset (promoted employees)"""
        if file_path is None:
            file_path = self.data_dir / "treatment copy.csv"
        df = pd.read_csv(file_path)
        return df
    
    def load_history(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load employee history dataset"""
        if file_path is None:
            file_path = self.data_dir / "wh_history_full.csv"
        df = pd.read_csv(file_path)
        # Convert date columns
        df['effective_start_date'] = pd.to_datetime(df['effective_start_date'])
        df['effective_end_date'] = pd.to_datetime(df['effective_end_date'])
        return df
    
    def get_employee_awards(
        self, 
        nom_id: int, 
        dataset: str = "control"
    ) -> pd.DataFrame:
        """Get all awards for a specific employee"""
        if dataset == "control":
            df = self.load_control()
        elif dataset == "treatment":
            df = self.load_treatment()
        else:
            raise ValueError("dataset must be 'control' or 'treatment'")
        
        return df[df['nom_id'] == nom_id].copy()
    
    def get_employee_history(self, pk_user: int) -> pd.DataFrame:
        """Get career history for a specific employee"""
        history_df = self.load_history()
        return history_df[history_df['pk_user'] == pk_user].copy()
    
    def load_merged_data(self) -> pd.DataFrame:
        """Load and merge control and treatment datasets"""
        control = self.load_control()
        treatment = self.load_treatment()
        return pd.concat([control, treatment], ignore_index=True)
    
    def get_employee_awards_merged(self, nom_id: int) -> pd.DataFrame:
        """Get all awards for an employee from merged dataset"""
        merged = self.load_merged_data()
        return merged[merged['nom_id'] == nom_id].copy()
    
    def is_vp(self, pk_user: int) -> bool:
        """Check if employee reached VP level based on history"""
        history = self.get_employee_history(pk_user)
        if len(history) == 0:
            return False
        
        latest_position = history.sort_values('effective_start_date').iloc[-1]['job_title']
        return 'VP' in str(latest_position) or 'Vice President' in str(latest_position) or 'SVP' in str(latest_position)
    
    def aggregate_employee_data(
        self, 
        nom_id: int, 
        dataset: str = "control"
    ) -> Dict:
        """Aggregate all data for an employee into a single dict"""
        awards = self.get_employee_awards(nom_id, dataset)
        history = self.get_employee_history(nom_id)
        
        # Combine award messages
        award_texts = []
        for _, row in awards.iterrows():
            award_texts.append(f"Title: {row['title']}\nMessage: {row['message']}")
        
        # Process history
        history_info = []
        for _, row in history.iterrows():
            history_info.append({
                'job_title': row['job_title'],
                'manager_id': row['fk_direct_manager'],
                'start_date': row['effective_start_date'],
                'end_date': row['effective_end_date'],
                'tenure_days': (row['effective_end_date'] - row['effective_start_date']).days if pd.notna(row['effective_end_date']) else None
            })
        
        return {
            'nom_id': nom_id,
            'awards': '\n\n---\n\n'.join(award_texts),
            'history': history_info,
            'num_awards': len(awards),
            'num_role_changes': len(history),
            'is_promoted': dataset == "treatment"
        }
    
    def aggregate_employee_data_merged(self, nom_id: int) -> Dict:
        """Aggregate employee data from merged dataset"""
        awards = self.get_employee_awards_merged(nom_id)
        history = self.get_employee_history(nom_id)
        
        # Combine award messages
        award_texts = []
        for _, row in awards.iterrows():
            award_texts.append(f"Title: {row['title']}\nMessage: {row['message']}")
        
        # Process history
        history_info = []
        for _, row in history.iterrows():
            history_info.append({
                'job_title': row['job_title'],
                'manager_id': row['fk_direct_manager'],
                'start_date': row['effective_start_date'],
                'end_date': row['effective_end_date'],
                'tenure_days': (row['effective_end_date'] - row['effective_start_date']).days if pd.notna(row['effective_end_date']) else None
            })
        
        # Check VP status from history
        is_vp = self.is_vp(nom_id)
        
        return {
            'nom_id': nom_id,
            'awards': '\n\n---\n\n'.join(award_texts),
            'history': history_info,
            'num_awards': len(awards),
            'num_role_changes': len(history),
            'is_vp': is_vp
        }

