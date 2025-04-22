import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

SEQUENCE_LENGTH = 12  # Max months padded
FEATURE_DIM = 10  # Salary + Credit combined (assuming 5 features for salary and 5 for credit)

# Initialize a standard scaler for normalization
scaler_salary = StandardScaler()
scaler_credit = StandardScaler()

def normalize(values, scale=1e6):
    """
    Normalize the data by scaling with a given scale factor (default is 1e6).
    """
    return [v / scale for v in values]

def extract_sequences(applicant: Dict[str, List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts and normalizes salary and credit sequences to fixed length for model input.
    
    Returns:
        salary_tensor: [SEQ_LEN, salary_dim]
        credit_tensor: [SEQ_LEN, credit_dim]
    """
    salary_records = applicant.get("salary_records", [])
    credit_records = applicant.get("credit_records", [])

    # Initialize tensors for salary and credit data
    salary_tensor = torch.zeros(SEQUENCE_LENGTH, 5)  # [12, 5] for salary
    credit_tensor = torch.zeros(SEQUENCE_LENGTH, 5)  # [12, 5] for credit

    # Process salary records and pad/truncate to SEQUENCE_LENGTH (12 months)
    salary_data = []
    for record in salary_records[-SEQUENCE_LENGTH:]:
        salary_data.append([
            record.get("salary", 0.0),
            record.get("salaryTaxSum", 0.0),
            record.get("inpsSum", 0.0),
            record.get("year", 2023),
            record.get("period", 1)
        ])
        
    # Pad with zero if less than SEQUENCE_LENGTH
    while len(salary_data) < SEQUENCE_LENGTH:
        salary_data.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])  # Padding with zero values
        
    # Ensure the data is numeric and clean any invalid entries
    salary_data = np.array(salary_data, dtype=np.float64)  # Ensure data is numeric
    salary_data = np.nan_to_num(salary_data)  # Replace NaN with zero
    
    # Normalize salary data
    salary_tensor[:, 0:3] = torch.tensor(scaler_salary.fit_transform(salary_data[:, :3]), dtype=torch.float32)
    salary_tensor[:, 3] = torch.tensor(salary_data[:, 3] / 2100.0, dtype=torch.float32)  # Normalize year
    salary_tensor[:, 4] = torch.tensor(salary_data[:, 4] / 12.0, dtype=torch.float32)  # Normalize period (month)

    # Process credit records and pad/truncate to SEQUENCE_LENGTH (12 months)
    credit_data = []
    for record in credit_records[-SEQUENCE_LENGTH:]:
        credit_data.append([
            float(record.get("overdue_debt_sum", 0.0)),
            float(record.get("amount", 0.0)),
            float(record.get("percent", 0.0)),
            float(record.get("max_uninter_overdue_percent", 0.0)),
            1.0 if record.get("contract_status", "0") == "1" else 0.0  # Active contract
        ])

    # Pad with zero if less than SEQUENCE_LENGTH
    while len(credit_data) < SEQUENCE_LENGTH:
        credit_data.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])  # Padding with zero values

    # Ensure the data is numeric and clean any invalid entries
    credit_data = np.array(credit_data, dtype=np.float64)  # Ensure data is numeric
    credit_data = np.nan_to_num(credit_data)  # Replace NaN with zero

    # Normalize credit data
    credit_tensor[:, 0:4] = torch.tensor(scaler_credit.fit_transform(credit_data[:, :4]), dtype=torch.float32)
    credit_tensor[:, 4] = torch.tensor(credit_data[:, 4], dtype=torch.float32)

    return salary_tensor, credit_tensor

def build_dataset(get_applicant_by_pinfl) -> List[Dict[str, any]]:
    """
    Build the dataset for training from the database. Fetch data using the provided
    `get_applicant_by_pinfl` function.
    
    Args:
    - get_applicant_by_pinfl: Function that retrieves applicant data by pinfl.
    
    Returns:
    - dataset: List of dictionaries with features for training.
    """
    pinfls = get_unique_pinfls()  # Assuming this is a function that gets all unique pinfls
    dataset = []

    for pinfl in pinfls:
        try:
            applicant = get_applicant_by_pinfl(pinfl)
            if not applicant.get("salary_records") or not applicant.get("credit_records"):
                continue

            salary_tensor, credit_tensor = extract_sequences(applicant)  # [12, 5], [12, 5]
            full_seq = torch.cat([salary_tensor, credit_tensor], dim=1)  # [12, 10]

            record = {
                "pinfl": pinfl,
                "sequence_tensor": full_seq.tolist(),
                "y_risk": 1,  # 🔧 Synthetic label — replace with true label if available
                "loan_limit": 25000000  # 🔧 Mock value — can replace with realistic calculation
            }

            dataset.append(record)

        except Exception as e:
            print(f"[SKIP] Failed on {pinfl}: {e}")
            continue

    return dataset
