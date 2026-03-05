import argparse
import numpy as np
import pandas as pd

def compute_metrics(input_file, output_file):
    # Load the CSV
    df = pd.read_csv(input_file)

    # y_true is the first column
    y_true = df.iloc[:, 0].values

    # y_pred is the second column
    y_pred = df.iloc[:, 1].values

    # Compute the absolute error
    abs_error = np.abs(y_true - y_pred)
    
    # pred_completi_LRR_5win_3k_1fut_0hd_0.0hp_Truetso
    
    # Compute metrics
    mae = np.mean(abs_error)
    std_abs_error = np.std(abs_error)
    max_abs_error = np.max(abs_error)
    percent_above_0_5 = np.mean(abs_error > 0.5) * 100
    percent_above_1 = np.mean(abs_error > 1) * 100
    
    # Extracting parts from the file name (pred_completi_LRR_5win_3k_1fut_0hd_0.0hp_Truetso)
    parts = input_file.split('_')
    rgs   = parts[2]
    win   = parts[3].replace("win", "")
    fut   = parts[4].replace("fut","")
   

    # Create a DataFrame for results
    metrics_df = pd.DataFrame([{
        'rgs': rgs, # regressor name
        'win': win, # window size
        'fut': fut, # future size
        'MEA': mae, # Mean Absolute Error
        'Std': std_abs_error,
        'Max': max_abs_error,
        '%>05': percent_above_0_5,
        '%>1': percent_above_1
    }])

    # Save results to CSV
    metrics_df.to_csv(output_file, index=False, float_format="%.4f")
    # print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute regression error metrics.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, help="Path to save the output CSV file.")

    args = parser.parse_args()
    compute_metrics(args.input_file, args.output_file)
