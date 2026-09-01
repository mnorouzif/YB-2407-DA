# correlation_age_networth.py

import pandas as pd

def calculate_correlation(file_path):
    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Display first few rows
        print("Dataset Preview:")
        print(data.head())

        # Check if required columns exist
        if 'age' not in data.columns or 'net_worth' not in data.columns:
            print("\nError: Required columns 'age' and 'net_worth' not found.")
            return

        # Calculate correlation
        correlation = data['age'].corr(data['net_worth'])

        # Display result
        print("\nCorrelation between Age and Net-worth:")
        print(correlation)

        # Interpretation
        if correlation > 0:
            print("\nInterpretation: Positive correlation — as age increases, net-worth tends to increase.")
        elif correlation < 0:
            print("\nInterpretation: Negative correlation — as age increases, net-worth tends to decrease.")
        else:
            print("\nInterpretation: No correlation between age and net-worth.")

    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


def main():
    # Provide dataset file name here
    file_path = "data.csv"

    calculate_correlation(file_path)


if __name__ == "__main__":
    main()