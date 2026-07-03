import pandas as pd
import os

def clean_data(input_file, output_file):
    print(f"--- Starting Data Cleaning Phase ---")
    if not os.path.exists(input_file):
        print(f"[Error] Could not find '{input_file}'.")
        print("Ensure you are running the script from the root 'AlgoHub_Task1_HousePricePrediction' folder.")
        return None

    print(f"Loading raw data from {input_file}...\n")
    df = pd.read_csv(input_file)

    # 1. Remove Duplicate Rows
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    duplicates_dropped = initial_rows - df.shape[0]
    print(f"[*] Dropped {duplicates_dropped} duplicate rows.")

    # 2. Handle Missing Values
    if df['price'].isnull().sum() > 0:
        df = df.dropna(subset=['price'])
        print("[*] Dropped rows with missing prices.")

    cols_to_fill = ['beds', 'baths', 'size']
    for col in cols_to_fill:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[*] Filled {missing_count} missing values in '{col}' with median: {median_val}")

    # 3. Fix Data Types
    # Beds should be whole numbers.
    df['beds'] = df['beds'].astype(int)
    # Zip codes don't have decimals.
    df['zip_code'] = df['zip_code'].astype(int)
    print("[*] Converted 'beds' and 'zip_code' to integers.")

    # 4. Save the Cleaned Dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"\n--- Cleaning Complete ---")
    print(f"Cleaned data saved to: {output_file}")
    print(f"Final dataset shape: {df.shape}")

    return df

if __name__ == "__main__":
    input_path = 'data/house_data.csv'
    output_path = 'data/house_data_clean.csv'
    
    clean_data(input_path, output_path)