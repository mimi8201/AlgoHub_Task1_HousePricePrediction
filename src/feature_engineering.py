import pandas as pd
import os
import joblib

def run_feature_engineering(input_path, output_path, model_dir):
    print("--- Starting Feature Engineering ---")
    
    # 1. Load the clean data
    if not os.path.exists(input_path):
        print(f"[Error] '{input_path}' not found. Did you run preprocess.py?")
        return
        
    df = pd.read_csv(input_path)
    
    # 2. Combine Beds and Baths
    df['beds_baths_total'] = df['beds'] + df['baths']
    print("[*] Created new feature: 'beds_baths_total'")
    
    # 3. Custom Feature: Square Footage per Room
    # A 2000 sqft house with 2 rooms feels luxurious. A 2000 sqft house with 8 rooms feels cramped.
    # We add 1 to avoid dividing by zero just in case.
    df['sqft_per_room'] = df['size'] / (df['beds_baths_total'] + 1)
    print("[*] Created new feature: 'sqft_per_room'")

    # 4. Frequency Encoding for zip_code
    # value_counts() counts how many times each zip code appears. to_dict() turns it into a dictionary mapping.
    zip_freq = df['zip_code'].value_counts().to_dict()
    
    # .map() replaces the zip code with its frequency count
    df['zip_code_freq'] = df['zip_code'].map(zip_freq)
    print("[*] Applied Frequency Encoding to 'zip_code'")
    
    df = df.drop(columns=['zip_code'])
    
    # 5. Save the mapping for our Streamlit App (CRITICAL STEP)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(zip_freq, os.path.join(model_dir, 'zip_freq_map.pkl'))
    print(f"[*] Saved zip frequency map to '{model_dir}/zip_freq_map.pkl'")
    
    # 6. Save the new engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n--- Feature Engineering Complete ---")
    print(f"Engineered data saved to: {output_path}")
    print(f"Final columns: {list(df.columns)}")

if __name__ == "__main__":
    run_feature_engineering(
        input_path='data/house_data_clean.csv', 
        output_path='data/house_data_features.csv',
        model_dir='models'
    )