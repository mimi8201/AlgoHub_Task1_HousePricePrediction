import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(file_path, output_folder="figures"):
    print("Loading data...")
    df = pd.read_csv(file_path)

    os.makedirs(output_folder, exist_ok=True)
    print(f"[*] Verified output folder: '{output_folder}/'\n")

    sns.set_theme(style="whitegrid")
    print("Generating and saving visualizations...\n")

    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50, kde=True, color='teal')
    plt.title('Distribution of House Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Number of Houses')

    plt.savefig(f'{output_folder}/1_price_distribution.png', bbox_inches='tight')
    plt.close() 
    print("Saved: 1_price_distribution.png")

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numerical_cols = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_cols.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(f'{output_folder}/2_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Saved: 2_correlation_heatmap.png")

    # 3. Size vs. Price Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='size', y='price', data=df, alpha=0.6, color='coral')
    plt.title('House Size vs. Price')
    plt.xlabel('Size (Square Feet)')
    plt.ylabel('Price ($)')
    plt.savefig(f'{output_folder}/3_size_vs_price.png', bbox_inches='tight')
    plt.close()
    print("Saved: 3_size_vs_price.png")

    # 4. Boxplot: Bedrooms vs. Price
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='beds', y='price', data=df, palette='Set2')
    plt.title('Price Distribution by Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price ($)')
    plt.savefig(f'{output_folder}/4_beds_vs_price.png', bbox_inches='tight')
    plt.close()
    print("Saved: 4_beds_vs_price.png")

    print("\nEDA complete! All graphs saved to the 'figures' folder.")

if __name__ == "__main__":
    run_eda('data/house_data.csv')