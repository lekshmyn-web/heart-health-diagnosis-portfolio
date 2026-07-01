import pandas as pd
from pathlib import Path

# Load your dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "heart_disease_cp2.csv"
OUTPUT_PATH = DATA_DIR / "cp2_shuffled.csv"

df = pd.read_csv(INPUT_PATH)


# Shuffle the rows
df_shuffled = df.sample(frac=1, random_state=42)

# Save the shuffled DataFrame to a new CSV file
df_shuffled.to_csv(OUTPUT_PATH, index=False)

print("Shuffled dataset saved as cp2_shuffled.csv.")



#checking the dataset
# Check initial dataset size
print("Initial dataset size:", df.shape)

# Drop rows with missing values
df_clean = df.dropna()

# Check cleaned dataset size
print("Cleaned dataset size:", df_clean.shape)
