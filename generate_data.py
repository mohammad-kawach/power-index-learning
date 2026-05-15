import pandas as pd
import numpy as np
import os
from src.banzhaf import exact_banzhaf, NUM_AGENTS

def generate_midterm_2d_dataset(num_games=20000): # Paper used k=200,000 for best results [cite: 151]
    os.makedirs("data", exist_ok=True)
    data = []
    print(f"Generating {num_games} 2D voting games...")
    
    for _ in range(num_games):
        # 1. Sample random weights for m agents
        weights = np.random.randint(1, 20, size=NUM_AGENTS)
        # 2. Set Quota (usually 50-80% of total weight)
        total = np.sum(weights)
        quota = np.random.randint(int(total * 0.5), int(total * 0.8) + 1)
        
        # 3. Get Ground Truth (Banzhaf indices quantify agent importance) [cite: 42]
        labels = exact_banzhaf(weights, quota)
        
        row = {f"weight_{i}": weights[i] for i in range(NUM_AGENTS)}
        row["quota"] = quota
        row.update({f"target_{i}": labels[i] for i in range(NUM_AGENTS)})
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv("data/midterm_2d_data.csv", index=False)
    print(f"2D Dataset saved to data/midterm_2d_data.csv | Shape: {df.shape}")

if __name__ == "__main__":
    generate_midterm_2d_dataset()