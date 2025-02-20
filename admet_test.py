import pandas as pd
import sys
from admet_ai import ADMETModel



# Load dataset
file_path = "tsne_results.csv"
df = pd.read_csv(file_path)

# Extract SMILES column
smiles_list = df['SMILES'].tolist()

# Initialize model
model = ADMETModel()

# Process all batches and save predictions
batch_size = 100
output_file = "Sdmet_predictions.csv"

with open(output_file, "w") as f:
    first_batch = True
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        
        preds = [model.predict(smiles=s) for s in batch]
        
        # Convert predictions to DataFrame
        preds_df = pd.DataFrame(preds)
        preds_df.insert(0, "SMILES", batch)  # Add SMILES column
        
        # Append to file
        preds_df.to_csv(output_file, mode='a', index=False, header=first_batch)
        first_batch = False  # Ensure header is written only once
