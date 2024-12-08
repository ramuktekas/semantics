import os
import pandas as pd

# Define variables
base_directory = "/Volumes/WD/Extracted_TS_Movie2_Saket"
rois = ["A1", "TA2", "PSL"]
output_directory = "/Volumes/WD/desktop/Figures8oct"  #
os.makedirs(output_directory, exist_ok=True)


# Loop through each ROI
for roi in rois:
    data = {}
    
    # Find all subject directories in the base directory
    for subject_id in os.listdir(base_directory):
        if subject_id.isdigit() and len(subject_id) == 6:  # Ensure subject ID is a 6-digit number
            file_path = os.path.join(base_directory, subject_id, f"TS_{roi}_{subject_id}.1D")
            
            if os.path.exists(file_path):
                # Load BOLD signal
                with open(file_path, 'r') as file:
                    bold_signal = [float(value) for value in file.read().split()]
                
                # Add data to the dictionary
                data[subject_id] = bold_signal
    
    # Create DataFrame
    if data:
        df = pd.DataFrame.from_dict(data, orient='index')  # Each subject is a row
        df = df.transpose()  # Transpose to make subjects columns
        df.columns.name = 'subject_id'  # Set column name as subject_id
        
        # Save to CSV
        output_file = os.path.join(output_directory, f"{roi}_tpn_182subs.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    else:
        print(f"No data found for ROI: {roi}")
