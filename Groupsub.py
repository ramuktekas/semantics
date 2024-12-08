import pandas as pd
import json

# Function to process data for a given ROI
def process_roi_file(roi):
    print(f"Processing ROI: {roi}")
    
    # Load the data file
    file_path = f"{roi}_tan_182subs.csv"
    data = pd.read_csv(file_path, header=None)
    print(f"Original data shape: {data.shape}")
    
    # Remove the last column
    data = data.iloc[:, :-1]
    print(f"After removing last column shape: {data.shape}")
    
    # Extract the subject IDs from the first row
    subject_ids = data.iloc[0, :].values
    print(f"Subject IDs shape: {subject_ids.shape}")
    
    # Remove the first row (subject IDs)
    data = data.iloc[1:, :]
    print(f"After removing subject IDs row shape: {data.shape}")
    
    # Calculate the mean of the time series for each subject
    mean_values = data.mean(axis=0).values
    print(f"Mean values array shape: {mean_values.shape}")
    
    # Create a dictionary of subject IDs and their corresponding mean values
    acw_values = {str(subject_id): mean for subject_id, mean in zip(subject_ids, mean_values)}
    print(f"Dictionary size: {len(acw_values)}")
    
    # Save dictionary to JSON file
    output_file = f"acw_values_{roi}.json"
    with open(output_file, 'w') as json_file:
        json.dump(acw_values, json_file, indent=4)
    print(f"Saved dictionary to {output_file}")

# List of ROIs
rois = ["A1", "TA2", "PSL"]

# Process each ROI
for roi in rois:
    process_roi_file(roi)
