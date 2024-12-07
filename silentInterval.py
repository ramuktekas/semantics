import pandas as pd

# Load the data
file_path = 'WD_MOVIE2_HOI.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define movie intervals
movie_intervals = [(20, 246), (267, 525), (545, 794), (815, 897)]

# Function to find silent intervals within a given movie interval
def find_silent_intervals(data, start_interval, end_interval):
    # Filter data within the movie interval
    movie_data = data[(data['Start_time'] >= start_interval) & (data['End_time'] <= end_interval)]
    
    # Initialize variables
    silent_intervals = []
    current_start = start_interval
    word_detected = False

    # Iterate through the rows
    for _, row in movie_data.iterrows():
        if pd.isna(row['Words']):  # If no words are spoken
            if not word_detected:  # If currently in a silent interval
                word_detected = True
                current_start = row['Start_time']
        else:  # Word detected
            if word_detected:  # End of a silent interval
                silent_intervals.append((current_start, row['Start_time']))
                word_detected = False

    # Check if the last interval remains open
    if word_detected:
        silent_intervals.append((current_start, end_interval))
    
    return silent_intervals

# Scan each movie interval for silent intervals
silent_intervals_by_movie = {}
for idx, (start, end) in enumerate(movie_intervals):
    silent_intervals_by_movie[idx] = find_silent_intervals(data, start, end)

# Print silent intervals for each movie
for movie_idx, intervals in silent_intervals_by_movie.items():
    print(f"Silent intervals in movie {movie_idx + 1}: {intervals}")
