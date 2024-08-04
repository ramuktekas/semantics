import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
file_path = '7T_MOVIE2_HO1.csv'  # Replace with the correct path to your file
data = pd.read_csv(file_path)

# Initialize the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to create sentences with a sliding window
def create_sentences(data, window_size=150, step_size=1):
    sentences = []
    start_times = []
    end_times = []
    max_time = data['End_time'].max()
    current_start = 0  # Start from 0
    
    while current_start <= max_time:
        current_end = current_start + window_size
        window_data = data[(data['Start_time'] >= current_start) & (data['Start_time'] < current_end)]
        sentence = ' '.join(window_data['Words'])
        sentences.append(sentence)
        start_times.append(current_start)
        end_times.append(current_end)
        current_start += step_size
    
    return sentences, start_times, end_times

# Create sentences
sentences, start_times, end_times = create_sentences(data)

# Calculate embeddings
embeddings = model.encode(sentences)

# Calculate cosine similarities between consecutive sentences
cosine_similarities = cosine_similarity(embeddings[:-1], embeddings[1:])

# Prepare the data for the CSV file
csv_data = {
    'startSen': start_times[:-1],  # Exclude the last start time
    'endSen': end_times[:-1],  # Exclude the last end time
    'Sentence': sentences[:-1],  # Exclude the last sentence as it has no subsequent sentence for similarity comparison
    'Next_Sentence': sentences[1:],  # Exclude the first sentence
    'Cosine_Similarity': cosine_similarities.diagonal()
}

df = pd.DataFrame(csv_data)

# Save to CSV
output_file_path = '150_1_7T_MOVIE2_HOI.csv'
df.to_csv(output_file_path, index=False)

# Print confirmation
print(f"CSV file saved to {output_file_path}")
