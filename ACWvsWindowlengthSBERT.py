import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

# Initialize the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to create sentences with a sliding window
def create_sentences(data, window_size=100, step_size=1):
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

# Function to calculate ACW for a range of window sizes
def calculate_acw(data, window_sizes):
    acw_results = []
    
    for window_size in window_sizes:
        sentences, start_times, end_times = create_sentences(data, window_size)
        embeddings = model.encode(sentences)
        cosine_similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        
        cosine_similarity_values = cosine_similarities.diagonal()
        nlags = len(cosine_similarity_values) - 1
        acf_values = acf(cosine_similarity_values, nlags=nlags, fft=True)
        acw = next((i for i, val in enumerate(acf_values) if val < 0), None)
        
        acw_results.append((window_size, acw))
    
    return acw_results

# Function to plot ACW vs. Length of sliding window
def plot_acw_vs_window_length(acw_results):
    window_lengths = [res[0] for res in acw_results]
    acw_values = [res[1] for res in acw_results]

    plt.figure(figsize=(10, 6))
    plt.plot(window_lengths, acw_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Length of Sliding Window')
    plt.ylabel('ACW (Lag where ACF goes below 0)')
    plt.title('ACW vs. Length of Sliding Window')
    plt.show()

# Main script
if __name__ == "__main__":
    file_path = '7T_MOVIE2_HO1.csv'  # Replace with the correct path to your file
    data = pd.read_csv(file_path)
    
    window_sizes = [1,2,3, 4, 5, 6, 7, 8, 9,10]
    acw_results = calculate_acw(data, window_sizes)
    plot_acw_vs_window_length(acw_results)
