import pickle
import torch
import numpy as np
import sqlite3
import pandas as pd
import ast
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import json
from transformers import RobertaTokenizer, RobertaModel
import torch
import os

#ADJUSTABLE PARAMETERS THAT CHANGE RUN SPEED 
minimum_playlist_inclusion = 750 # the minimum number of playlists a song has to be in to be considered for model output
playlist_subset = 80000 # subset of playlist to train on - maximum of 1,000,000
gensim_song_embeddings = True  # Dictates which song embeddings to used. true means gensim are used, false means ours are used.
backup_available = True   # If a backup file of the dataframe has been created set to true otherwise false


# Depending on the song embedding type chosen, song embeddings are loaded into a dicitonary of track_uri to embedding

if(gensim_song_embeddings == False):
    with open("/running_python/CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1-track2idx.pkl", 'rb') as f:
        data_uris = pickle.load(f)

    model_data = torch.load("/running_python/CBOW_run_1M_min_5_PP@2024-04-26-10-27-43_con5_pl1000000_emb64_ep1.pt", map_location=torch.device('cpu'))

    embeddings = model_data['embedding.weight']
    embeddings_dict = {}

    for uri, idx in iter(data_uris.items()):
        embeddings_dict[uri] = embeddings[idx]
else:
    
    with open("running_python/song_embeddings.pkl", 'rb') as f:
        embeddings_dict = pickle.load(f)

    for key, value in embeddings_dict.items():
        embeddings_dict[key] = torch.tensor(value)

    embedding_tensors = list(embeddings_dict.values())

    average_embedding = torch.mean(torch.stack(embedding_tensors), dim=0) # An average embedding is calculated as a PAD for tracks that 

# Load artist embeddings from pickle file
with open("running_python/artist_embeddings.pkl", "rb") as f:
    artist_embeddings = pickle.load(f)

# Loading up all training data from the database

conn = sqlite3.connect("running_python/spotify.db")
if backup_available == False:
    sql_query = '''
        SELECT 
            DISTINCT playlist.pid, 
            playlist.playlist_name, 
            artist.artist_uri, 
            track.track_uri,
            playlist.num_albums,
            playlist.num_tracks,
            playlist.num_artists
        FROM 
            playlist_track
        JOIN 
            track ON playlist_track.track_uri = track.track_uri
        JOIN 
            album ON track.album_uri = album.album_uri
        JOIN 
            artist ON album.artist_uri = artist.artist_uri
        JOIN 
            playlist ON playlist_track.pid = playlist.pid
    '''

    df = pd.read_sql_query(sql_query, conn)
    #dataframe holding each playlist as a row with its songs, artists and other metadata
    grouped = df.groupby(['pid', 'playlist_name', 'num_albums', 'num_tracks', 'num_artists']).agg({
        'artist_uri': lambda x: list(x),
        'track_uri': lambda x: list(x)
    }).reset_index()

    grouped.rename(columns={'artist_uri': 'artists', 'track_uri': 'songs'}, inplace=True)

    conn.close()

    grouped.to_csv('back_up.csv', index=False)

    read_from_csv = False
else:
    grouped = pd.read_csv('running_python/back_up.csv')
    read_from_csv = True

if read_from_csv:
    grouped['artists'] = grouped['artists'].apply(ast.literal_eval)
    grouped['songs'] = grouped['songs'].apply(ast.literal_eval)


# Function to calculate average embedding for a list of embeddings
def calculate_average_embedding(embedding_list):
    if embedding_list:
        return np.mean(embedding_list, axis=0)
    else:
        return np.zeros_like(next(iter(artist_embeddings.values())))
    

# Function to create an embedding to represent the playlists name
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')
def generate_playlist_name_embedding(playlist_name):
    if not isinstance(playlist_name, str):
        # If playlist_name is None, create a pad embedding of length 768
        return np.zeros(768) 
    else:
        tokenized_input = tokenizer(playlist_name, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = roberta_model(**tokenized_input)
        last_hidden_states = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
        
        return np.atleast_1d(mean_embedding)


# Creating Training and testing sets - do I need test set if using challenge dataset

import torch
import random
from sklearn.model_selection import train_test_split

# K values to iterate through giving a variety of songs to train on for the model
k_values = [1,5,10,25,50]

subset_size = playlist_subset
random_subset = grouped.sample(n=subset_size, random_state=42)

train_df, test_df = train_test_split(random_subset, test_size=0.2, random_state=42)

training_embeddings = []  # Holds the embeddings of k songs in the playlist, average artist embedding, average song embedding, playlist name embedding, and additional features
training_labels = []  # Holds the remaining songs in the playlist that are to be predicted
training_seed_tracks = [] # Holds the seed tracks so that they cannot be chosen when finding the best 500 songs for a playlist
for _, row in train_df.iterrows():
    p_songs = row['songs']
    k = random.choice(k_values)
    if len(p_songs) >= (2 * k):
        selected_songs = random.sample(p_songs, k)
        # Calculate average song embedding only over the selected k songs
        selected_song_embeddings = [embeddings_dict.get(song_uri, []) for song_uri in selected_songs]
        for song_uri in selected_songs:
            if song_uri in embeddings_dict:
                selected_song_embeddings.append(embeddings_dict[song_uri])
                
        if selected_song_embeddings:
            # Calculate the average song embedding
            average_song_embedding = np.mean(selected_song_embeddings, axis=0)
        else:
            # If no embeddings found, use a zero vector
            average_song_embedding = np.zeros_like(next(iter(embeddings_dict.values())))

        # Calculate average artist embedding only over the artists of the selected k songs
        selected_artists = [artist_uri for song_uri in selected_songs for artist_uri in row['artists']]

        average_artist_embedding = calculate_average_embedding([artist_embeddings.get(artist_uri, []) for artist_uri in selected_artists])

        playlist_name_embedding = generate_playlist_name_embedding(row['playlist_name'])

        p_embeddings = [
            np.atleast_1d(average_song_embedding), 
            np.atleast_1d(average_artist_embedding), 
            playlist_name_embedding,  
            row[['num_albums', 'num_tracks', 'num_artists']].values  # Additional features as a single array
        ]
        training_embeddings.append(np.concatenate(p_embeddings))
        # Convert remaining songs to a list of strings
        remaining_songs = [str(song) for song in p_songs if song not in selected_songs]
        training_labels.append(remaining_songs)
        training_seed_tracks.append(selected_songs)

# Convert the training data to PyTorch tensors
training_tensor = torch.tensor(training_embeddings, dtype=torch.float32)
# Convert the training labels to a list of lists of strings
training_labels_tensor = [labels for labels in training_labels]

# Modify testing set similarly
testing_embeddings = []  # Holds the embeddings of k songs in the playlist, average artist embedding, average song embedding, playlist name embedding, and additional features
testing_labels = []  # Holds the remaining songs in the playlist that are to be predicted
testing_seed_tracks = []
for _, row in test_df.iterrows():
    p_songs = row['songs']
    if len(p_songs) >(2* k):
        selected_songs = random.sample(p_songs, k)
        # Calculate average song embedding only over the selected k songs
        selected_song_embeddings = [embeddings_dict.get(song_uri, []) for song_uri in selected_songs]
        for song_uri in selected_songs:
            # Check if the embedding exists in embeddings_dict
            if song_uri in embeddings_dict:
                selected_song_embeddings.append(embeddings_dict[song_uri])
        # Check if there are embeddings for selected songs
        if selected_song_embeddings:
            # Calculate the average song embedding
            average_song_embedding = np.mean(selected_song_embeddings, axis=0)
        else:
            # If no embeddings found, use a zero vector
            average_song_embedding = np.zeros_like(next(iter(embeddings_dict.values())))
        # Calculate average artist embedding only over the artists of the selected k songs
        selected_artists = [artist_uri for song_uri in selected_songs for artist_uri in row['artists']]
        average_artist_embedding = calculate_average_embedding([artist_embeddings.get(artist_uri, []) for artist_uri in selected_artists])

        playlist_name_embedding = generate_playlist_name_embedding(row['playlist_name'])

        p_embeddings = [
            np.atleast_1d(average_song_embedding),  # Ensure at least one-dimensional
            np.atleast_1d(average_artist_embedding),  # Ensure at least one-dimensional
            playlist_name_embedding,  # Playlist name embedding
            row[['num_albums', 'num_tracks', 'num_artists']].values  # Additional features as a single array
        ]
        testing_embeddings.append(np.concatenate(p_embeddings))
        # Convert remaining songs to a list of strings
        remaining_songs = [str(song) for song in p_songs if song not in selected_songs]
        testing_labels.append(remaining_songs)
        testing_seed_tracks.append(selected_songs)

# Convert the testing data to PyTorch tensors
testing_tensor = torch.tensor(testing_embeddings, dtype=torch.float32)
# Convert the testing labels to a list of lists of strings
testing_labels_tensor = [labels for labels in testing_labels]







# ------LOSS CALCULATION-------

def r_precision(y_true, y_pred):
    # Convert y_true to a set for quick lookup -- Probably not needed
    y_true_set = set(y_true)

    # Determine the number of relevant items -- Not 100% sure
    r = len(y_true_set)
    
    # Filter y_pred to only include items that are in y_true -- Not 100% sure if this is correct
    relevant_predictions = [song for song in y_pred if song in y_true_set]
    
    # Calculate the number of relevant items that should be considered
    r_actual = min(r, len(relevant_predictions))
    
    # Count how many of the top-r_actual predictions are in the true set of relevant songs
    relevant_count = len(relevant_predictions[:r_actual])
    
    if r == 0:
        return 0  # Avoid division by zero if there are no relevant items
    return relevant_count / r

def recommended_songs_clicks(y_true, y_pred):
    # Convert ground truth list to set for quick look-up -- Again, probably not needed
    true_set = set(y_true)
    
    # Find the first relevant track in the predictions
    for i, track in enumerate(y_pred, start=1):
        if track in true_set:
            # how many blocks of 10 tracks are needed
            return (i - 1) // 10 + 1
    
    # If no relevant track is found, set a default value
    # Since the max number of clicks possible plus one is mentioned as 51
    return 51


def dcg(relevances, rank):
    relevances = np.array(relevances)
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0

def idcg(relevances):
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg(sorted_relevances, len(sorted_relevances))

def ndcg(y_true, y_pred):
    relevances = [1 if song in y_true else 0 for song in y_pred]
    actual_dcg = dcg(relevances, len(relevances))
    ideal_dcg = idcg(relevances)
    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg


def playlist_loss(y_true, y_pred):
    total_loss = 0
    num_samples = len(y_true)
    
    for i in range(num_samples):
        # Convert y_true and y_pred to sets for quick lookup
        y_true_set = set(y_true[i])
        y_pred_set = set(y_pred[i])

        # Calculate R-Precision
        r_precision_value = r_precision(y_true_set, y_pred_set)

        # Calculate NDCG
        ndcg_value = ndcg(y_true_set, y_pred_set)

        # Calculate clicks
        clicks_value = recommended_songs_clicks(y_true_set, y_pred_set)

        # Define weights
        weight_r = 1
        weight_ndcg = 1
        weight_clicks = 1/50  # Adjusted weight for clicks

        # Calculate individual losses
        r_loss = 1 - r_precision_value
        ndcg_loss = 1 - ndcg_value

        # Subtract 1/50 to keep loss within the range of 0 to 3, as 1 click (1/50) is the minimum
        clicks_loss = clicks_value * weight_clicks - 1/50

        # Combine individual losses
        loss = (r_loss * weight_r) + (ndcg_loss * weight_ndcg) + clicks_loss

        # Accumulate total loss
        total_loss += loss
    
    # Calculate average loss
    average_loss = total_loss / num_samples
    
    return average_loss


# FILTER EMBEDDINGS TO SONGS IN MORE THAN N PLAYLISTS

filtered_embeddings_dict = {}

# Calculate playlist counts for each song URI
playlist_counts = {}
for songs_list in grouped['songs']:
    for song_uri in songs_list:
        playlist_counts[song_uri] = playlist_counts.get(song_uri, 0) + 1

# Filter embeddings dictionary to include only songs that appear in more than 5 playlists
for song_uri, embedding in embeddings_dict.items():
    if playlist_counts.get(song_uri, 0) > minimum_playlist_inclusion:
        filtered_embeddings_dict[song_uri] = embedding


# MODEL DEFINITION


# Move model and tensors to CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model defintion
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.input_size = input_size
        self.dense1 = nn.Linear(input_size, 256)  #
        self.dense2 = nn.Linear(256, 128)  #
        self.dense3 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        return x

# Custom layer to find closest embeddings
class ClosestEmbeddingsLayer(nn.Module):
    def __init__(self, num_closest=500):
        super(ClosestEmbeddingsLayer, self).__init__()
        self.num_closest = num_closest

    def forward(self, generated_embeddings, seed_tracks):
        print(f'...Finding closest embeddings...')
        closest_embeddings_batch = []
        generated_embeddings = generated_embeddings.to('cpu')
        uris = list(filtered_embeddings_dict.keys())
        
        for i, generated_embedding in enumerate(generated_embeddings):
            current_seed_tracks = seed_tracks[i]
            current_seed_tracks_set = set(current_seed_tracks)
            
            embedding_arrays = np.stack(list(filtered_embeddings_dict.values()))
            # Compute dot product between all embeddings and the generated embedding
            scores = np.dot(embedding_arrays, generated_embedding.detach().numpy())
            # Filter out seed tracks
            top_500_indices = [j for j, uri in enumerate(uris) if uri not in current_seed_tracks_set]
            
            # Get indices of the top 500 closest embeddings excluding seed_tracks
            top_500_indices = np.argpartition(scores[top_500_indices], -500)[-500:]
            # Sort the top indices by actual similarity score for accurate ranking
            top_500_indices = top_500_indices[np.argsort(-scores[top_500_indices])]
            closest_uris = [uris[top_500_indices[j]] for j in range(len(top_500_indices))]
            closest_embeddings_batch.append(closest_uris)

        return closest_embeddings_batch




input_size = training_tensor.shape[1]  # Get input size from the shape of your input tensor
model = CustomModel(input_size).to(device)
closest_embeddings_layer = ClosestEmbeddingsLayer().to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 1
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(training_tensor), batch_size):
        optimizer.zero_grad()
        batch_input = training_tensor[i:i+batch_size].to(device)
        batch_target = training_labels_tensor[i:i+batch_size]  # Use training labels tensor
        batch_seed_tracks = training_seed_tracks[i:i+batch_size] 
        output = model(batch_input)
        closest_embeddings = closest_embeddings_layer(output, batch_seed_tracks)
        loss = playlist_loss(batch_target, closest_embeddings)
        loss_tensor = torch.tensor(loss, requires_grad=True, device=device)
        loss_tensor.backward()  # Compute gradients
        optimizer.step()  # Update weights

    # Validation
    model.eval()
    val_loss_total = 0.0  # Variable to accumulate the total validation loss
    with torch.no_grad():
        for i in range(0, len(testing_tensor), batch_size):
            batch_input = testing_tensor[i:i+batch_size].to(device)
            batch_target = testing_labels_tensor[i:i+batch_size]  # Use testing labels tensor
            batch_seed_tracks = testing_seed_tracks[i:i+batch_size] 
            val_output = model(batch_input)
            val_closest_embeddings = closest_embeddings_layer(val_output, batch_seed_tracks)
            val_loss = playlist_loss(batch_target, val_closest_embeddings)
            val_loss_tensor = torch.tensor(val_loss, dtype=torch.float32, device=device)  # Convert to PyTorch tensor
            val_loss_total += val_loss_tensor.item()

    # Calculate the average validation loss across all batches
    average_val_loss = val_loss_total / (len(testing_tensor) / batch_size)
    print(f"Epoch {epoch+1}, Validation Loss: {average_val_loss:.4f}")


# CREATION OF FILES FROM THE CHALLENGE DATASET


file_path = "running_python/challenge_set.json"

# Load the JSON data from the file
with open(file_path, 'r') as file:
    playlists_data = json.load(file)

all_playlists = playlists_data['playlists']

# Get the length of the pad embeddings for artists and songs
pad_embedding_length_artist = len(next(iter(artist_embeddings.values())))
pad_embedding_length_song = len(next(iter(embeddings_dict.values())))

final_test_embeddings = []
final_test_playlist_ids = []
final_test_seed_tracks = []

for playlist in all_playlists:
    # Extract playlist name and pid
    playlist_name = playlist.get('name', None)
    playlist_pid = playlist.get('pid', None)
    tracks = playlist['tracks']
    seed_tracks = []
    # Calculate average song embedding or use pad embedding if tracks list is empty
    selected_song_embeddings = []
    for song_dict in tracks:
        song_uri = song_dict['track_uri']
        embedding = embeddings_dict.get(song_uri)
        #print(f"embedding: {embedding}")
        if embedding is not None:
            selected_song_embeddings.append(embedding)
            seed_tracks.append(embedding)

    if selected_song_embeddings:
        average_song_embedding = calculate_average_embedding(selected_song_embeddings)
    else:
        average_song_embedding = np.zeros(pad_embedding_length_song)

    selected_artists = []
    for song_dict in tracks:
        song_uri = song_dict['track_uri']
        artists = artist_embeddings.get(song_uri)
        if artists:
            selected_artists.extend(artists)

    if selected_artists:
        average_artist_embedding = calculate_average_embedding([artist_embeddings.get(artist_uri, []) for artist_uri in selected_artists])
    else:
        average_artist_embedding = np.zeros(pad_embedding_length_artist)

    playlist_name_embedding = generate_playlist_name_embedding(playlist_name)
    
    # Additional features
    additional_features = np.array([playlist['num_holdouts'], playlist['num_tracks'], playlist['num_samples']])

    # Assemble the playlist embeddings
    p_embedding = [
        np.atleast_1d(average_song_embedding),  # Ensure at least one-dimensional
        np.atleast_1d(average_artist_embedding),  # Ensure at least one-dimensional
        np.atleast_1d(playlist_name_embedding),  # Playlist name embedding
        additional_features  # Additional features as a single array
    ]
    
    # Append the playlist embeddings to the final test set
    final_test_embeddings.append(np.concatenate(p_embedding))
    
    # Add the playlist ID to the final test playlist IDs
    final_test_playlist_ids.append(playlist_pid)
    final_test_seed_tracks.append(seed_tracks)


final_test_tensor = torch.tensor(final_test_embeddings, dtype=torch.float32)



playlist_predictions = []

# Put the final test data through the trained model to generate predictions
model.eval()
with torch.no_grad():
    for i in range(0, len(final_test_tensor), batch_size):
        batch_input = final_test_tensor[i:i+batch_size].to(device)
        seed_tracks  = final_test_seed_tracks[i:i+batch_size]
        output = model(batch_input)
        closest_embeddings = closest_embeddings_layer(output, seed_tracks)
        for j, playlist_id in enumerate(final_test_playlist_ids[i:i+batch_size]):
            # Convert playlist_id to string
            playlist_id_str = str(playlist_id)
            # Get the top 500 predicted songs for the current playlist
            predicted_songs = closest_embeddings[j][:500]
            # Combine the playlist_id with the predicted songs and convert them to a comma-separated string
            playlist_prediction = ','.join([playlist_id_str] + predicted_songs)
            # Append the playlist prediction to the list
            playlist_predictions.append(playlist_prediction)

# Write the playlist predictions to a CSV file
csv_file_path = "running_python/playlist_predictions_2.csv"
with open(csv_file_path, 'w') as csvfile:
    # Write each playlist prediction to a separate line in the CSV file
    for playlist_prediction in playlist_predictions:
        csvfile.write(playlist_prediction + '\n')