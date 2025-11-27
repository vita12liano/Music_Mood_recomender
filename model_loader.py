import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import Counter
from datetime import datetime

#Paths
MODEL_DIR = "models"
DATA_DIR = "data/processed"

class MLPCluster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def prepare_dataset(df_model, feature_cols):
    
    #Drop PCA columns if exist
    columns_to_drop = [col for col in ["pca_x", "pca_y"] if col in df_model.columns]
    if columns_to_drop:
        df_model = df_model.drop(columns=columns_to_drop)
    
    #Compute distances to centroids
    df_model['distance_to_centroid'] = 0.0
    
    for cluster_id in df_model["macro_cluster"].unique():
        cluster_data = df_model[df_model["macro_cluster"] == cluster_id]
        cluster_features = cluster_data[feature_cols].values
        centroid = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        df_model.loc[df_model["macro_cluster"] == cluster_id, 'distance_to_centroid'] = distances
    
    #Compute normalized distance
    df_model['distance_normalized'] = df_model.groupby('macro_cluster')['distance_to_centroid'] \
                                               .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
    
    #Computing metrics for ranking
    df_model['rank_distance'] = df_model.groupby('macro_cluster')['distance_to_centroid'].rank(ascending=True)
    df_model['rank_popularity'] = df_model.groupby('macro_cluster')['popularity'].rank(ascending=False)
    
    weight_distance = 0.4
    weight_pop = 0.6
    df_model['combined_rank'] = (
        weight_distance * (1 - df_model['rank_distance'] / df_model.groupby('macro_cluster')['rank_distance'].transform('max')) +
        weight_pop * (1 - df_model['rank_popularity'] / df_model.groupby('macro_cluster')['rank_popularity'].transform('max'))
    )
    
    return df_model

def load_model():
    #Load scaler
    scaler_mean = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy"))
    scaler_scale = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    #Load label encoder
    le_classes = np.load(os.path.join(MODEL_DIR, "label_encoder_classes.npy"), allow_pickle=True)
    
    #Load model
    feature_cols = [
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "loudness", "speechiness", "tempo", "valence", "duration_ms"
    ]
    
    input_dim = len(feature_cols)
    hidden_dim = 64
    num_classes = len(le_classes)
    device = torch.device("cpu")
    
    model = MLPCluster(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp_subcluster.pth"), map_location=device))
    model.eval()
    
    #Load dataset and prepare ranking
    df_model = pd.read_csv(os.path.join(DATA_DIR, "spotify_dataset_clustered.csv"))
    df_model = prepare_dataset(df_model, feature_cols)  
    
    return model, scaler, le_classes, df_model, feature_cols, device

def recommend_playlist(df_model, mood, activity, time_of_day, age):
    mood_map = {
        "relax": ["0_0", "0_1", "1_0", "1_1"],
        "happy": ["2_1", "2_2", "2_5"],
        "sad": ["1_0", "1_1"],
        "workout": ["2_3", "2_4", "2_5"],
        "focus": ["0_0", "1_0", "1_2"],
        "party": ["2_0", "2_1", "2_5"]
    }
    
    activity_map = {
        "study": ["0_0", "1_0", "1_2"],
        "walking": ["2_1", "2_2", "0_1"],
        "running": ["2_3", "2_4", "2_5"],
        "relaxing": ["0_0", "0_1", "1_0"],
        "party": ["2_0", "2_1", "2_5"]
    }
    
    time_map = {
        "morning": ["2_2", "2_5", "0_1"],
        "afternoon": ["2_1", "2_2"],
        "evening": ["0_0", "1_1", "2_0"],
        "night": ["0_0", "1_0", "1_1"]
    }
    
    clusters = []
    clusters += mood_map.get(mood, [])
    clusters += activity_map.get(activity, [])
    clusters += time_map.get(time_of_day, [])
    
    if mood == "party" and activity == "party" and time_of_day == 'night':
        clusters = [c for c in clusters if c not in ["0_0", "1_0", "1_1"]]
        clusters += ["2_0", "2_1", "2_5"]
    
    Coun = dict(Counter(clusters))
    cluster_sum = len(clusters)
    playlist_length = 20
    
    weighted_clusters = {}
    for cluster, count in Coun.items():
        weight = count / cluster_sum
        n_songs = max(1, int(weight * playlist_length))
        weighted_clusters[cluster] = n_songs
    
    candidated_songs = pd.DataFrame()
    for cluster, count in weighted_clusters.items():
        cluster_songs = (
            df_model[df_model["subcluster"] == cluster]
            .sort_values('combined_rank', ascending=False)
            .head(count * 3)
        )
        candidated_songs = pd.concat([candidated_songs, cluster_songs])
    
    year = datetime.now().year
    birth_year = int(year) - int(age)
    
    if mood in ("party", "happy"):
        youth_start = birth_year + 15
        youth_end = birth_year + 30
        if youth_start <= year:
            age_filtered = df_model[
                (df_model["subcluster"] == max(Coun, key=Coun.get)) &
                (df_model["year"] >= youth_start) &
                (df_model["year"] <= min(youth_end, year))
            ].sort_values('popularity', ascending=False).head(5)
            candidated_songs = pd.concat([candidated_songs, age_filtered])
    
    candidated_songs = candidated_songs.drop_duplicates(subset=['track_id'])
    final_playlist = []
    artist_count = {}
    
    for _, song in candidated_songs.sort_values('combined_rank', ascending=False).iterrows():
        artist = song['artist_name']
        if artist_count.get(artist, 0) < 2:
            final_playlist.append(song)
            artist_count[artist] = artist_count.get(artist, 0) + 1
        if len(final_playlist) >= 20:
            break
    
    return pd.DataFrame(final_playlist)