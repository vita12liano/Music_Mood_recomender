# 🎧 Music Mood Recommender
**Accademic Project for FDS (2025), Data science (Sapienza, University of Rome.)**  
**A hybrid ML-based system for mood-aware, context-aware music playlist generation.**

---

## 🔗 Overview

**Music Mood Recommender** is an intelligent system that generates personalized playlists by combining:

- User **mood**
- **Activity**
- **Time of day**
- **Weather**
- **Age-based temporal preference modeling**
- **Favourite artists**
- **Preferred languages** (multi-selection)
- **Exploration mode** (discover new artists vs. familiar music)

The system integrates:

- **Unsupervised clustering** (UMAP → HDBSCAN → K-Means)  
- **Deep Learning** (MLP classifier for subcluster prediction)  
- **Hybrid scoring recommender system**  
- **Telegram Bot interface**

---

## 📦 Dataset Source
The dataset was downloaded from Kaggle, and it contains the following Spotify numerical audio features:

| Feature            | Meaning                                                |
|--------------------|--------------------------------------------------------|
| `acousticness`     | Likelihood of being acoustic                           |
| `danceability`     | Suitability for dancing                                |
| `energy`           | Perceived intensity                                    |
| `instrumentalness` | Probability of instrumental track                      |
| `liveness`         | Presence of live performance                           |
| `loudness`         | dBFS measurement                                       |
| `speechiness`      | Spoken-word dominance                                  |
| `tempo`            | Beats per minute                                       |
| `valence`          | Musical positivity                                     |
| `duration_ms`      | Track duration                                         |

**Link:** https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020/code

---

## 🧠 System Architecture

┌────────────────────────────────────────────────────────────────────┐
│                     Music Mood Recommender (core)                  │
│                    recommender_05.recommend_playlist()             │
└────────────────────────────────────────────────────────────────────┘
                │
                │  User inputs
                │  (mood, activity, part_of_day, weather, age,
                │   explorer, fav_artists, language_prefs, n)
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1) Build target profile (heuristics)                               │
│    build_target_profile(...)                                       │
│    - creates a 10-D "target vector" over Spotify audio features    │
│      [acousticness, danceability, ..., duration_ms]                │
│    - derives (year_pref, year_low, year_high) from age             │
│    - optional "special steering" presets for rare patterns         │
└────────────────────────────────────────────────────────────────────┘
                │
                │ target_profile (10-D) + year range
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2) Candidate pool filtering (df_local)                             │
│    - special moods: keep only tracks flagged is_kids/is_christmas… │
│    - normal moods: remove problematic flags                        │
│    - optional language filter on main_language                     │
│    Output: df_local (subset of df)                                 │
└────────────────────────────────────────────────────────────────────┘
                │
                │ df_local indices
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3) Representations used by the model                               │
│    - TRUE input space (model): 10-D scaled features                │
│         X_scaled = scaler.transform(df[feature_cols])              │
│         X_local  = X_scaled restricted to df_local                 │
│    - UMAP 2D is NOT used here (only for visualization elsewhere)   │
└────────────────────────────────────────────────────────────────────┘
                │
                │ target_profile (10-D) → scaled
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 4) Subcluster prediction (MLP classifier)                          │
│    predict_subcluster_from_profile(profile)                        │
│    - scaler.transform(target_profile)                              │
│    - MLP → logits → softmax probabilities                          │
│    - optional small prior bias for rare subclusters                │
│    Output: subcluster_pred                                         │
└────────────────────────────────────────────────────────────────────┘
                │
                │ subcluster_pred + target_profile
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5) Neighbour subclusters (centroid similarity)                     │
│    find_neighbour_subclusters(profile, top_k=3)                    │
│    - compare target_profile to subcluster centroids (mean features)│
│    - cosine similarity in 10-D scaled space                        │
│    Output: neighbour_subclusters                                   │
└────────────────────────────────────────────────────────────────────┘
                │
                │ target_profile + X_local
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6) Base similarity to tracks (content-based)                       │
│    mood_sim = cosine_similarity(target_profile_scaled, X_local)    │
│    -> "how close each track is to the target audio profile"        │
└────────────────────────────────────────────────────────────────────┘
                │
                │ plus cluster membership info
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 7) Cluster bonus (hierarchical structure)                          │
│    - same subcluster: highest bonus                                │
│    - neighbour subclusters: medium bonus                           │
│    - same macro_cluster: smaller bonus                             │
│    - other macro_cluster: minimal bonus                            │
│    (weights depend on explorer mode)                               │
└────────────────────────────────────────────────────────────────────┘
                │
                │ + other scoring components
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 8) Multi-factor scoring (all normalized 0..1)                      │
│    A) mood_cluster_score = 0.6*mood_sim + 0.4*cluster_bonus        │
│    B) time_score        = temporal_score(year, range, explorer)    │
│    C) pop_score         = popularity_score(popularity, explorer)   │
│    D) weather_score     = compute_weather_score(df_local, weather) │
│    E) day_score         = compute_part_of_day_score(df_local, time)│
│    F) user_taste_score  = compute_user_taste_score(fav_artists)    │
│                                                                    │
│    Final weighted sum:                                             │
│      0.35*taste + 0.25*(mood_cluster) + 0.20*time                  │
│    + 0.10*pop   + 0.05*weather        + 0.05*day                   │
└────────────────────────────────────────────────────────────────────┘
                │
                │ final_score per track
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 9) Post-filters & diversification                                  │
│    - safe mode: constrain year window + pop>=30 (if available)     │
│    - drop duplicate track_id                                       │
│    - cap max 3 tracks per artist                                   │
│    - if fav_artists given: enforce minimum % from "other artists"  │
└────────────────────────────────────────────────────────────────────┘
                │
                │ sort by score
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 10) Output                                                         │
│     top_result = top-N rows with scores and metadata               │
│     (track_id, track_name, artist, year, popularity, cluster info…)│
└────────────────────────────────────────────────────────────────────┘

## 📁 Repository Structure
Music_Mood_recommender/
├── bot/
│   └── telegram_bot.py       # Bot engine.
├── data/
│   ├── processed/            # Saving processed data, for bot functioning.
│   │  └── spotify_dataset_clustered.csv
│   └── data.csv
│
├── models/                   # Saving trained MLP model, for bot functioning.
│   ├── mlp_subcluster.pth
│   ├── scaler_mean.npy
│   ├── scaler_scale.npy
│   └── label_encoder_classes.npy
│
├── recommender_05.py         # Recommendation system.
├── notebooks/                # Project workflow
│   ├── 02_spotify_kaggle_dt.ipynb
│   ├── 03_feature_engineering_&_clustering.ipynb
│   ├── 04_MLP_model.ipynb
│   └── 05_recommender.ipynb
└── Spotify_API.py            # bot for saving the generated playlist on
                                your spotify-account's library (limit of 25 users only allowed)
                              # This second bot works 24/7 via cloud server 
                                (doesn't need deploying).

## 💻📱 How to run and use Music-Mood recommender?

You can use this project in **three** ways:

1. Directly via the **Telegram bot**
2. Running the **Telegram bot locally** from this repo
3. Using the **recommender as a standalone Python function**

-------------------------------------------------------

### 1. Use the Telegram bot on Telegram

If the bot is already deployed by the owners, you can simply:

1. Open **Telegram**
2. Search for the bot by its username: `@<your_bot_username>`
3. Start a chat and type `/start`
4. Follow the guided questions (mood, activity, weather, etc.) and receive your playlist 🎧

-------------------------------------------------------

### 2. Run the Telegram bot locally

If you want to run the bot from this repository:

1. **Clone the repository** and move into the project folder:
    ```bash
    git clone https://github.com/vita12liano/Music_Mood_recomender.git
    cd Music_Mood_recomender
2. **Create and activate a virtual environment** (optional but recommended):
    python -m venv .venv
    source .venv/bin/activate      # macOS / Linux
    .venv\Scripts\activate       # Windows (PowerShell / CMD)
3. **Install dependecies**:
    pip install -r requirements.txt
4. **Create your ".env" file, and insert your Telegram-bot's Token**:
    TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
5. **Make sure the data and model files are in place**:
	•	data/processed/spotify_dataset_clustered.csv
	•	models/mlp_subcluster.pth
	•	models/scaler_mean.npy
	•	models/scaler_scale.npy
	•	models/label_encoder_classes.npy
6. **Run the "telegram_bot.py" file**:
    python telegram_bot.py

-------------------------------------------------------

## 3. Use the recommender as a standalone Python module

You can also bypass Telegram entirely and call the recommender directly from Python.

Make sure you are in the project root and your environment is set up (data + models as in option 2. Run the Telegram bot locally), then:
    from recommender_05 import recommend_playlist

    playlist = recommend_playlist(
        mood="happy",
        activity="party",
        part_of_day="evening",
        weather="sunny",
        age=23,
        explorer=True,              # True = more exploration, False = safer / more popular
        n=20,                       # number of recommended tracks
        fav_artists=["Avicii"],     # optional list of favourite artists
        language_prefs=["en"]       # optional list of language codes (e.g. ["en", "it"])
    )
    print(playlist.head())

## 📚 References

### Features Analisys:

### Clustering & Subclustering:

### MLP model usage:

### Hybrid recommender:

## 🧑‍🧑‍🧒‍🧒 Team:
• Vitaliano Barberio 1992511
• Debora Siri 
• Mirko Impera
Sapienza University of Rome, Data Science — Fundamentals of Data Science (2025)

