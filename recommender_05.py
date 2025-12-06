#!/usr/bin/env python
# coding: utf-8

# # Cell 1 - Import & Data loading

# In[277]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn
import torch.nn.functional as F


# In[278]:


# Cartelle
import os
import numpy as np
import pandas as pd
# ... (tutti gli altri import che avevi già)

# === Path robusti basati sulla posizione di questo file ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

data_path = os.path.join(DATA_PROCESSED_DIR, "spotify_dataset_clustered.csv")
model_path = os.path.join(MODEL_DIR, "mlp_subcluster.pth")
scaler_mean_path = os.path.join(MODEL_DIR, "scaler_mean.npy")
scaler_scale_path = os.path.join(MODEL_DIR, "scaler_scale.npy")
le_classes_path = os.path.join(MODEL_DIR, "label_encoder_classes.npy")

print("DATA_PROCESSED_DIR:", DATA_PROCESSED_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("CSV path:", data_path)

# Audio-features usate dal modello
feature_cols = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
]



# In[315]:


df = pd.read_csv(data_path)
df.head()


# # Cell 2 - Loading scaler, label encoder & defining MLP model

# In[280]:


# Scaler
scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Classi del LabelEncoder
le_classes = np.load(le_classes_path, allow_pickle=True).astype(str)
num_classes = len(le_classes)

# Modello
input_dim = len(feature_cols)
hidden_dim = 64


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


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = MLPCluster(input_dim, hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()




# # Cell 3 - Precompute: feature matrix & subcluster summary

# In[281]:


# Matrice feature globale
X_raw = df[feature_cols].values
X_scaled = scaler.transform(X_raw)

subcluster_summary = (
    df.groupby("subcluster")[feature_cols]
      .mean()
      .sort_index()
)

subcluster_summary.head()


# # Cell   -

# In[282]:


# Flag problematici nel dataset
PROBLEMATIC_FLAG_COLS = ["is_kids", "is_christmas", "is_nursery", "is_religious"]

# Mappa mood → categorie da includere
MOOD_FLAG_MAP = {
    "kids": ["is_kids", "is_nursery"],
    "children": ["is_kids", "is_nursery"],
    "nursery": ["is_nursery"],
    "christmas": ["is_christmas"],
    "xmas": ["is_christmas"],
    "holiday": ["is_christmas"],
    "religious": ["is_religious"],
    "gospel": ["is_religious"],
}


# # Cell 4 - Helpers for profile targeting (mood, activity, weather, age)

# In[283]:


# === 6. Costruzione profilo utente + range temporale (15–30 anni) ===

def build_target_profile(mood: str,
                         activity: str,
                         weather: str,
                         part_of_day: str,
                         age: int,
                         explorer: bool,
                         df_global: pd.DataFrame):
    """
    Costruisce un vettore 'base' nelle coordinate delle audio-features
    (acousticness, danceability, ...), usando euristiche su:
    - mood
    - activity
    - weather
    - part_of_day

    Inoltre, calcola:
      year_low  = anno quando l'utente aveva ~15 anni
      year_high = anno quando l'utente avrà ~30 anni
      year_pref = media (centro del range)

    Restituisce:
      base (dict delle feature target),
      (year_pref, year_low, year_high)
    """

    # Valori "base" neutri
    base = {
        "acousticness": 0.5,
        "danceability": 0.5,
        "energy": 0.5,
        "instrumentalness": 0.1,
        "liveness": 0.2,
        "loudness": -10.0,
        "speechiness": 0.05,
        "tempo": 120.0,
        "valence": 0.5,
        "duration_ms": df_global["duration_ms"].median() if "duration_ms" in df_global.columns else 210_000,
    }

    m = mood.lower().strip()
    a = activity.lower().strip()
    w = weather.lower().strip()
    d = part_of_day.lower().strip()

    # --- Mood ---
    if m in ["happy", "joy", "joyful", "positiv", "upbeat"]:
        base["valence"] += 0.2
        base["danceability"] += 0.2
        base["energy"] += 0.1
    elif m in ["sad", "melancholic", "low", "blue"]:
        base["valence"] -= 0.2
        base["energy"] -= 0.1
        base["acousticness"] += 0.2
    elif m in ["relaxed", "calm", "chill"]:
        base["energy"] -= 0.1
        base["acousticness"] += 0.2
        base["tempo"] -= 10
    elif m in ["angry", "aggressive"]:
        base["energy"] += 0.2
        base["valence"] -= 0.1
        base["tempo"] += 10
    # 🔹 Nuovi mood speciali legati alle categorie "problematiche"
    elif m in ["kids", "children", "nursery"]:
        # allegro, semplice, abbastanza ballabile
        base["valence"] += 0.25
        base["energy"] -= 0.05
        base["danceability"] += 0.10

    elif m in ["christmas", "xmas", "holiday"]:
        # caldo, medio-alto valence, spesso acustico
        base["valence"] += 0.20
        base["acousticness"] += 0.10
        base["tempo"] -= 3.0

    elif m in ["religious", "gospel"]:
        # più raccolto, acustico, non troppo veloce
        base["energy"] -= 0.10
        base["acousticness"] += 0.20
        base["tempo"] -= 5.0
    # --- Activity ---
    if a in ["party", "dancing", "dance"]:
        base["danceability"] += 0.25
        base["energy"] += 0.2
        base["valence"] += 0.1
        base["tempo"] += 15
    elif a in ["study", "focus", "work", "reading"]:
        base["energy"] -= 0.1
        base["acousticness"] += 0.15
        base["instrumentalness"] += 0.2
        base["speechiness"] -= 0.02
    elif a in ["gym", "workout", "run", "running"]:
        base["energy"] += 0.25
        base["tempo"] += 20
        base["danceability"] += 0.15
    elif a in ["commute", "travel"]:
        base["energy"] += 0.05
        base["tempo"] += 5

    # --- Weather ---
    if w in ["sunny", "clear"]:
        base["valence"] += 0.1
        base["energy"] += 0.05
    elif w in ["rainy", "storm", "stormy"]:
        base["valence"] -= 0.1
        base["acousticness"] += 0.1
    elif w in ["snow", "snowy"]:
        base["acousticness"] += 0.05
        base["instrumentalness"] += 0.05

    # --- Part of day ---
    if d in ["morning"]:
        base["energy"] += 0.05
        base["tempo"] += 5
    elif d in ["night", "late night"]:
        base["energy"] -= 0.05
        base["acousticness"] += 0.05
        base["tempo"] -= 5
    elif d in ["evening"]:
        base["valence"] += 0.05

    # Clamp (0..1) per feature in [0,1], loudness e tempo li lasciamo più liberi
    for k in ["acousticness", "danceability", "energy",
              "instrumentalness", "liveness",
              "speechiness", "valence"]:
        base[k] = float(np.clip(base[k], 0.0, 1.0))

    # --- Range temporale: 15–30 anni di vita dell'utente ---
    # Età ragionevole
    age_clipped = int(np.clip(age, 15, 70))

    # Utilizziamo la mediana degli anni del dataset come "oggi"
    current_year = 2025

    year_low = current_year - (age_clipped - 10)   # circa anno in cui aveva 15 anni
    year_high = current_year - (age_clipped - 30)  # circa quando avrà 30 anni
    if year_low > year_high:
        year_low, year_high = year_high, year_low
    year_pref = int((year_low + year_high) / 2)

    # Limitiamo agli anni effettivamente presenti nel dataset
    if "year" in df_global.columns and df_global["year"].notna().any():
        yrs = df_global["year"].dropna().values
        year_low = max(year_low, int(yrs.min()))
        year_high = min(year_high, int(yrs.max()))
        year_pref = int(np.clip(year_pref, yrs.min(), yrs.max()))

    return base, (year_pref, year_low, year_high)


# # Cell 7 - Temporal score & Popularity score

# In[284]:


# === 7. Funzioni di scoring: tempo, popolarità, meteo, orario, gusti utente ===

def temporal_score(years, year_pref, year_low, year_high, explorer: bool = False):
    """
    Punteggio temporale:
    - 1.0 dentro il range [year_low, year_high] (10–30 anni)
    - decrescente man mano che ci si allontana
    - se explorer=True, decadenza più lenta
    """
    years = np.asarray(years, dtype=float)

    score = np.ones_like(years, dtype=float)
    core_mask = (years >= year_low) & (years <= year_high)

    dist = np.zeros_like(years, dtype=float)
    left_mask = years < year_low
    right_mask = years > year_high

    dist[left_mask] = year_low - years[left_mask]
    dist[right_mask] = years[right_mask] - year_high

    base_decay = 0.12 if explorer else 0.25
    score[~core_mask] = np.exp(-base_decay * dist[~core_mask])

    return score


def popularity_score(pops, explorer: bool = False):
    """
    Normalizza la popolarità 0..1.
    - explorer=False: penalizza molto i brani troppo poco popolari
    - explorer=True: penalizza meno le basse popolarità
    """
    pops = np.asarray(pops, dtype=float)
    if np.isnan(pops).all():
        return np.ones_like(pops) * 0.5

    p_norm = (pops - pops.min()) / (pops.max() - pops.min() + 1e-8)

    if not explorer:
        return np.power(p_norm, 1.5)
    else:
        return np.sqrt(p_norm)


def compute_weather_score(df_local: pd.DataFrame, weather: str):
    """
    Punteggia i brani in base al meteo, usando valence/energy/acousticness.
    """
    w = weather.lower().strip()
    val = df_local["valence"].values if "valence" in df_local.columns else np.zeros(len(df_local))
    en = df_local["energy"].values if "energy" in df_local.columns else np.zeros(len(df_local))
    ac = df_local["acousticness"].values if "acousticness" in df_local.columns else np.zeros(len(df_local))

    if w in ["sunny", "clear"]:
        score = 0.6 * val + 0.4 * en
    elif w in ["rainy", "storm", "stormy"]:
        score = 0.6 * ac + 0.4 * (1 - val)
    elif w in ["snow", "snowy"]:
        score = 0.5 * ac + 0.5 * val
    else:
        score = 0.5 * val + 0.5 * en

    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score


def compute_part_of_day_score(df_local: pd.DataFrame, part_of_day: str):
    """
    Usa tempo/energy/acousticness per adattare musica alla fascia oraria.
    """
    d = part_of_day.lower().strip()
    tempo = df_local["tempo"].values if "tempo" in df_local.columns else np.zeros(len(df_local))
    en = df_local["energy"].values if "energy" in df_local.columns else np.zeros(len(df_local))
    ac = df_local["acousticness"].values if "acousticness" in df_local.columns else np.zeros(len(df_local))

    if d == "morning":
        score = 0.5 * en + 0.5 * (tempo / (tempo.max() + 1e-8))
    elif d in ["evening"]:
        score = 0.5 * en + 0.5 * (1 - ac)
    elif d in ["night", "late night"]:
        score = 0.7 * ac + 0.3 * (1 - tempo / (tempo.max() + 1e-8))
    else:
        score = 0.5 * en + 0.5 * (tempo / (tempo.max() + 1e-8))

    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score


def compute_user_taste_score(df_local: pd.DataFrame,
                             fav_artists=None,
                             explorer: bool = False):
    """
    Punteggio di affinità ai gusti utente basato su fav_artists (lista di stringhe).
    - match forte se l'artista è nei preferiti
    - match medio se artista 'simile' nel nome (rudimentale)
    - se nessun preferito, ritorna costante 0.5
    """
    if fav_artists is None or len(fav_artists) == 0:
        return np.ones(len(df_local)) * 0.5

    fav_clean = [a.strip().lower() for a in fav_artists if a.strip() != ""]
    if len(fav_clean) == 0:
        return np.ones(len(df_local)) * 0.5

    if "artist_name" not in df_local.columns:
        return np.ones(len(df_local)) * 0.5

    artists = df_local["artist_name"].astype(str).str.lower().values

    score = np.zeros(len(df_local), dtype=float)

    for i, art in enumerate(artists):
        if any(fa == art for fa in fav_clean):
            score[i] = 1.0
        elif any(fa in art for fa in fav_clean):
            score[i] = 0.7
        else:
            score[i] = 0.2

    # explorer = True → penalizzo meno il "non match"
    if explorer:
        score = 0.3 + 0.7 * score

    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score


# In[285]:


# === 8. Uso del MLP per predire subcluster + vicinanza tra subcluster ===

def build_full_feature_vector_from_profile(profile_dict: dict):
    """
    Converte il dizionario 'profile' in un vettore nella stessa
    order dei feature_cols.
    """
    return np.array([profile_dict[c] for c in feature_cols], dtype=float)


def predict_subcluster_from_profile(profile_dict: dict):
    """
    Usa l'MLP addestrato per predire il subcluster più probabile
    dato il profilo utente nelle coordinate delle audio-features.
    """
    x = build_full_feature_vector_from_profile(profile_dict).reshape(1, -1)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    subcluster_pred = le_classes[pred_idx]
    return subcluster_pred, probs


def find_neighbour_subclusters(profile_dict: dict, top_k: int = 3):
    """
    Trova i subclusters più vicini al profilo utente nello spazio
    delle medie delle audio-features (subcluster_summary).
    """
    x = np.array([profile_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    centers = subcluster_summary[feature_cols].values
    centers_scaled = scaler.transform(centers)

    sims = cosine_similarity(x_scaled, centers_scaled)[0]
    order = np.argsort(-sims)

    neigh_subclusters = subcluster_summary.index[order[:top_k]].tolist()
    return neigh_subclusters, sims[order[:top_k]], order[:top_k]


# In[286]:


def _is_fav_artist_series(artist_series: pd.Series, fav_set):
    """
    Ritorna una Series booleana: True se il nome artista contiene
    almeno uno dei favourite artists (in lowercase).
    """
    if not fav_set:
        return pd.Series(False, index=artist_series.index)

    lower = artist_series.astype(str).str.lower()

    def check_name(name: str) -> bool:
        return any(fav in name for fav in fav_set)

    return lower.apply(check_name)


# # Cell 10 - Main

# In[316]:


# === 9. Funzione principale di raccomandazione ===

def recommend_playlist(mood: str,
                       activity: str,
                       part_of_day: str,
                       weather: str,
                       age: int,
                       explorer: bool,
                       n: int = 10,
                       fav_artists=None,
                       language_prefs=None):  # <<< AGGIUNTO
    """
    Ritorna un DataFrame con le top-n tracce consigliate.

    Input utente:
      - mood:
          * mood "normali" (happy, sad, relaxed, angry, ecc.)
          * mood speciali:
              - "kids", "children"  → preferisci tracce con is_kids / is_nursery
              - "nursery"           → preferisci tracce con is_nursery
              - "christmas", "xmas", "holiday" → preferisci tracce con is_christmas
              - "religious", "gospel"          → preferisci tracce con is_religious
      - activity
      - part_of_day
      - weather
      - age
      - explorer (True/False)
      - fav_artists: lista di stringhe (es. ["Taylor Swift", "Drake"])
      - language_prefs: lista di codici lingua (es. ["it"], ["en", "it"])
                        se None o lista vuota → nessun vincolo di lingua   # <<< DOC
      - n: numero di canzoni
    """

    #1. Validate MOOD
    VALID_MOODS = [
        # Fully supported moods
        'happy', 'joy', 'joyful', 'positiv', 'upbeat',
        'sad', 'melancholic', 'low', 'blue',
        'relaxed', 'calm', 'chill',
        'angry', 'aggressive',

        # Special moods
        'kids', 'children', 'nursery',
        'christmas', 'xmas', 'holiday',
        'religious', 'gospel'
    ]

    mood_clean = str(mood).strip().lower()
    if mood_clean not in VALID_MOODS:
        raise ValueError(
            f"Invalid mood: '{mood}'. Valid moods: {', '.join(VALID_MOODS)}"
        )


    #2. Validate ACTIVITY
    VALID_ACTIVITIES = [
        'party', 'dancing', 'dance',
        'study', 'focus', 'work', 'reading',
        'gym', 'workout', 'run', 'running',
        'commute', 'travel'
    ]

    activity_clean = str(activity).strip().lower()
    if activity_clean not in VALID_ACTIVITIES:
        raise ValueError(
            f"Invalid activity: '{activity}'. Valid activities: {', '.join(VALID_ACTIVITIES)}"
        )


    #3. Validate PART_OF_DAY
    VALID_TIMES = [
        'morning',
        'night', 'late night',
        'evening'
    ]

    time_clean = str(part_of_day).strip().lower()
    if time_clean not in VALID_TIMES:
        raise ValueError(
            f"Invalid part_of_day: '{part_of_day}'. Valid times: {', '.join(VALID_TIMES)}"
        )


    #4. Validate WEATHER
    VALID_WEATHER = [
        'sunny', 'clear',
        'rainy', 'storm', 'stormy',
        'snow', 'snowy'
    ]

    weather_clean = str(weather).strip().lower()
    if weather_clean not in VALID_WEATHER:
        raise ValueError(
            f"Invalid weather: '{weather}'. Valid weather types: {', '.join(VALID_WEATHER)}"
        )
    # 5. Validate AGE
    if not isinstance(age, (int, float)):
        raise TypeError(f"Age must be a number, received: {type(age).__name__}")

    if age < 5 or age > 120:
        raise ValueError(f"Invalid age: {age}. Must be between 5 and 120.")

    # 6. Validate EXPLORER
    if not isinstance(explorer, bool):
        raise TypeError(f"Explorer must be True or False, received: {type(explorer).__name__}")

    # 7. Validate N
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, received: {type(n).__name__}")

    if n < 1 or n > 100:
        raise ValueError(f"Invalid n: {n}. Must be between 1 and 100.")
    # 8. Validate LANGUAGE_PREFS
    VALID_LANGUAGES = ['en', 'de', 'es', 'it', 'pt', 'fr', 'other']
        
    if language_prefs is not None:
        if not isinstance(language_prefs, list):
            raise TypeError(f"language_prefs must be a list, received: {type(language_prefs).__name__}")
        
        invalid_langs = [
            lang for lang in language_prefs 
            if lang and str(lang).strip().lower() not in VALID_LANGUAGES
        ]
        
        if invalid_langs:
            raise ValueError(
                f"Invalid language codes: {', '.join(invalid_langs)}. "
                f"Valid languages: {', '.join(VALID_LANGUAGES)}"
            )
    # END OF VALIDATION
    if fav_artists is None:
        fav_artists = []
    if language_prefs is None:          # <<< AGGIUNTO
        language_prefs = []             # <<<

    # 1) Profilo target + range temporale [year_low, year_high]
    profile, (year_pref, year_low, year_high) = build_target_profile(
        mood=mood,
        activity=activity,
        weather=weather,
        part_of_day=part_of_day,
        age=age,
        explorer=explorer,
        df_global=df
    )

    # 2) df_local di base: partiamo da tutto il dataset
    df_local = df.copy()

    # --- Gestione canzoni "problematiche" in base al mood ---

    mood_clean = (mood or "").strip().lower()
    is_special_mood = mood_clean in MOOD_FLAG_MAP   # <<< NUOVO: flag mood speciale

    if any(col in df_local.columns for col in PROBLEMATIC_FLAG_COLS):
        # Caso 1: mood speciale (kids, christmas, nursery, religious...):
        #   → teniamo SOLO le canzoni con le flag corrispondenti
        if is_special_mood:
            cols = [c for c in MOOD_FLAG_MAP[mood_clean] if c in df_local.columns]

            if cols:
                mask_special = np.zeros(len(df_local), dtype=bool)
                for c in cols:
                    mask_special |= df_local[c].fillna(False).to_numpy().astype(bool)

                df_local = df_local[mask_special].copy()

        # Caso 2: mood "normale"
        #   → escludiamo tutte le canzoni marcate come kids / christmas / nursery / religious
        else:
            mask_keep = np.ones(len(df_local), dtype=bool)
            for c in PROBLEMATIC_FLAG_COLS:
                if c in df_local.columns:
                    mask_keep &= ~df_local[c].fillna(False).to_numpy().astype(bool)
            df_local = df_local[mask_keep].copy()

    # Se per qualche motivo siamo rimasti senza canzoni (es. nessuna christmas nel dataset),
    # facciamo un fallback: torniamo al df non filtrato, ma almeno senza kids.
    if df_local.empty:
        df_local = df.copy()
        if "is_kids" in df_local.columns:
            df_local = df_local[df_local["is_kids"] == False].copy()

    # Escludiamo comunque le tracce "per bambini" se NON siamo in un mood kids/nursery
    if mood_clean not in ["kids", "children", "nursery"]:
        if "is_kids" in df_local.columns:
            df_local = df_local[df_local["is_kids"] == False].copy()

    # --- Filtro LINGUA vincolante (usa colonna main_language) ---   # <<< BLOCCO NUOVO
    langs_clean = {str(l).strip().lower() for l in language_prefs if l and str(l).strip() != ""}
    if langs_clean and ("main_language" in df_local.columns):
        mask_lang = df_local["main_language"].astype(str).str.lower().isin(langs_clean)
        df_lang_filtered = df_local[mask_lang].copy()
        if not df_lang_filtered.empty:
            df_local = df_lang_filtered
        else:
            print(f"⚠️ Nessun brano trovato per le lingue richieste {langs_clean}. "
                  f"Ignoro il filtro lingua e uso tutte le lingue disponibili.")
    # ---------------------------------------------------------------

    # Mappiamo df_local su X_scaled (stesse righe)
    mask_local = df.index.isin(df_local.index)
    X_local = X_scaled[mask_local]   # shape = (len(df_local), n_features)

    # 3) Predizione subcluster dal MLP
    subcluster_pred, probs = predict_subcluster_from_profile(profile)

    # 4) Subcluster vicini
    neighbour_subclusters, sim_sub, idx_order = find_neighbour_subclusters(profile, top_k=3)
    if subcluster_pred not in neighbour_subclusters:
        neighbour_subclusters = [subcluster_pred] + neighbour_subclusters[:-1]

    # 5) Mood similarity (cosine tra profilo target e X_local)
    x_target_raw = build_full_feature_vector_from_profile(profile).reshape(1, -1)
    x_target_scaled = scaler.transform(x_target_raw)
    mood_sim = cosine_similarity(x_target_scaled, X_local)[0]  # shape = (len(df_local),)

    # 6) Cluster bonus (basato su df_local)
    sub = df_local["subcluster"].astype(str)
    macro = df_local["macro_cluster"]

    macro_pred = int(str(subcluster_pred).split("_")[0])

    if not explorer:
        same_sub = 1.0
        neighbour = 0.8
        same_macro = 0.4
        other_macro = 0.1
    else:
        same_sub = 1.0
        neighbour = 0.9
        same_macro = 0.6
        other_macro = 0.3

    cluster_bonus = np.zeros(len(df_local), dtype=float)
    cluster_bonus[sub == subcluster_pred] = same_sub
    cluster_bonus[(sub.isin(neighbour_subclusters)) & (sub != subcluster_pred)] = neighbour
    cluster_bonus[(~sub.isin(neighbour_subclusters)) & (macro == macro_pred)] = same_macro
    cluster_bonus[(macro != macro_pred)] = other_macro

    # 7) Temporal & popularity (sempre su df_local)
    if "year" in df_local.columns:
        years_local = df_local["year"].fillna(year_pref).values
    else:
        years_local = np.ones(len(df_local)) * year_pref

    if "popularity" in df_local.columns:
        pops_local = df_local["popularity"].fillna(df_local["popularity"].mean()).values
    else:
        pops_local = np.ones(len(df_local)) * 50

    # >>> QUI IL CAMBIO IMPORTANTE: niente selezione temporale per i mood speciali

    if is_special_mood:
        # Nessuna preferenza temporale: tutte le epoche vanno bene.
        time_score_raw = np.ones(len(df_local), dtype=float)
    else:
        # Usa normalmente il range derivato dall'età
        time_score_raw = temporal_score(years_local, year_pref, year_low, year_high, explorer)

    pop_score_raw = popularity_score(pops_local, explorer=explorer)

    # 8) Weather, part_of_day & gusti utente (su df_local)
    weather_score_raw = compute_weather_score(df_local, weather)
    day_score_raw = compute_part_of_day_score(df_local, part_of_day)
    user_taste_raw = compute_user_taste_score(df_local, fav_artists=fav_artists, explorer=explorer)

    # 9) Normalizzazioni (tutti 0..1) per combinare bene i pesi
    def _norm(arr):
        arr = np.asarray(arr, dtype=float)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    mood_sim_norm = _norm(mood_sim)
    cluster_bonus_norm = _norm(cluster_bonus)
    time_score_norm = _norm(time_score_raw)
    pop_score_norm = _norm(pop_score_raw)
    weather_score_norm = _norm(weather_score_raw)
    day_score_norm = _norm(day_score_raw)
    user_taste_norm = _norm(user_taste_raw)

    # combinazione mood + cluster
    mood_cluster_score = 0.6 * mood_sim_norm + 0.4 * cluster_bonus_norm

    # 10) Pesi (gusti utente > mood/cluster > tempo > pop > meteo > orario)
    w_taste = 0.35
    w_mood_cluster = 0.25
    w_time = 0.20
    w_pop = 0.10
    w_weather = 0.05
    w_day = 0.05

    final_score = (
        w_taste * user_taste_norm +
        w_mood_cluster * mood_cluster_score +
        w_time * time_score_norm +
        w_pop * pop_score_norm +
        w_weather * weather_score_norm +
        w_day * day_score_norm
    )

    # 11) Costruzione DataFrame risultati (stesso length di df_local)
    result = df_local.copy()
    result["score"] = final_score
    result["user_taste_score"] = user_taste_norm
    result["mood_cluster_score"] = mood_cluster_score
    result["time_score"] = time_score_norm
    result["pop_score"] = pop_score_norm
    result["weather_score"] = weather_score_norm
    result["day_score"] = day_score_norm

    # 12) Filtri "ragionevoli" quando explorer=False
    #     → disattivati per i mood speciali (niente filtro per anno basato sull'età)
    if (not explorer) and (not is_special_mood) and "year" in result.columns and "popularity" in result.columns:
        years_col = result["year"].fillna(year_pref)
        pops_col = result["popularity"].fillna(result["popularity"].mean())

        margin = 3  # allarghiamo un po' il range rispetto [year_low, year_high]
        low = max(year_low - margin, years_col.min())
        high = min(year_high + margin, years_col.max())

        mask = (
            (years_col >= low) &
            (years_col <= high) &
            (pops_col >= 30)
        )
        result = result[mask].copy()

    # 13) Rimuovi duplicati di track_id se necessario
    if "track_id" in result.columns:
        result = result.drop_duplicates("track_id")
    # SMART DEDUPLICATION
    if "track_name" in result.columns and "artist_name" in result.columns:
        import re
        n_before = len(result)

        def normalize_track(name):
            name = str(name).lower().strip()
            patterns = [
                r'\s*-\s*remaster.*$',
                r'\s*\(remaster.*\)$',
                r'\s*-\s*single version.*$',
                r'\s*\(single version.*\)$',
                r'\s*-\s*edited version.*$',
                r'\s*\(feat\..*\)$',           
                r'\s*\(ft\..*\)$',             
                r'\s*\(featuring.*\)$',        
                r'\s*-\s*feat\..*$',           
                r'\s*\[feat\..*\]$',           
            ]
            for p in patterns:
                name = re.sub(p, '', name, flags=re.IGNORECASE)
            return name.strip()

        def get_main_artist(artist_str):
            artist = str(artist_str).lower().strip()
            artist = re.split(r'[,&/]', artist)[0].strip()
            artist = re.sub(r'\[|\]', '', artist)
            return artist

        result['song_signature'] = (
            result['track_name'].apply(normalize_track) +
            '___' +
            result['artist_name'].apply(get_main_artist)
        )

        # SORT by score BEFORE deduplication
        result = result.sort_values('score', ascending=False)

        # Deduplication - keep first (highest score)
        result = result.drop_duplicates(subset='song_signature', keep='first')
        
        # Drop the signature column
        result = result.drop(columns=['song_signature'])

        n_after = len(result)
        print(f"Removed {n_before - n_after} duplicates (semantic name + main artist)")
    # 14) Varietà artisti: max 3 brani per artista (prima di miscelare fav/others)
    if "artist_name" in result.columns:
        result["artist_rank"] = result.groupby("artist_name").cumcount()
        result = result[result["artist_rank"] < 3].drop(columns=["artist_rank"])

    # 15) Ordina per score (ranking globale)
    result_sorted = result.sort_values("score", ascending=False)

    # 16) Composizione obbligatoria tra artisti preferiti e altri
    if ("artist_name" in result_sorted.columns) and fav_artists:
        fav_set = {a.strip().lower() for a in fav_artists if a and a.strip() != ""}

        # mask artisti preferiti con match "contains"
        mask_fav = _is_fav_artist_series(result_sorted["artist_name"], fav_set)

        df_fav = result_sorted[mask_fav]
        df_other = result_sorted[~mask_fav]

        # target minimo di brani con artisti diversi dai preferiti
        if explorer:
            min_other_ratio = 0.5   # almeno 50% altri artisti
        else:
            min_other_ratio = 0.3   # almeno 30% altri artisti

        target_other = int(np.ceil(min_other_ratio * n))

        # quanti "altri" riusciamo effettivamente a prendere?
        n_other = min(target_other, len(df_other))
        pick_other = df_other.head(n_other)

        # riempi il resto con i preferiti
        remaining_slots = n - len(pick_other)
        pick_fav = df_fav.head(remaining_slots)

        # se ancora mancano brani, riempi col resto del ranking globale
        already_idx = set(pick_other.index) | set(pick_fav.index)
        leftover = result_sorted[~result_sorted.index.isin(already_idx)].head(
            n - len(pick_other) - len(pick_fav)
        )

        top_result = pd.concat([pick_other, pick_fav, leftover]).head(n)

    else:
        # caso senza artist_name o senza fav_artists → semplice top-n
        top_result = result_sorted.head(n)

    # 17) Colonne da mostrare
    cols_show = [
        "track_id", "track_name", "artist_name", "genre", "year", "popularity",
        "macro_cluster", "subcluster", "subcluster_label",
        "score", "user_taste_score", "mood_cluster_score",
        "time_score", "pop_score", "weather_score", "day_score"
    ]
    cols_exist = [c for c in cols_show if c in top_result.columns]
    top_result = top_result[cols_exist]

    # 18) Log di debug
    print(
        f"User input → mood='{mood}', activity='{activity}', part_of_day='{part_of_day}', "
        f"weather='{weather}', age={age}, explorer={explorer}, "
        f"fav_artists={fav_artists}, language_prefs={language_prefs}"  # <<< LOG AGGIORNATO
    )
    print(f"Predicted subcluster: {subcluster_pred}")
    print("Neighbour subclusters:", neighbour_subclusters)
    print(f"Preferred year center: {year_pref}, range=[{year_low}, {year_high}]")
    print("Candidate pool size (after mood/problematic/lingua filters):", len(result_sorted))

    if ("artist_name" in top_result.columns) and fav_artists:
        fav_set = {a.strip().lower() for a in fav_artists if a and a.strip() != ""}
        mask_fav_pl = _is_fav_artist_series(top_result["artist_name"], fav_set)
        n_fav_final = mask_fav_pl.sum()
        print(
            f"In playlist: {n_fav_final}/{n} brani di favourite artists "
            f"({n - n_fav_final} di altri artisti)."
        )

    return top_result