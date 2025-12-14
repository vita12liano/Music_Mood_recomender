import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn
import torch.nn.functional as F

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

# Caricamento dataset
df = pd.read_csv(data_path)

# === Scaler, label encoder, modello MLP ===

scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

le_classes = np.load(le_classes_path, allow_pickle=True).astype(str)
num_classes = len(le_classes)

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

# === Matrice feature globale & summary per subcluster ===

X_raw = df[feature_cols].values
X_scaled = scaler.transform(X_raw)

subcluster_summary = (
    df.groupby("subcluster")[feature_cols]
      .mean()
      .sort_index()
)

# Flag problematici nel dataset
PROBLEMATIC_FLAG_COLS = [
    "is_kids",
    "is_christmas",
    "is_nursery",
    "is_religious",
    "is_soundtrack",
]

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
      year_low  = anno quando l'utente aveva ~10–15 anni
      year_high = anno quando l'utente avrà ~30 anni
      year_pref = media (centro del range)

    Le feature sono centrate così (indicativo):
      - acousticness ↑ per input più tranquilli
      - danceability ↑ per input più tranquilli
      - energy ↑ per input più aggressivi/energetici
      - instrumentalness ↑ sia per tranquilli che per energetici
      - liveness ↑ per input più aggressivi
      - loudness ↑ per input più energetici
      - speechiness ↑ per input più tranquilli
      - tempo ↑ per input più aggressivi/energetici
      - valence: 0 ≈ sad, 0.5 ≈ angry, 1 ≈ happy
      - duration_ms: più lunga per mood/attività tranquille, più corta per gym/party

    Restituisce:
      base (dict delle feature target),
      (year_pref, year_low, year_high)
    """

    # --- Valori "base" neutri ---
    base = {
        "acousticness": 0.5,
        "danceability": 0.5,
        "energy": 0.5,
        "instrumentalness": 0.1,
        "liveness": 0.2,
        "loudness": -10.0,   # dBFS (valori più alti = meno negativi = più forti)
        "speechiness": 0.05,
        "tempo": 120.0,
        "valence": 0.5,
        "duration_ms": df_global["duration_ms"].median()
        if "duration_ms" in df_global.columns else 210_000,
    }

    m = (mood or "").lower().strip()
    a = (activity or "").lower().strip()
    w = (weather or "").lower().strip()
    d = (part_of_day or "").lower().strip()

    # ------------------------------------------------------------------
    # 1) Costruiamo dei punteggi astratti:
    #    - calm_level   → quanto l'input è "tranquillo"
    #    - energy_level → quanto è "energetico"
    #    - agg_level    → quanto è "aggressivo"
    #    Questi livelli poi guidano TUTTE le feature.
    # ------------------------------------------------------------------
    calm_level = 0.0
    energy_level = 0.0
    agg_level = 0.0

    # -----------------------
    # 1a. Mood → valence + livelli
    # -----------------------
    # Valence ancorata ai valori richiesti:
    #   sad   → ~0.1–0.2
    #   angry → 0.5
    #   happy → ~0.9
    #   relaxed → intermedio "felice ma soft" (~0.7)
    if m == "happy":
        base["valence"] = 0.9
        energy_level += 1.0
    elif m == "sad":
        base["valence"] = 0.15
        calm_level += 1.0
    elif m in ["relaxed", "calm", "chill"]:
        base["valence"] = 0.7
        calm_level += 1.5
    elif m in ["angry", "aggressive"]:
        base["valence"] = 0.5
        agg_level += 1.5
        energy_level += 1.0
    # Mood speciali (anche per filtraggio "problematic tracks")
    elif m in ["kids", "children", "nursery"]:
        base["valence"] = 0.9
        calm_level += 1.0
    elif m in ["christmas", "xmas", "holiday"]:
        base["valence"] = 0.8
        calm_level += 1.0
    elif m in ["religious", "gospel"]:
        base["valence"] = 0.6
        calm_level += 1.0
    else:
        # mood generico
        base["valence"] = 0.5

    # -----------------------
    # 1b. Activity → livelli
    # -----------------------
    calm_activities = ["study", "focus", "work", "reading", "chill", "chilling", "commute", "travel"]
    energetic_activities = ["gym", "workout", "run", "running", "party", "dancing", "dance"]

    if a in calm_activities:
        calm_level += 1.0
    if a in energetic_activities:
        energy_level += 1.5
        if a in ["party", "dancing", "dance"]:
            agg_level += 0.5  # party un po' più "spinto"

    # -----------------------
    # 1c. Weather & part_of_day come piccoli aggiustamenti
    # -----------------------
    # Meteo:
    if w in ["sunny", "clear"]:
        base["valence"] += 0.05
        energy_level += 0.3
    elif w in ["rainy", "storm", "stormy"]:
        base["valence"] -= 0.05
        calm_level += 0.3
    elif w in ["snow", "snowy"]:
        calm_level += 0.2

    # Fascia oraria:
    if d == "morning":
        energy_level += 0.2
    elif d in ["evening"]:
        # sera leggermente più "energetica" ma anche adatta a chill
        energy_level += 0.1
        calm_level += 0.1
    elif d in ["night", "late night"]:
        calm_level += 0.5
        energy_level -= 0.2

    # Clamp livelli a [0, 2] per evitare eccessi
    calm_level = float(np.clip(calm_level, 0.0, 2.0))
    energy_level = float(np.clip(energy_level, 0.0, 2.0))
    agg_level = float(np.clip(agg_level, 0.0, 2.0))

    # ------------------------------------------------------------------
    # 2) Mappiamo questi livelli sulle singole feature
    # ------------------------------------------------------------------

    # acousticness: ↑ con calma, ↓ con energia
    base["acousticness"] += 0.20 * calm_level - 0.10 * energy_level

    # danceability: richiesta ↑ per input più tranquilli
    base["danceability"] += 0.20 * calm_level - 0.05 * agg_level

    # energy: ↑ con energia/aggressività, ↓ con calma
    base["energy"] += 0.25 * energy_level + 0.20 * agg_level - 0.20 * calm_level

    # instrumentalness: ↑ sia per tranquilli che per energetici
    base["instrumentalness"] += 0.10 * calm_level + 0.10 * energy_level

    # liveness: ↑ per aggressivi / live-feel
    base["liveness"] += 0.15 * agg_level + 0.05 * energy_level

    # loudness (dBFS): valori più alti = meno negativi → più forti
    base["loudness"] += 3.0 * energy_level + 2.0 * agg_level - 2.0 * calm_level

    # speechiness: ↑ per input più tranquilli
    base["speechiness"] += 0.15 * calm_level - 0.10 * agg_level

    # tempo (BPM): ↑ per input aggressivi/energetici, ↓ per molto tranquilli
    base["tempo"] += 10.0 * energy_level + 8.0 * agg_level - 6.0 * calm_level

    # duration_ms:
    #   - più lunga per mood/attività tranquille
    #   - più corta per gym/party
    dur = base["duration_ms"]
    dur += 30_000 * calm_level   # +30s per livello di calma
    dur -= 20_000 * energy_level  # -20s per livello di energia
    base["duration_ms"] = max(90_000, float(dur))  # almeno 90s

    # ------------------------------------------------------------------
    # 3) SPECIAL STEERING per subcluster rari
    #    (stesse logiche di prima, ma sopra abbiamo dato un profilo sensato)
    #    Questi preset sovrascrivono il profilo quando il pattern di input matcha.
    # ------------------------------------------------------------------

    # 0_0 – Short Spoken Calm
    # relaxed + (study/work/reading) + evening/night
    if (
        m == "relaxed"
        and a in ["reading", "study", "work"]
        and d in ["evening", "night"]
    ):
        base = {
            "acousticness": 0.467771,
            "danceability": 0.671075,
            "energy": 0.255682,
            "instrumentalness": 0.005130,
            "liveness": 0.330862,
            "loudness": -18.688232,
            "speechiness": 0.914860,
            "tempo": 107.482467,
            "valence": 0.544188,
            "duration_ms": 182_201.0,
        }

    # 1_0 – Deep Calm & Minimal
    # sad + (study/work/reading) + evening/night
    if (
        m == "sad"
        and a in ["reading", "study", "work"]
        and d in ["evening", "night"]
    ):
        base = {
            "acousticness": 0.905267,
            "danceability": 0.308582,
            "energy": 0.156997,
            "instrumentalness": 0.800893,
            "liveness": 0.167886,
            "loudness": -20.604054,
            "speechiness": 0.048183,
            "tempo": 96.757403,
            "valence": 0.178934,
            "duration_ms": 326_022.0,
        }

    # 1_2 – Epic Intense
    # angry/happy + (gym/run/party) + evening/night
    if (
        m in ["angry", "happy"]
        and a in ["gym", "workout", "run", "running", "party", "dancing", "dance"]
        and d in ["evening", "night"]
    ):
        base = {
            "acousticness": 0.123278,
            "danceability": 0.531765,
            "energy": 0.705924,
            "instrumentalness": 0.676251,
            "liveness": 0.195493,
            "loudness": -9.490165,
            "speechiness": 0.058513,
            "tempo": 123.730674,
            "valence": 0.556025,
            "duration_ms": 262_283.0,
        }

    # 2_3 – Soft Sad Calm
    # sad/relaxed + (chill/commute) + evening/night + meteo non "super soleggiato"
    if (
        m in ["sad", "relaxed"]
        and a in ["chill", "chilling", "commute", "travel"]
        and d in ["evening", "night"]
        and w in ["rainy", "snow", "snowy", "cloudy", "stormy"]
    ):
        base = {
            "acousticness": 0.866599,
            "danceability": 0.406124,
            "energy": 0.196410,
            "instrumentalness": 0.024760,
            "liveness": 0.163935,
            "loudness": -15.543252,
            "speechiness": 0.044752,
            "tempo": 96.954149,
            "valence": 0.285893,
            "duration_ms": 209_920.0,
        }

    # 2_5 – Energetic Live Mood
    # happy/relaxed + party/gym + evening/night + meteo buono
    if (
        m in ["happy", "relaxed"]
        and a in ["party", "gym", "workout", "run", "running", "dancing", "dance"]
        and d in ["evening", "night"]
        and w in ["sunny", "clear"]
    ):
        base = {
            "acousticness": 0.430542,
            "danceability": 0.502758,
            "energy": 0.602284,
            "instrumentalness": 0.029864,
            "liveness": 0.718388,
            "loudness": -9.929079,
            "speechiness": 0.101848,
            "tempo": 118.597994,
            "valence": 0.546971,
            "duration_ms": 254_856.0,
        }

    # ------------------------------------------------------------------
    # 4) Clamp delle feature [0,1] dove ha senso
    # ------------------------------------------------------------------
    for k in ["acousticness", "danceability", "energy",
              "instrumentalness", "liveness",
              "speechiness", "valence"]:
        base[k] = float(np.clip(base[k], 0.0, 1.0))

    # --- Range temporale: 15–30 anni di vita dell'utente ---
    age_clipped = int(np.clip(age, 15, 70))
    current_year = 2025

    year_low = current_year - (age_clipped - 10)
    year_high = current_year - (age_clipped - 30)
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
# === 7. Funzioni di scoring ===

def temporal_score(years, year_pref, year_low, year_high, explorer: bool = False):
    years = np.asarray(years, dtype=float)

    score = np.ones_like(years, dtype=float)
    core_mask = (years >= year_low) & (years <= year_high)

    dist = np.zeros_like(years, dtype=float)
    left_mask = years < year_low
    right_mask = years > year_high

    dist[left_mask] = year_low - years[left_mask]
    dist[right_mask] = years[right_mask] - year_high
    base_decay = 0.05 if explorer else 0.25
    score[~core_mask] = np.exp(-base_decay * dist[~core_mask])

    return score


def popularity_score(pops, explorer: bool = False):
    pops = np.asarray(pops, dtype=float)
    if np.isnan(pops).all():
        return np.ones_like(pops) * 0.5

    p_norm = (pops - pops.min()) / (pops.max() - pops.min() + 1e-8)

    if not explorer:
        return np.power(p_norm, 1.5)
    else:
        return np.sqrt(p_norm)


def compute_weather_score(df_local: pd.DataFrame, weather: str):
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

    if explorer:
        score = 0.3 + 0.7 * score

    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score

# === 8. Uso del MLP per predire subcluster + vicinanza ===

def build_full_feature_vector_from_profile(profile_dict: dict):
    return np.array([profile_dict[c] for c in feature_cols], dtype=float)


def predict_subcluster_from_profile(profile_dict: dict):
    """
    Usa l'MLP addestrato per predire il subcluster più probabile
    dato il profilo utente.

    Qui applichiamo anche un piccolo bias ai logits per dare
    una chance in più ai subcluster rari.
    """
    x = build_full_feature_vector_from_profile(profile_dict).reshape(1, -1)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))

        # Bias morbido per cluster rari
        cluster_prior = {
            "0_0": 0.4,
            "1_0": 0.5,
            "1_2": 0.6,
            "2_3": 0.5,
            "2_5": 0.8,
        }
        prior = torch.zeros_like(logits)
        for idx, label in enumerate(le_classes):
            bias = cluster_prior.get(str(label), 0.0)
            if bias != 0.0:
                prior[0, idx] = bias

        logits = logits + prior
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    subcluster_pred = le_classes[pred_idx]
    return subcluster_pred, probs


def find_neighbour_subclusters(profile_dict: dict, top_k: int = 3):
    x = np.array([profile_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    centers = subcluster_summary[feature_cols].values
    centers_scaled = scaler.transform(centers)

    sims = cosine_similarity(x_scaled, centers_scaled)[0]
    order = np.argsort(-sims)

    neigh_subclusters = subcluster_summary.index[order[:top_k]].tolist()
    return neigh_subclusters, sims[order[:top_k]], order[:top_k]


def _is_fav_artist_series(artist_series: pd.Series, fav_set):
    if not fav_set:
        return pd.Series(False, index=artist_series.index)

    lower = artist_series.astype(str).str.lower()

    def check_name(name: str) -> bool:
        return any(fav in name for fav in fav_set)

    return lower.apply(check_name)

# === 9. Funzione principale di raccomandazione ===

def recommend_playlist(mood: str,
                       activity: str,
                       part_of_day: str,
                       weather: str,
                       age: int,
                       explorer: bool,
                       n: int = 10,
                       fav_artists=None,
                       language_prefs=None):

    if fav_artists is None:
        fav_artists = []
    if language_prefs is None:
        language_prefs = []

    profile, (year_pref, year_low, year_high) = build_target_profile(
        mood=mood,
        activity=activity,
        weather=weather,
        part_of_day=part_of_day,
        age=age,
        explorer=explorer,
        df_global=df
    )

    df_local = df.copy()

    mood_clean = (mood or "").strip().lower()
    is_special_mood = mood_clean in MOOD_FLAG_MAP

    if any(col in df_local.columns for col in PROBLEMATIC_FLAG_COLS):
        if is_special_mood:
            cols = [c for c in MOOD_FLAG_MAP[mood_clean] if c in df_local.columns]
            if cols:
                mask_special = np.zeros(len(df_local), dtype=bool)
                for c in cols:
                    mask_special |= df_local[c].fillna(False).to_numpy().astype(bool)
                df_local = df_local[mask_special].copy()
        else:
            mask_keep = np.ones(len(df_local), dtype=bool)
            for c in PROBLEMATIC_FLAG_COLS:
                if c in df_local.columns:
                    mask_keep &= ~df_local[c].fillna(False).to_numpy().astype(bool)
            df_local = df_local[mask_keep].copy()

    if df_local.empty:
        df_local = df.copy()
        if "is_kids" in df_local.columns:
            df_local = df_local[df_local["is_kids"] == False].copy()

    if mood_clean not in ["kids", "children", "nursery"]:
        if "is_kids" in df_local.columns:
            df_local = df_local[df_local["is_kids"] == False].copy()

    langs_clean = {str(l).strip().lower() for l in language_prefs if l and str(l).strip() != ""}
    if langs_clean and ("main_language" in df_local.columns):
        mask_lang = df_local["main_language"].astype(str).str.lower().isin(langs_clean)
        df_lang_filtered = df_local[mask_lang].copy()
        if not df_lang_filtered.empty:
            df_local = df_lang_filtered
        else:
            print(f"⚠️ Nessun brano trovato per le lingue richieste {langs_clean}. "
                  f"Ignoro il filtro lingua e uso tutte le lingue disponibili.")

    mask_local = df.index.isin(df_local.index)
    X_local = X_scaled[mask_local]

    subcluster_pred, probs = predict_subcluster_from_profile(profile)

    neighbour_subclusters, sim_sub, idx_order = find_neighbour_subclusters(profile, top_k=3)
    if subcluster_pred not in neighbour_subclusters:
        neighbour_subclusters = [subcluster_pred] + neighbour_subclusters[:-1]

    x_target_raw = build_full_feature_vector_from_profile(profile).reshape(1, -1)
    x_target_scaled = scaler.transform(x_target_raw)
    mood_sim = cosine_similarity(x_target_scaled, X_local)[0]

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

    if "year" in df_local.columns:
        years_local = df_local["year"].fillna(year_pref).values
    else:
        years_local = np.ones(len(df_local)) * year_pref

    if "popularity" in df_local.columns:
        pops_local = df_local["popularity"].fillna(df_local["popularity"].mean()).values
    else:
        pops_local = np.ones(len(df_local)) * 50

    if is_special_mood:
        time_score_raw = np.ones(len(df_local), dtype=float)
    else:
        time_score_raw = temporal_score(years_local, year_pref, year_low, year_high, explorer)

    pop_score_raw = popularity_score(pops_local, explorer=explorer)
    weather_score_raw = compute_weather_score(df_local, weather)
    day_score_raw = compute_part_of_day_score(df_local, part_of_day)
    user_taste_raw = compute_user_taste_score(df_local, fav_artists=fav_artists, explorer=explorer)

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

    mood_cluster_score = 0.6 * mood_sim_norm + 0.4 * cluster_bonus_norm

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

    result = df_local.copy()
    result["score"] = final_score
    result["user_taste_score"] = user_taste_norm
    result["mood_cluster_score"] = mood_cluster_score
    result["time_score"] = time_score_norm
    result["pop_score"] = pop_score_norm
    result["weather_score"] = weather_score_norm
    result["day_score"] = day_score_norm

    if (not explorer) and (not is_special_mood) and "year" in result.columns and "popularity" in result.columns:
        years_col = result["year"].fillna(year_pref)
        pops_col = result["popularity"].fillna(result["popularity"].mean())

        margin = 3
        low = max(year_low - margin, years_col.min())
        high = min(year_high + margin, years_col.max())

        mask = (
            (years_col >= low) &
            (years_col <= high) &
            (pops_col >= 30)
        )
        result = result[mask].copy()

    if "track_id" in result.columns:
        result = result.drop_duplicates("track_id")

    if "artist_name" in result.columns:
        result["artist_rank"] = result.groupby("artist_name").cumcount()
        result = result[result["artist_rank"] < 3].drop(columns=["artist_rank"])

    result_sorted = result.sort_values("score", ascending=False)

    if ("artist_name" in result_sorted.columns) and fav_artists:
        fav_set = {a.strip().lower() for a in fav_artists if a and a.strip() != ""}

        mask_fav = _is_fav_artist_series(result_sorted["artist_name"], fav_set)

        df_fav = result_sorted[mask_fav]
        df_other = result_sorted[~mask_fav]

        if explorer:
            min_other_ratio = 0.5
        else:
            min_other_ratio = 0.3

        target_other = int(np.ceil(min_other_ratio * n))

        n_other = min(target_other, len(df_other))
        pick_other = df_other.head(n_other)

        remaining_slots = n - len(pick_other)
        pick_fav = df_fav.head(remaining_slots)

        already_idx = set(pick_other.index) | set(pick_fav.index)
        leftover = result_sorted[~result_sorted.index.isin(already_idx)].head(
            n - len(pick_other) - len(pick_fav)
        )

        top_result = pd.concat([pick_other, pick_fav, leftover]).head(n)

    else:
        top_result = result_sorted.head(n)

    cols_show = [
        "track_id", "track_name", "artist_name", "genre", "year", "popularity",
        "macro_cluster", "subcluster", "subcluster_label",
        "score", "user_taste_score", "mood_cluster_score",
        "time_score", "pop_score", "weather_score", "day_score"
    ]
    cols_exist = [c for c in cols_show if c in top_result.columns]
    top_result = top_result[cols_exist]

    print(
        f"User input → mood='{mood}', activity='{activity}', part_of_day='{part_of_day}', "
        f"weather='{weather}', age={age}, explorer={explorer}, "
        f"fav_artists={fav_artists}, language_prefs={language_prefs}"
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
    