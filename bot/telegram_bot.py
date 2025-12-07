#!/usr/bin/env python
# coding: utf-8

"""
Telegram Bot - Music Mood Recommender
Enhanced version with advanced filters
"""

import os
import sys
import logging

import pandas as pd
from dotenv import load_dotenv
from typing import Optional
from difflib import get_close_matches
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    logger.warning(f".env file not found at: {ENV_PATH}")

# Import recommender
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from recommender_05 import recommend_playlist, df

# Valid values
VALID_MOODS = [
    'happy', 'joy', 'joyful', 'positiv', 'upbeat',
    'sad', 'melancholic', 'low', 'blue',
    'relaxed', 'calm', 'chill',
    'angry', 'aggressive',
    'kids', 'children', 'nursery',
    'christmas', 'xmas', 'holiday',
    'religious', 'gospel'
]

VALID_ACTIVITIES = [
    'party', 'dancing', 'dance',
    'study', 'focus', 'work', 'reading',
    'gym', 'workout', 'run', 'running',
    'commute', 'travel'
]

VALID_TIMES = ['morning', 'night', 'late night', 'evening']
VALID_WEATHER = ['sunny', 'clear', 'rainy', 'storm', 'stormy', 'snow', 'snowy']
VALID_LANGUAGES = ['en', 'de', 'es', 'it', 'pt', 'fr', 'other']

DISPLAY_MOODS = ['happy', 'sad', 'relaxed', 'angry', 'kids', 'christmas', 'religious']
DISPLAY_ACTIVITIES = ['party', 'study', 'gym', 'commute']
DISPLAY_TIMES = ['morning', 'evening', 'night']
DISPLAY_WEATHER = ['sunny', 'rainy', 'snow']

# Conversation states
(
    MOOD,
    ACTIVITY,
    PART_OF_DAY,
    WEATHER,
    AGE,
    EXPLORER,
    FAV_ARTISTS,
    FAV_ARTISTS_CONFIRM,
    ADVANCED_PREFS,
    LANGUAGES,
    RECENCY_PREF,
    DURATION_PREF,
    DANCEABILITY_PREF,
    N_SONGS,
) = range(14)


def create_inline_keyboard(options: list[str], columns: int = 3) -> InlineKeyboardMarkup:
    """Create inline keyboard with buttons arranged in columns."""
    keyboard = []
    for i in range(0, len(options), columns):
        row = [
            InlineKeyboardButton(opt.capitalize(), callback_data=opt)
            for opt in options[i:i + columns]
        ]
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: show mood selection."""
    context.user_data.clear()
    
    keyboard = create_inline_keyboard(DISPLAY_MOODS, columns=3)
    
    await update.message.reply_text(
        "Welcome to Music Mood Recommender.\n\n"
        "I will help you create a personalized playlist based on your mood and context.\n\n"
        "What is your current mood?\n"
        "Select from the options or type your mood.",
        reply_markup=keyboard,
    )
    return MOOD


async def handle_mood_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    mood = query.data
    context.user_data["mood"] = mood
    await query.edit_message_text(f"Mood: {mood}")
    
    keyboard = create_inline_keyboard(DISPLAY_ACTIVITIES, columns=3)
    await query.message.reply_text(
        "What are you doing right now?\nSelect an activity or type your own.",
        reply_markup=keyboard,
    )
    return ACTIVITY


async def handle_mood_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    mood = (update.message.text or "").strip().lower()
    
    if mood not in VALID_MOODS:
        suggestions = [m for m in VALID_MOODS if mood in m][:5]
        suggestion_text = f"\nDid you mean: {', '.join(suggestions)}" if suggestions else ""
        
        await update.message.reply_text(
            f"'{mood}' is not recognized.\n"
            f"Valid moods: {', '.join(VALID_MOODS[:10])}, ..."
            f"{suggestion_text}"
        )
        return MOOD
    
    context.user_data["mood"] = mood
    keyboard = create_inline_keyboard(DISPLAY_ACTIVITIES, columns=3)
    await update.message.reply_text(
        "What are you doing right now?\nSelect an activity or type your own.",
        reply_markup=keyboard,
    )
    return ACTIVITY


async def handle_activity_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    activity = query.data
    context.user_data["activity"] = activity
    await query.edit_message_text(f"Activity: {activity}")
    
    keyboard = create_inline_keyboard(DISPLAY_TIMES, columns=3)
    await query.message.reply_text(
        "What part of the day is it?",
        reply_markup=keyboard,
    )
    return PART_OF_DAY


async def handle_activity_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    activity = (update.message.text or "").strip().lower()
    
    if activity not in VALID_ACTIVITIES:
        suggestions = [a for a in VALID_ACTIVITIES if activity in a][:5]
        suggestion_text = f"\nDid you mean: {', '.join(suggestions)}" if suggestions else ""
        
        await update.message.reply_text(
            f"'{activity}' is not recognized.\n"
            f"Valid activities: {', '.join(VALID_ACTIVITIES)}"
            f"{suggestion_text}"
        )
        return ACTIVITY
    
    context.user_data["activity"] = activity
    keyboard = create_inline_keyboard(DISPLAY_TIMES, columns=3)
    await update.message.reply_text(
        "What part of the day is it?",
        reply_markup=keyboard,
    )
    return PART_OF_DAY


async def handle_time_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    part_of_day = query.data
    context.user_data["part_of_day"] = part_of_day
    await query.edit_message_text(f"Time: {part_of_day}")
    
    keyboard = create_inline_keyboard(DISPLAY_WEATHER, columns=3)
    await query.message.reply_text(
        "How is the weather outside?",
        reply_markup=keyboard,
    )
    return WEATHER


async def handle_time_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    part_of_day = (update.message.text or "").strip().lower()
    
    if part_of_day not in VALID_TIMES:
        await update.message.reply_text(
            f"'{part_of_day}' is not recognized.\n"
            f"Valid times: {', '.join(VALID_TIMES)}"
        )
        return PART_OF_DAY
    
    context.user_data["part_of_day"] = part_of_day
    keyboard = create_inline_keyboard(DISPLAY_WEATHER, columns=3)
    await update.message.reply_text(
        "How is the weather outside?",
        reply_markup=keyboard,
    )
    return WEATHER


async def handle_weather_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    weather = query.data
    context.user_data["weather"] = weather
    await query.edit_message_text(f"Weather: {weather}")
    
    await query.message.reply_text(
        "How old are you?\nEnter your age as a number."
    )
    return AGE


async def handle_weather_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    weather = (update.message.text or "").strip().lower()
    
    if weather not in VALID_WEATHER:
        await update.message.reply_text(
            f"'{weather}' is not recognized.\n"
            f"Valid weather: {', '.join(VALID_WEATHER)}"
        )
        return WEATHER
    
    context.user_data["weather"] = weather
    await update.message.reply_text(
        "How old are you?\nEnter your age as a number."
    )
    return AGE


async def ask_explorer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    age_text = (update.message.text or "").strip()
    
    try:
        age = int(age_text)
        if age < 5 or age > 120:
            await update.message.reply_text(
                f"{age} does not seem valid.\n"
                "Enter your age between 5 and 120."
            )
            return AGE
    except ValueError:
        await update.message.reply_text(
            f"'{age_text}' is not a valid number.\n"
            "Enter your age as a number."
        )
        return AGE

    context.user_data["age"] = age
    
    keyboard = [
        [
            InlineKeyboardButton("Yes, explore", callback_data="explorer_yes"),
            InlineKeyboardButton("No, safe mode", callback_data="explorer_no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Do you want to explore new artists?\n"
        "Explorer mode gives more variety. Safe mode sticks to popular tracks.",
        reply_markup=reply_markup,
    )
    return EXPLORER


async def handle_explorer_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    explorer = query.data == "explorer_yes"
    context.user_data["explorer"] = explorer
    
    mode_text = "Explorer mode" if explorer else "Safe mode"
    await query.edit_message_text(mode_text)
    
    await query.message.reply_text(
        "List your favorite artists (optional).\n"
        "Separate names with commas, or type 'skip'.\n\n"
        "Example: Drake, The Weeknd, Dua Lipa"
    )
    return FAV_ARTISTS


async def handle_explorer_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    explorer_text = (update.message.text or "").strip().lower()
    
    yes_values = ['yes', 'y', 'true', '1', 'explore', 'explorer']
    no_values = ['no', 'n', 'false', '0', 'safe']
    
    if explorer_text in yes_values:
        explorer = True
        mode_text = "Explorer mode"
    elif explorer_text in no_values:
        explorer = False
        mode_text = "Safe mode"
    else:
        await update.message.reply_text(
            f"'{explorer_text}' is not valid.\nAnswer 'yes' or 'no'."
        )
        return EXPLORER
    
    context.user_data["explorer"] = explorer
    await update.message.reply_text(
        f"{mode_text}\n\n"
        "List your favorite artists (optional).\n"
        "Separate names with commas, or type 'skip'.\n\n"
        "Example: Drake, The Weeknd, Dua Lipa"
    )
    return FAV_ARTISTS


async def ask_advanced_prefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    fav_text = (update.message.text or "").strip()
    
    skip_keywords = ["none", "no", "n", "skip", "", "any"]
    
    if fav_text.lower() in skip_keywords:
        context.user_data["fav_artists"] = []
        
        keyboard = [
            [
                InlineKeyboardButton("Yes, set preferences", callback_data="advanced_yes"),
                InlineKeyboardButton("No, skip", callback_data="advanced_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "No artist filter.\n\n"
            "Would you like to set advanced preferences?\n"
            "(Language, year range, duration, danceability)",
            reply_markup=reply_markup,
        )
        return ADVANCED_PREFS
    
    entered_artists = [a.strip() for a in fav_text.split(",") if a.strip()]
    all_artists = df["artist_name"].unique().tolist()
    all_artists_lower = [a.lower() for a in all_artists]
    
    matched_artists = []
    not_found = []
    
    for entered in entered_artists:
        if entered.lower() in all_artists_lower:
            idx = all_artists_lower.index(entered.lower())
            matched_artists.append(all_artists[idx])
            continue
        
        matches = get_close_matches(entered.lower(), all_artists_lower, n=1, cutoff=0.6)
        if matches:
            idx = all_artists_lower.index(matches[0])
            matched_artists.append(all_artists[idx])
        else:
            not_found.append(entered)
    
    if not matched_artists:
        not_found_str = ", ".join(not_found)
        await update.message.reply_text(
            f"No artists found: {not_found_str}\n\n"
            "Try different names or type 'skip'."
        )
        return FAV_ARTISTS
    
    context.user_data["matched_artists_temp"] = matched_artists
    context.user_data["not_found_temp"] = not_found
    
    if not not_found:
        keyboard = [
            [
                InlineKeyboardButton("Yes, correct", callback_data="confirm_yes"),
                InlineKeyboardButton("No, try again", callback_data="confirm_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        matched_str = ", ".join(matched_artists)
        
        await update.message.reply_text(
            f"Found: {matched_str}\n\nIs this correct?",
            reply_markup=reply_markup,
        )
        return FAV_ARTISTS_CONFIRM
    
    keyboard = [
        [InlineKeyboardButton("Continue with found", callback_data="continue_partial")],
        [InlineKeyboardButton("Try again", callback_data="retry_artists")],
        [InlineKeyboardButton("Skip", callback_data="skip_artists")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    not_found_str = ", ".join(not_found)
    matched_str = ", ".join(matched_artists)
    
    await update.message.reply_text(
        f"Not found: {not_found_str}\nFound: {matched_str}\n\n"
        "What would you like to do?",
        reply_markup=reply_markup
    )
    return FAV_ARTISTS


async def handle_artist_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    if query.data == "confirm_yes":
        matched = context.user_data.get("matched_artists_temp", [])
        context.user_data["fav_artists"] = matched
        context.user_data.pop("matched_artists_temp", None)
        context.user_data.pop("not_found_temp", None)
        
        matched_str = ", ".join(matched)
        await query.edit_message_text(f"Using: {matched_str}")
        
        keyboard = [
            [
                InlineKeyboardButton("Yes, set preferences", callback_data="advanced_yes"),
                InlineKeyboardButton("No, skip", callback_data="advanced_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "Would you like to set advanced preferences?\n"
            "(Language, year range, duration, danceability)",
            reply_markup=reply_markup,
        )
        return ADVANCED_PREFS
    else:
        context.user_data.pop("matched_artists_temp", None)
        context.user_data.pop("not_found_temp", None)
        
        await query.edit_message_text(
            "Enter artist names again, separated by commas.\nOr type 'skip'."
        )
        return FAV_ARTISTS


async def handle_artist_confirm_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip().lower()
    
    yes_values = ['yes', 'y', 'ok', 'correct', '1']
    no_values = ['no', 'n', 'nope', '0', 'wrong', 'retry']
    
    if text in yes_values:
        matched = context.user_data.get("matched_artists_temp", [])
        context.user_data["fav_artists"] = matched
        context.user_data.pop("matched_artists_temp", None)
        context.user_data.pop("not_found_temp", None)
        
        matched_str = ", ".join(matched)
        keyboard = [
            [
                InlineKeyboardButton("Yes, set preferences", callback_data="advanced_yes"),
                InlineKeyboardButton("No, skip", callback_data="advanced_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Using: {matched_str}\n\n"
            "Would you like to set advanced preferences?\n"
            "(Language, year range, duration, danceability)",
            reply_markup=reply_markup,
        )
        return ADVANCED_PREFS
    elif text in no_values:
        context.user_data.pop("matched_artists_temp", None)
        context.user_data.pop("not_found_temp", None)
        
        await update.message.reply_text(
            "Enter artist names again, separated by commas.\nOr type 'skip'."
        )
        return FAV_ARTISTS
    else:
        await update.message.reply_text("Answer 'yes' or 'no'.")
        return FAV_ARTISTS_CONFIRM


async def handle_continue_partial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    matched = context.user_data.get("matched_artists_temp", [])
    context.user_data["fav_artists"] = matched
    context.user_data.pop("matched_artists_temp", None)
    context.user_data.pop("not_found_temp", None)
    
    matched_str = ", ".join(matched)
    await query.edit_message_text(f"Using: {matched_str}")
    
    keyboard = [
        [
            InlineKeyboardButton("Yes, set preferences", callback_data="advanced_yes"),
            InlineKeyboardButton("No, skip", callback_data="advanced_no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "Would you like to set advanced preferences?\n"
        "(Language, year range, duration, danceability)",
        reply_markup=reply_markup,
    )
    return ADVANCED_PREFS


async def handle_retry_artists(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    context.user_data.pop("matched_artists_temp", None)
    context.user_data.pop("not_found_temp", None)
    
    await query.edit_message_text(
        "Enter artist names again, separated by commas.\nOr type 'skip'."
    )
    return FAV_ARTISTS


async def handle_skip_artists(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    context.user_data["fav_artists"] = []
    context.user_data.pop("matched_artists_temp", None)
    context.user_data.pop("not_found_temp", None)
    
    await query.edit_message_text("No artist filter")
    
    keyboard = [
        [
            InlineKeyboardButton("Yes, set preferences", callback_data="advanced_yes"),
            InlineKeyboardButton("No, skip", callback_data="advanced_no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "Would you like to set advanced preferences?\n"
        "(Language, year range, duration, danceability)",
        reply_markup=reply_markup,
    )
    return ADVANCED_PREFS


async def handle_advanced_prefs_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    if query.data == "advanced_yes":
        await query.edit_message_text("Advanced preferences enabled")
        
        common_languages = ['en', 'it', 'es', 'fr', 'de', 'pt', 'any']
        keyboard = create_inline_keyboard(common_languages, columns=4)
        
        await query.message.reply_text(
            "Preferred language for songs?\n"
            "Select a language or type 'any' for no preference.\n"
            "You can type multiple languages separated by commas (e.g., en,it)",
            reply_markup=keyboard,
        )
        return LANGUAGES
    else:
        await query.edit_message_text("Advanced preferences skipped")
        
        context.user_data["language_prefs"] = []
        context.user_data["recency_pref"] = None
        context.user_data["duration_pref"] = None
        context.user_data["danceability_pref"] = None
        
        numbers = ['5', '10', '15', '20', '25', '30']
        keyboard = create_inline_keyboard(numbers, columns=3)
        
        await query.message.reply_text(
            "How many songs do you want?\n"
            "Select a number or type your preferred length.",
            reply_markup=keyboard,
        )
        return N_SONGS


async def handle_advanced_prefs_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip().lower()
    
    yes_values = ['yes', 'y', 'ok', '1']
    no_values = ['no', 'n', '0', 'skip']
    
    if text in yes_values:
        common_languages = ['en', 'it', 'es', 'fr', 'de', 'pt', 'any']
        keyboard = create_inline_keyboard(common_languages, columns=4)
        
        await update.message.reply_text(
            "Advanced preferences enabled\n\n"
            "Preferred language for songs?\n"
            "Select a language or type 'any' for no preference.\n"
            "You can type multiple languages separated by commas (e.g., en,it)",
            reply_markup=keyboard,
        )
        return LANGUAGES
    elif text in no_values:
        context.user_data["language_prefs"] = []
        context.user_data["recency_pref"] = None
        context.user_data["duration_pref"] = None
        context.user_data["danceability_pref"] = None
        
        numbers = ['5', '10', '15', '20', '25', '30']
        keyboard = create_inline_keyboard(numbers, columns=3)
        
        await update.message.reply_text(
            "Advanced preferences skipped\n\n"
            "How many songs do you want?\n"
            "Select a number or type your preferred length.",
            reply_markup=keyboard,
        )
        return N_SONGS
    else:
        await update.message.reply_text(f"'{text}' is not valid.\nAnswer 'yes' or 'no'.")
        return ADVANCED_PREFS


async def handle_language_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    lang = query.data
    
    if lang == "any":
        language_prefs = []
        await query.edit_message_text("No language preference")
    else:
        language_prefs = [lang]
        await query.edit_message_text(f"Language: {lang.upper()}")
    
    context.user_data["language_prefs"] = language_prefs
    
    keyboard = [
        [InlineKeyboardButton("Recent (2015+)", callback_data="recency_recent")],
        [InlineKeyboardButton("Modern (2000-2014)", callback_data="recency_modern")],
        [InlineKeyboardButton("Classic (1980-1999)", callback_data="recency_classic")],
        [InlineKeyboardButton("Old (pre-1980)", callback_data="recency_old")],
        [InlineKeyboardButton("No preference", callback_data="recency_any")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "Do you prefer older or more recent songs?",
        reply_markup=reply_markup,
    )
    return RECENCY_PREF


async def handle_language_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang_text = (update.message.text or "").strip().lower()
    
    if lang_text in ["any", "none", "no", "all", "skip", ""]:
        language_prefs = []
    else:
        entered_langs = [l.strip().lower() for l in lang_text.split(",") if l.strip()]
        invalid_langs = [lang for lang in entered_langs if lang not in VALID_LANGUAGES]
        
        if invalid_langs:
            await update.message.reply_text(
                f"Invalid language codes: {', '.join(invalid_langs)}\n"
                f"Valid codes: {', '.join(VALID_LANGUAGES)}\n"
                "Enter valid codes separated by commas (e.g., en,it) or type 'any'."
            )
            return LANGUAGES
        
        language_prefs = entered_langs
    
    context.user_data["language_prefs"] = language_prefs
    
    keyboard = [
        [InlineKeyboardButton("Recent (2015+)", callback_data="recency_recent")],
        [InlineKeyboardButton("Modern (2000-2014)", callback_data="recency_modern")],
        [InlineKeyboardButton("Classic (1980-1999)", callback_data="recency_classic")],
        [InlineKeyboardButton("Old (pre-1980)", callback_data="recency_old")],
        [InlineKeyboardButton("No preference", callback_data="recency_any")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Do you prefer older or more recent songs?",
        reply_markup=reply_markup,
    )
    return RECENCY_PREF


async def handle_recency_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    recency = query.data.replace("recency_", "")
    
    recency_map = {
        "recent": (2015, 2025),
        "modern": (2000, 2014),
        "classic": (1980, 1999),
        "old": (0, 1979),
        "any": None
    }
    
    context.user_data["recency_pref"] = recency_map.get(recency, None)
    
    recency_labels = {
        "recent": "Recent (2015+)",
        "modern": "Modern (2000-2014)",
        "classic": "Classic (1980-1999)",
        "old": "Old (pre-1980)",
        "any": "No preference"
    }
    
    await query.edit_message_text(f"Year range: {recency_labels.get(recency, 'Any')}")
    
    keyboard = [
        [InlineKeyboardButton("Short (<3 min)", callback_data="duration_short")],
        [InlineKeyboardButton("Medium (3-5 min)", callback_data="duration_medium")],
        [InlineKeyboardButton("Long (>5 min)", callback_data="duration_long")],
        [InlineKeyboardButton("No preference", callback_data="duration_any")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "What song length do you prefer?",
        reply_markup=reply_markup,
    )
    return DURATION_PREF


async def handle_recency_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip().lower()
    
    if text in ["any", "none", "no", "all", "skip", ""]:
        context.user_data["recency_pref"] = None
    else:
        await update.message.reply_text("Use the buttons above or type 'any' to skip.")
        return RECENCY_PREF
    
    keyboard = [
        [InlineKeyboardButton("Short (<3 min)", callback_data="duration_short")],
        [InlineKeyboardButton("Medium (3-5 min)", callback_data="duration_medium")],
        [InlineKeyboardButton("Long (>5 min)", callback_data="duration_long")],
        [InlineKeyboardButton("No preference", callback_data="duration_any")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "What song length do you prefer?",
        reply_markup=reply_markup,
    )
    return DURATION_PREF


async def handle_duration_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    duration = query.data.replace("duration_", "")
    
    if duration == "any":
        context.user_data["duration_pref"] = None
        await query.edit_message_text("Duration: No preference")
    else:
        context.user_data["duration_pref"] = duration
        duration_labels = {
            "short": "Short (<3 min)",
            "medium": "Medium (3-5 min)",
            "long": "Long (>5 min)"
        }
        await query.edit_message_text(f"Duration: {duration_labels.get(duration, 'Any')}")
    
    keyboard = [
        [InlineKeyboardButton("High danceability", callback_data="dance_high")],
        [InlineKeyboardButton("Low danceability", callback_data="dance_low")],
        [InlineKeyboardButton("No preference", callback_data="dance_any")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "Do you want danceable tracks?",
        reply_markup=reply_markup,
    )
    return DANCEABILITY_PREF


async def handle_duration_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip().lower()
    
    if text in ["any", "none", "no", "all", "skip", ""]:
        context.user_data["duration_pref"] = None
        
        keyboard = [
            [InlineKeyboardButton("High danceability", callback_data="dance_high")],
            [InlineKeyboardButton("Low danceability", callback_data="dance_low")],
            [InlineKeyboardButton("No preference", callback_data="dance_any")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Do you want danceable tracks?",
            reply_markup=reply_markup,
        )
        return DANCEABILITY_PREF
    else:
        await update.message.reply_text("Use the buttons above or type 'any' to skip.")
        return DURATION_PREF


async def handle_danceability_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    dance = query.data.replace("dance_", "")
    
    if dance == "any":
        context.user_data["danceability_pref"] = None
        await query.edit_message_text("Danceability: No preference")
    else:
        context.user_data["danceability_pref"] = dance
        dance_labels = {"high": "High danceability", "low": "Low danceability"}
        await query.edit_message_text(f"Danceability: {dance_labels.get(dance, 'Any')}")
    
    numbers = ['5', '10', '15', '20', '25', '30']
    keyboard = create_inline_keyboard(numbers, columns=3)
    
    await query.message.reply_text(
        "How many songs do you want?\n"
        "Select a number or type your preferred length.",
        reply_markup=keyboard,
    )
    return N_SONGS


async def handle_danceability_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip().lower()
    
    if text in ["any", "none", "no", "all", "skip", ""]:
        context.user_data["danceability_pref"] = None
        
        numbers = ['5', '10', '15', '20', '25', '30']
        keyboard = create_inline_keyboard(numbers, columns=3)
        
        await update.message.reply_text(
            "How many songs do you want?\n"
            "Select a number or type your preferred length.",
            reply_markup=keyboard,
        )
        return N_SONGS
    else:
        await update.message.reply_text("Use the buttons above or type 'any' to skip.")
        return DANCEABILITY_PREF


async def handle_n_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    n = int(query.data)
    context.user_data["n"] = n
    
    await query.edit_message_text(f"Playlist length: {n} songs")
    await query.message.reply_text("Generating your playlist. This may take a moment.")
    
    return await generate_playlist_final(query.message, context)


async def generate_playlist_from_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    n_text = (update.message.text or "").strip()
    
    try:
        n = int(n_text)
        if n <= 0:
            raise ValueError("Negative number")
        if n > 100:
            await update.message.reply_text(
                "Maximum is 100 songs.\nEnter a number between 1 and 100."
            )
            return N_SONGS
    except ValueError:
        await update.message.reply_text(
            f"'{n_text}' is not a valid number.\nEnter a number (e.g., 10, 20, 30)."
        )
        return N_SONGS
    
    context.user_data["n"] = n
    await update.message.reply_text("Generating your playlist. This may take a moment.")
    
    return await generate_playlist_final(update.message, context)


async def generate_playlist_final(message, context: ContextTypes.DEFAULT_TYPE) -> int:
    mood = context.user_data.get("mood", "happy")
    activity = context.user_data.get("activity", "party")
    part_of_day = context.user_data.get("part_of_day", "evening")
    weather = context.user_data.get("weather", "clear")
    age = context.user_data.get("age", 25)
    explorer = context.user_data.get("explorer", False)
    fav_artists = context.user_data.get("fav_artists", [])
    language_prefs = context.user_data.get("language_prefs", [])
    recency_pref = context.user_data.get("recency_pref", None)
    duration_pref = context.user_data.get("duration_pref", None)
    danceability_pref = context.user_data.get("danceability_pref", None)
    n = context.user_data.get("n", 20)
    
    try:
        df_rec = recommend_playlist(
            mood=mood,
            activity=activity,
            part_of_day=part_of_day,
            weather=weather,
            age=age,
            explorer=explorer,
            n=n,
            fav_artists=fav_artists if fav_artists else None,
            language_prefs=language_prefs if language_prefs else None,
            recency_pref=recency_pref,
            duration_pref=duration_pref,
            danceability_pref=danceability_pref,
        )
        
        if df_rec is None or len(df_rec) == 0:
            await message.reply_text(
                "Could not find suitable tracks for this configuration.\n"
                "Filters might be too restrictive. Try adjusting preferences or enable explorer mode.\n\n"
                "Type /start to try again."
            )
            return ConversationHandler.END
        
        actual_count = len(df_rec)
        
        if actual_count < n:
            await message.reply_text(
                f"Note: Only {actual_count} tracks match your filters. "
                f"You requested {n}, but the current selection criteria limit available results. "
                f"Consider relaxing some filters for more variety."
            )
        
        lines = ["Here is your personalized playlist:\n"]
        
        for i, row in enumerate(df_rec.itertuples(index=False), start=1):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)
            
            track_name = row_dict.get("track_name", "Unknown track")
            artist_name = row_dict.get("artist_name", "Unknown artist")
            year = row_dict.get("year", "")
            popularity = row_dict.get("popularity", "")
            track_id = row_dict.get("track_id", "")
            
            spotify_url = None
            if isinstance(track_id, str) and track_id.strip():
                spotify_url = f"https://open.spotify.com/track/{track_id.strip()}"
            
            line = f"{i}. {track_name} - {artist_name}"
            
            details = []
            if year and not pd.isna(year):
                try:
                    details.append(str(int(year)))
                except:
                    details.append(str(year))
            if popularity and not pd.isna(popularity):
                try:
                    details.append(f"pop {int(popularity)}")
                except:
                    pass
            
            if details:
                line += f" ({', '.join(details)})"
            
            lines.append(line)
            
            if spotify_url:
                lines.append(f"   {spotify_url}")
        
        text = "\n".join(lines)
        MAX_LEN = 3500

        if len(text) <= MAX_LEN:
            await message.reply_text(text)
        else:
            chunks = []
            current = ""
            
            for line in text.split("\n"):
                if len(current) + len(line) + 1 > MAX_LEN:
                    chunks.append(current)
                    current = line + "\n"
                else:
                    current += line + "\n"
            
            if current:
                chunks.append(current)
            
            for chunk in chunks:
                await message.reply_text(chunk)
        
        await message.reply_text("\nEnjoy your music.\nType /start to create another playlist.")
        
    except ValueError as e:
        await message.reply_text(
            f"Input validation error:\n{str(e)}\n\nType /start to try again."
        )
    except Exception as e:
        logger.exception("Error generating playlist")
        await message.reply_text(
            "An unexpected error occurred.\nType /start to try again."
        )
    
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Conversation cancelled.\nType /start to begin again.")
    return ConversationHandler.END


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not found. Set it in your .env file.")
    
    application = ApplicationBuilder().token(token).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MOOD: [
                CallbackQueryHandler(handle_mood_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mood_text)
            ],
            ACTIVITY: [
                CallbackQueryHandler(handle_activity_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_activity_text)
            ],
            PART_OF_DAY: [
                CallbackQueryHandler(handle_time_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_time_text)
            ],
            WEATHER: [
                CallbackQueryHandler(handle_weather_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_weather_text)
            ],
            AGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_explorer)
            ],
            EXPLORER: [
                CallbackQueryHandler(handle_explorer_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_explorer_text)
            ],
            FAV_ARTISTS: [
                CallbackQueryHandler(handle_continue_partial, pattern="^continue_partial$"),
                CallbackQueryHandler(handle_retry_artists, pattern="^retry_artists$"),
                CallbackQueryHandler(handle_skip_artists, pattern="^skip_artists$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_advanced_prefs)
            ],
            FAV_ARTISTS_CONFIRM: [
                CallbackQueryHandler(handle_artist_confirm, pattern="^confirm_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_artist_confirm_text)
            ],
            ADVANCED_PREFS: [
                CallbackQueryHandler(handle_advanced_prefs_button, pattern="^advanced_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_advanced_prefs_text)
            ],
            LANGUAGES: [
                CallbackQueryHandler(handle_language_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_language_text)
            ],
            RECENCY_PREF: [
                CallbackQueryHandler(handle_recency_button, pattern="^recency_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_recency_text)
            ],
            DURATION_PREF: [
                CallbackQueryHandler(handle_duration_button, pattern="^duration_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_duration_text)
            ],
            DANCEABILITY_PREF: [
                CallbackQueryHandler(handle_danceability_button, pattern="^dance_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_danceability_text)
            ],
            N_SONGS: [
                CallbackQueryHandler(handle_n_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, generate_playlist_from_text)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    application.add_handler(conv_handler)
    
    logger.info("Bot started")
    logger.info("Waiting for messages")
    application.run_polling()


if __name__ == "__main__":
    main()