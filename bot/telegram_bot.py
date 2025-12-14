# telegram_bot.py
# Telegram Bot - Music Mood Recommender with /back support (disabled after playlist generation)

import os
import sys
import logging
import urllib.parse
import secrets
from difflib import get_close_matches

import pandas as pd
from dotenv import load_dotenv
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

# ======================================
# LOGGING
# ======================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ======================================
# ENV & IMPORTS
# ======================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from recommender_05 import recommend_playlist, df  # noqa: E402

# ======================================
# STATES
# ======================================
(
    MOOD,
    ACTIVITY,
    PART_OF_DAY,
    WEATHER,
    AGE,
    EXPLORER,
    FAV_ARTISTS,
    ASK_LANG_PREFS,
    LANGUAGES,
    N_SONGS,
    SPOTY,
) = range(11)

# ======================================
# DISPLAY OPTIONS
# ======================================
DISPLAY_MOODS = ["happy", "sad", "relaxed", "angry", "kids", "christmas", "religious"]
DISPLAY_ACTIVITIES = ["party", "study", "gym", "commute", "chilling"]
DISPLAY_TIMES = ["morning", "afternoon", "evening", "night"]
DISPLAY_WEATHER = ["sunny", "rainy", "snow", "cloudy"]

VALID_LANGUAGES = ["en", "it", "es", "fr", "de", "pt", "other"]

# ======================================
# KEYBOARDS
# ======================================
def kb(options, cols=3) -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(options), cols):
        rows.append([InlineKeyboardButton(o.capitalize(), callback_data=o) for o in options[i:i + cols]])
    return InlineKeyboardMarkup(rows)


def yes_no_kb(yes_cb: str, no_cb: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Yes", callback_data=yes_cb),
        InlineKeyboardButton("No", callback_data=no_cb),
    ]])


def language_multiselect_keyboard(selected: set[str]) -> InlineKeyboardMarkup:
    rows = []
    for code in VALID_LANGUAGES:
        label = f"✅ {code.upper()}" if code in selected else code.upper()
        rows.append([InlineKeyboardButton(label, callback_data=f"lang_toggle:{code}")])

    rows.append([
        InlineKeyboardButton("Any", callback_data="lang_any"),
        InlineKeyboardButton("Done", callback_data="lang_done"),
    ])
    return InlineKeyboardMarkup(rows)


async def _reply_use_buttons(update: Update, text: str, reply_markup=None) -> None:
    if update.message:
        await update.message.reply_text(
            f"{text}\n\nPlease use the buttons below 👇",
            reply_markup=reply_markup,
        )

# ======================================
# /BACK SUPPORT (history stack)
# - Works only BEFORE playlist generation
# - Disabled after generation (SPOTY)
# ======================================
def _ensure_history(context: ContextTypes.DEFAULT_TYPE) -> None:
    if "history" not in context.user_data or not isinstance(context.user_data["history"], list):
        context.user_data["history"] = []


def _push_state(context: ContextTypes.DEFAULT_TYPE, state: int) -> None:
    _ensure_history(context)
    context.user_data["history"].append(state)


async def back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Go back one step.
    Works until the playlist is generated. After generation, /back is disabled.
    """
    # If playlist already generated (we set this flag), /back disabled
    if context.user_data.get("playlist_generated", False):
        await update.message.reply_text(
            "Back is disabled after the playlist is generated.\n"
            "Type /start to create a new playlist."
        )
        # Do NOT go back, do NOT end the conversation: just ignore /back.
        # If we're already in SPOTY, keep SPOTY; otherwise, keep the current step if available.
        current = context.user_data.get("current_state")
        return SPOTY if current is None else current

    history = context.user_data.get("history", [])
    if not history:
        await update.message.reply_text(
            "You are already at the beginning.\nType /start to restart."
        )
        return MOOD

    prev = history.pop()
    context.user_data["history"] = history

    prompts = {
        MOOD: ("What is your current mood?", kb(DISPLAY_MOODS)),
        ACTIVITY: ("What are you doing right now?", kb(DISPLAY_ACTIVITIES)),
        PART_OF_DAY: ("What part of the day is it?", kb(DISPLAY_TIMES)),
        WEATHER: ("How is the weather outside?", kb(DISPLAY_WEATHER)),
        AGE: ("How old are you?\nEnter your age as a number.", None),
        EXPLORER: (
            "Do you want to explore new artists?\n"
            "Explorer mode gives more variety. Safe mode sticks to popular tracks.",
            InlineKeyboardMarkup([[
                InlineKeyboardButton("Yes, explore", callback_data="explorer_yes"),
                InlineKeyboardButton("No, safe mode", callback_data="explorer_no"),
            ]])
        ),
        FAV_ARTISTS: (
            "List your favorite artists (optional).\n"
            "Separate names with commas, or type 'skip'.\n\n"
            "Example: Drake, The Weeknd, Dua Lipa",
            None
        ),
        ASK_LANG_PREFS: (
            "Do you want to set language preferences for the playlist?\n"
            "(You can select multiple languages.)",
            yes_no_kb("langprefs_yes", "langprefs_no")
        ),
        LANGUAGES: (
            "Select one or more languages.\n"
            "Tap to toggle ✅, then press Done.\n\n"
            "Tip: press Any to clear selection (no preference).",
            language_multiselect_keyboard(context.user_data.get("language_selected", set()))
        ),
        N_SONGS: ("How many songs do you want?", kb(["5", "10", "15", "20", "25", "30"])),
    }

    text, markup = prompts.get(prev, ("Going back…", None))
    await update.message.reply_text(text, reply_markup=markup)
    return prev

# ======================================
# START
# ======================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    _ensure_history(context)
    context.user_data["playlist_generated"] = False

    await update.message.reply_text(
        "Hi! I'm Calliope 🎧\n\n"
        "I will build a personalized playlist for you.\n\n"
        "💡 You can type /back at any time to go back one step (until the playlist is generated).\n\n"
        "What is your current mood?",
        reply_markup=kb(DISPLAY_MOODS),
    )
    return MOOD

# ======================================
# MOOD
# ======================================
async def handle_mood_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, MOOD)

    query = update.callback_query
    await query.answer()

    mood = query.data
    context.user_data["mood"] = mood
    await query.edit_message_text(f"Mood: {mood}")

    await query.message.reply_text(
        "What are you doing right now?",
        reply_markup=kb(DISPLAY_ACTIVITIES),
    )
    return ACTIVITY


async def handle_mood_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(update, "I’m expecting a mood selection.", kb(DISPLAY_MOODS))
    return MOOD

# ======================================
# ACTIVITY
# ======================================
async def handle_activity_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, ACTIVITY)

    query = update.callback_query
    await query.answer()

    activity = query.data
    context.user_data["activity"] = activity
    await query.edit_message_text(f"Activity: {activity}")

    await query.message.reply_text(
        "What part of the day is it?",
        reply_markup=kb(DISPLAY_TIMES),
    )
    return PART_OF_DAY


async def handle_activity_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(update, "I’m expecting an activity selection.", kb(DISPLAY_ACTIVITIES))
    return ACTIVITY

# ======================================
# PART OF DAY
# ======================================
async def handle_time_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, PART_OF_DAY)

    query = update.callback_query
    await query.answer()

    part_of_day = query.data
    context.user_data["part_of_day"] = part_of_day
    await query.edit_message_text(f"Time: {part_of_day}")

    await query.message.reply_text(
        "How is the weather outside?",
        reply_markup=kb(DISPLAY_WEATHER),
    )
    return WEATHER


async def handle_time_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(update, "I’m expecting a time-of-day selection.", kb(DISPLAY_TIMES))
    return PART_OF_DAY

# ======================================
# WEATHER
# ======================================
async def handle_weather_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, WEATHER)

    query = update.callback_query
    await query.answer()

    weather = query.data
    context.user_data["weather"] = weather
    await query.edit_message_text(f"Weather: {weather}")

    await query.message.reply_text("How old are you?\nEnter your age as a number.")
    return AGE


async def handle_weather_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(update, "I’m expecting a weather selection.", kb(DISPLAY_WEATHER))
    return WEATHER

# ======================================
# AGE -> EXPLORER
# ======================================
async def ask_explorer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    age_text = (update.message.text or "").strip()

    try:
        age = int(age_text)
    except ValueError:
        await update.message.reply_text(
            f"'{age_text}' is not a valid number.\n"
            "Please enter a valid numeric age (e.g., 21)."
        )
        return AGE

    if age < 5 or age > 120:
        await update.message.reply_text(
            f"{age} does not seem valid.\n"
            "Please enter a valid age between 5 and 120."
        )
        return AGE

    _push_state(context, AGE)
    context.user_data["age"] = age

    await update.message.reply_text(
        "Do you want to explore new artists?\n"
        "Explorer mode gives more variety. Safe mode sticks to popular tracks.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("Yes, explore", callback_data="explorer_yes"),
            InlineKeyboardButton("No, safe mode", callback_data="explorer_no"),
        ]]),
    )
    return EXPLORER

# ======================================
# EXPLORER
# ======================================
async def handle_explorer_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, EXPLORER)

    query = update.callback_query
    await query.answer()

    explorer = query.data == "explorer_yes"
    context.user_data["explorer"] = explorer
    await query.edit_message_text("Explorer mode" if explorer else "Safe mode")

    await query.message.reply_text(
        "List your favorite artists (optional).\n"
        "Separate names with commas, or type 'skip'.\n\n"
        "Example: Drake, The Weeknd, Dua Lipa"
    )
    return FAV_ARTISTS


async def handle_explorer_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(
        update,
        "I’m expecting a choice about Explorer mode.",
        InlineKeyboardMarkup([[
            InlineKeyboardButton("Yes, explore", callback_data="explorer_yes"),
            InlineKeyboardButton("No, safe mode", callback_data="explorer_no"),
        ]]),
    )
    return EXPLORER

# ======================================
# FAV ARTISTS -> ASK LANG PREFS
# ======================================
async def ask_language_prefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, FAV_ARTISTS)

    fav_text = (update.message.text or "").strip()
    skip_keywords = ["none", "no", "n", "skip", "", "any"]

    if fav_text.lower() in skip_keywords:
        context.user_data["fav_artists"] = []
    else:
        entered_artists = [a.strip() for a in fav_text.split(",") if a.strip()]

        all_artists = df["artist_name"].dropna().unique().tolist()
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

        if entered_artists and not matched_artists:
            await update.message.reply_text(
                "No artists were found for what you typed.\n\n"
                "Type 'skip' to continue without an artist filter, "
                "or enter names of artists that exist in the dataset."
            )
            return FAV_ARTISTS

        context.user_data["fav_artists"] = matched_artists

        if not_found:
            await update.message.reply_text(
                f"Note: these artists were not found and will be ignored: {', '.join(not_found)}"
            )

    await update.message.reply_text(
        "Do you want to set language preferences for the playlist?\n"
        "(You can select multiple languages.)",
        reply_markup=yes_no_kb("langprefs_yes", "langprefs_no"),
    )
    return ASK_LANG_PREFS

# ======================================
# ASK LANG PREFS (buttons expected)
# ======================================
async def handle_ask_langprefs_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, ASK_LANG_PREFS)

    query = update.callback_query
    await query.answer()

    if query.data == "langprefs_no":
        context.user_data["language_prefs"] = []
        await query.edit_message_text("Language preferences: none")

        await query.message.reply_text(
            "How many songs do you want?",
            reply_markup=kb(["5", "10", "15", "20", "25", "30"]),
        )
        return N_SONGS

    # YES -> multi-select
    context.user_data["language_selected"] = set()
    await query.edit_message_text(
        "Select one or more languages.\n"
        "Tap to toggle ✅, then press Done.\n\n"
        "Tip: press Any to clear selection (no preference)."
    )
    await query.message.reply_text(
        "Languages:",
        reply_markup=language_multiselect_keyboard(context.user_data["language_selected"]),
    )
    return LANGUAGES


async def handle_ask_langprefs_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(
        update,
        "I’m expecting a Yes/No answer about language preferences.",
        yes_no_kb("langprefs_yes", "langprefs_no"),
    )
    return ASK_LANG_PREFS

# ======================================
# LANGUAGES (buttons ONLY)
# ======================================
async def handle_language_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    data = query.data
    selected: set[str] = context.user_data.get("language_selected", set())

    if data == "lang_any":
        selected.clear()
        context.user_data["language_selected"] = selected
        await query.edit_message_reply_markup(reply_markup=language_multiselect_keyboard(selected))
        return LANGUAGES

    if data == "lang_done":
        _push_state(context, LANGUAGES)

        context.user_data["language_prefs"] = sorted(list(selected))
        prefs = context.user_data["language_prefs"]
        msg = "Language preferences: none" if not prefs else f"Language preferences: {', '.join([p.upper() for p in prefs])}"
        await query.message.reply_text(msg)

        await query.message.reply_text(
            "How many songs do you want?",
            reply_markup=kb(["5", "10", "15", "20", "25", "30"]),
        )
        return N_SONGS

    if data.startswith("lang_toggle:"):
        code = data.split(":", 1)[1].strip().lower()
        if code in VALID_LANGUAGES:
            if code in selected:
                selected.remove(code)
            else:
                selected.add(code)

        context.user_data["language_selected"] = selected
        await query.edit_message_reply_markup(reply_markup=language_multiselect_keyboard(selected))
        return LANGUAGES

    return LANGUAGES


async def handle_language_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    selected = context.user_data.get("language_selected", set())
    await _reply_use_buttons(
        update,
        "I’m expecting you to select languages using the buttons.",
        language_multiselect_keyboard(selected),
    )
    return LANGUAGES

# ======================================
# N_SONGS -> PLAYLIST (typing allowed)
# ======================================
async def handle_n_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, N_SONGS)

    query = update.callback_query
    await query.answer()

    n = int(query.data)
    context.user_data["n"] = n

    await query.edit_message_text(f"Playlist length: {n} songs")
    await query.message.reply_text("Generating your playlist. This may take a moment.")

    return await generate_playlist_final(query.message, context)


async def generate_playlist_from_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _push_state(context, N_SONGS)

    n_text = (update.message.text or "").strip()
    try:
        n = int(n_text)
    except ValueError:
        await update.message.reply_text(
            f"'{n_text}' is not a valid number.\nPlease enter a numeric value (e.g., 10, 20, 30)."
        )
        return N_SONGS

    if n <= 0:
        await update.message.reply_text("Please enter a number greater than 0.")
        return N_SONGS

    if n > 100:
        await update.message.reply_text("Maximum is 100 songs.\nEnter a number between 1 and 100.")
        return N_SONGS

    context.user_data["n"] = n
    await update.message.reply_text("Generating your playlist. This may take a moment.")
    return await generate_playlist_final(update.message, context)

# ======================================
# PLAYLIST GENERATION
# (after this, /back is disabled)
# ======================================
async def generate_playlist_final(message, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Disable /back after generation
    context.user_data["playlist_generated"] = True

    mood = context.user_data.get("mood", "happy")
    activity = context.user_data.get("activity", "party")
    part_of_day = context.user_data.get("part_of_day", "evening")
    weather = context.user_data.get("weather", "clear")
    age = context.user_data.get("age", 25)
    explorer = context.user_data.get("explorer", False)
    fav_artists = context.user_data.get("fav_artists", [])
    language_prefs = context.user_data.get("language_prefs", [])
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
        )

        if df_rec is None or len(df_rec) == 0:
            await message.reply_text(
                "Could not find suitable tracks for this configuration.\n"
                "Try adjusting your inputs or enable Explorer mode.\n\n"
                "Type /start to try again."
            )
            return ConversationHandler.END

        actual_count = len(df_rec)
        if actual_count < n:
            await message.reply_text(
                f"Note: Only {actual_count} tracks match your filters. "
                f"You requested {n}. Consider relaxing preferences for more variety."
            )

        lines = ["Here is your personalized playlist:\n"]
        context.user_data["sp_id"] = []

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
                context.user_data["sp_id"].append(track_id.strip())

            line = f"{i}. {track_name} - {artist_name}"

            details = []
            if year and not pd.isna(year):
                try:
                    details.append(str(int(year)))
                except Exception:
                    details.append(str(year))
            if popularity and not pd.isna(popularity):
                try:
                    details.append(f"pop {int(popularity)}")
                except Exception:
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

    except Exception:
        logger.exception("Error generating playlist")
        await message.reply_text(
            "An unexpected error occurred.\nType /start to try again."
        )
        return ConversationHandler.END

    await message.reply_text(
        "Want to add this playlist to your Spotify?",
        reply_markup=yes_no_kb("yes_on_spoty", "no_on_spoty"),
    )
    return SPOTY

# ======================================
# SPOTY (buttons expected) + /back disabled here by design
# ======================================
async def answer_add_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    message = query.message

    if data == "yes_on_spoty":
        song_list = context.user_data.get("sp_id", [])
        session_token = secrets.token_urlsafe(32)

        login_url = "https://create-playlist-jm.onrender.com/login?" + urllib.parse.urlencode({
            "songs": ",".join(song_list),
            "token": session_token,
        })

        await message.reply_text(
            "Before creating the playlist, you need to connect your Spotify account.\n"
            f"Click here to log in: {login_url}"
        )
        await message.reply_text("Enjoy your music 🎶\nType /start to create another playlist.")
        return ConversationHandler.END

    await message.reply_text("Okay, no problem! 😊")
    await message.reply_text("Enjoy your music 🎶\nType /start to create another playlist.")
    return ConversationHandler.END


async def handle_spoty_text_wrong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _reply_use_buttons(update, "I’m expecting a Yes/No selection.", yes_no_kb("yes_on_spoty", "no_on_spoty"))
    return SPOTY


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Conversation cancelled.\nType /start to begin again.")
    return ConversationHandler.END

# ======================================
# MAIN
# ======================================
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
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mood_text_wrong),
            ],
            ACTIVITY: [
                CallbackQueryHandler(handle_activity_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_activity_text_wrong),
            ],
            PART_OF_DAY: [
                CallbackQueryHandler(handle_time_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_time_text_wrong),
            ],
            WEATHER: [
                CallbackQueryHandler(handle_weather_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_weather_text_wrong),
            ],
            AGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_explorer),
            ],
            EXPLORER: [
                CallbackQueryHandler(handle_explorer_button, pattern="^explorer_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_explorer_text_wrong),
            ],
            FAV_ARTISTS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_language_prefs),
            ],
            ASK_LANG_PREFS: [
                CallbackQueryHandler(handle_ask_langprefs_button, pattern="^langprefs_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ask_langprefs_text_wrong),
            ],
            LANGUAGES: [
                CallbackQueryHandler(handle_language_button, pattern=r"^(lang_toggle:.*|lang_any|lang_done)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_language_text_wrong),
            ],
            N_SONGS: [
                CallbackQueryHandler(handle_n_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, generate_playlist_from_text),
            ],
            SPOTY: [
                CallbackQueryHandler(answer_add_playlist, pattern="^(yes_on_spoty|no_on_spoty)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_spoty_text_wrong),
            ],
        },
        fallbacks=[
            CommandHandler("back", back),
            CommandHandler("cancel", cancel),
        ],
        allow_reentry=True,
    )

    application.add_handler(conv_handler)

    logger.info("Bot started")
    application.run_polling()


if __name__ == "__main__":
    main()