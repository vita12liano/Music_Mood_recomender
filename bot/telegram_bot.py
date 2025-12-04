#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging

import pandas as pd
from dotenv import load_dotenv     # <-- IMPORT ESSENZIALE
from typing import Optional
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ===============================
# 0. Logging
# ===============================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ===============================
# 0.5 LOAD .ENV (IMPORTANTISSIMO)
# ===============================

# Path al progetto root:  /Music-mood pred/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)      # <-- CARICA IL TOKEN DALLA .env
else:
    print("⚠️  WARNING: .env file not found at:", ENV_PATH)

# ===============================
# 1. Import recommender_05.py
# ===============================

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from recommender_05 import recommend_playlist  # noqa: E402

# ===============================
# 2. Conversation states
# ===============================

(
    MOOD,
    ACTIVITY,
    PART_OF_DAY,
    WEATHER,
    AGE,
    EXPLORER,
    FAV_ARTISTS,
    LANGUAGES,
    N_SONGS,
) = range(9)


# ===============================
# 3. Handlers
# ===============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    await update.message.reply_text(
        "Hi! I'm your Music Mood Recommender Bot 🎧\n\n"
        "I'll ask you a few questions and then build a custom playlist for you.\n\n"
        "First question:\n"
        "👉 *What is your current mood?*\n"
        "_Examples: happy, sad, relaxed, angry, kids, christmas, religious..._",
        parse_mode="Markdown",
    )
    context.user_data.clear()
    return MOOD


async def ask_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    mood = (update.message.text or "").strip()
    context.user_data["mood"] = mood

    await update.message.reply_text(
        "Great! 🎯\n\n"
        "👉 *What are you doing right now?*\n"
        "_Examples: party, study, gym, commute, reading..._",
        parse_mode="Markdown",
    )
    return ACTIVITY


async def ask_part_of_day(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    activity = (update.message.text or "").strip()
    context.user_data["activity"] = activity

    await update.message.reply_text(
        "Nice!\n\n"
        "👉 *What part of the day is it?*\n"
        "_Examples: morning, afternoon, evening, night_",
        parse_mode="Markdown",
    )
    return PART_OF_DAY


async def ask_weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    part_of_day = (update.message.text or "").strip()
    context.user_data["part_of_day"] = part_of_day

    await update.message.reply_text(
        "Got it 🌇\n\n"
        "👉 *How is the weather outside?*\n"
        "_Examples: sunny, rainy, snow, cloudy, stormy..._",
        parse_mode="Markdown",
    )
    return WEATHER


async def ask_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    weather = (update.message.text or "").strip()
    context.user_data["weather"] = weather

    await update.message.reply_text(
        "Thanks!\n\n"
        "👉 *How old are you?* (just a number, e.g. 24)",
        parse_mode="Markdown",
    )
    return AGE


async def ask_explorer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    age_text = (update.message.text or "").strip()
    try:
        age = int(age_text)
        if age <= 0 or age > 120:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "Please type a *valid age* (positive integer, e.g. 24).",
            parse_mode="Markdown",
        )
        return AGE

    context.user_data["age"] = int(age)

    await update.message.reply_text(
        "Perfect 👌\n\n"
        "👉 *Do you want to explore new artists or stay closer to safe/popular choices?*\n"
        "Type *yes* for explorer mode, *no* for a safer playlist.\n"
        "_(You can also reply: y/n, true/false, 1/0)_",
        parse_mode="Markdown",
    )
    return EXPLORER


def _parse_explorer(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in ["yes", "y", "true", "1"]:
        return True
    if t in ["no", "n", "false", "0"]:
        return False
    return None


async def ask_fav_artists(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    explorer_text = (update.message.text or "").strip()
    exp_val = _parse_explorer(explorer_text)
    if exp_val is None:
        await update.message.reply_text(
            "Please answer *yes* or *no* (or y/n, true/false, 1/0) for explorer mode.",
            parse_mode="Markdown",
        )
        return EXPLORER

    context.user_data["explorer"] = exp_val

    await update.message.reply_text(
        "Got it 🎲\n\n"
        "👉 *Write a list of your favourite artists*, separated by commas.\n"
        "_Example: Taylor Swift, Drake, Arctic Monkeys_\n"
        "If you don't want to specify any, just type *none*.",
        parse_mode="Markdown",
    )
    return FAV_ARTISTS


async def ask_languages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    fav_text = (update.message.text or "").strip()
    if fav_text.lower() in ["none", "no", "n", ""]:
        fav_artists = []
    else:
        fav_artists = [a.strip() for a in fav_text.split(",") if a.strip() != ""]

    context.user_data["fav_artists"] = fav_artists

    await update.message.reply_text(
        "Great! 🌍\n\n"
        "👉 *Which languages do you prefer for the songs?*\n"
        "- Use ISO-style codes like: `en`, `it`, `es`...\n"
        "- Separate multiple languages with commas: `en,it` or `es,pt`.\n"
        "- Type *any* if you don't want any language filter.\n",
        parse_mode="Markdown",
    )
    return LANGUAGES


async def ask_n_songs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang_text = (update.message.text or "").strip().lower()

    if lang_text in ["any", "none", "no", "all", ""]:
        language_prefs: list[str] = []
    else:
        language_prefs = [
            l.strip().lower()
            for l in lang_text.split(",")
            if l.strip() != ""
        ]

    context.user_data["language_prefs"] = language_prefs

    await update.message.reply_text(
        "Almost done! 🎵\n\n"
        "👉 *How many songs do you want?* (e.g. 10, 15, 20)\n"
        "_Maximum recommended: 30._",
        parse_mode="Markdown",
    )
    return N_SONGS


async def generate_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    n_text = (update.message.text or "").strip()
    try:
        n = int(n_text)
        if n <= 0:
            raise ValueError
        if n > 50:
            n = 30
    except ValueError:
        await update.message.reply_text(
            "Please type a *valid integer number* of songs (e.g. 10, 15, 20).",
            parse_mode="Markdown",
        )
        return N_SONGS

    context.user_data["n"] = n

    mood = context.user_data.get("mood", "happy")
    activity = context.user_data.get("activity", "none")
    part_of_day = context.user_data.get("part_of_day", "evening")
    weather = context.user_data.get("weather", "clear")
    age = context.user_data.get("age", 25)
    explorer = context.user_data.get("explorer", False)
    fav_artists = context.user_data.get("fav_artists", [])
    language_prefs = context.user_data.get("language_prefs", [])

    await update.message.reply_text("Building your playlist… 🎶 Please wait a moment.")

    try:
        df_rec = recommend_playlist(
            mood=mood,
            activity=activity,
            part_of_day=part_of_day,
            weather=weather,
            age=age,
            explorer=explorer,
            n=n,
            fav_artists=fav_artists,
            language_prefs=language_prefs,
        )

        if df_rec is None or len(df_rec) == 0:
            await update.message.reply_text(
                "Sorry, I couldn't find any suitable tracks for this configuration. "
                "Try changing mood, age range or explorer mode."
            )
            return ConversationHandler.END
        
        lines = []
        lines.append("Here is your recommended playlist: 🎧\n")
        for i, row in enumerate(df_rec.itertuples(index=False), start=1):
            # Access columns in a safe way
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)

            track_name = row_dict.get("track_name", "Unknown track")
            artist_name = row_dict.get("artist_name", "Unknown artist")
            year = row_dict.get("year", "")
            popularity = row_dict.get("popularity", "")
            track_id = row_dict.get("track_id", "")

            # Costruisci URL Spotify dal track_id, se disponibile
            spotify_url = None
            if isinstance(track_id, str) and track_id.strip() != "":
                tid = track_id.strip()
                # se sembra un ID Spotify "pulito", costruisco l'URL
                # (di solito 22 caratteri, ma non mi fisso sulla lunghezza)
                spotify_url = f"https://open.spotify.com/track/{tid}"

            line = f"{i}. {track_name} — {artist_name}"
            details = []
            if year != "" and not pd.isna(year):
                try:
                    details.append(str(int(year)))
                except Exception:
                    details.append(str(year))
            if popularity != "" and not pd.isna(popularity):
                try:
                    details.append(f"pop {int(popularity)}")
                except Exception:
                    details.append(f"pop {popularity}")
            if details:
                line += f" ({', '.join(details)})"

            lines.append(line)

            # Aggiungi il link Spotify su una nuova riga indentata, se presente
            if spotify_url:
                lines.append(f"   ↳ {spotify_url}")

        text = "\n".join(lines)

        MAX_LEN = 4000
        if len(text) <= MAX_LEN:
            await update.message.reply_text(text)
        else:
            for i in range(0, len(text), MAX_LEN):
                await update.message.reply_text(text[i:i + MAX_LEN])

    except Exception:
        logger.exception("Error while generating playlist")
        await update.message.reply_text(
            "An internal error occurred while generating your playlist. "
            "Please try again later."
        )

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Conversation cancelled. Type /start to begin again.")
    return ConversationHandler.END


# ===============================
# 4. Main
# ===============================

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in the .env file!")

    application = ApplicationBuilder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_activity)],
            ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_part_of_day)],
            PART_OF_DAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_weather)],
            WEATHER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_age)],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_explorer)],
            EXPLORER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_fav_artists)],
            FAV_ARTISTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_languages)],
            LANGUAGES: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_n_songs)],
            N_SONGS: [MessageHandler(filters.TEXT & ~filters.COMMAND, generate_playlist)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    logger.info("Bot is starting...")
    application.run_polling()   # <-- niente await, niente asyncio.run


if __name__ == "__main__":
    main()