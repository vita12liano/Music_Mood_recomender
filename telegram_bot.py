from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from model_loader import load_model, recommend_playlist

#Conversation states
MOOD, ACTIVITY, TIME, AGE = range(4)

#Load model at startup
print("Model loading...")
model, scaler, le_classes, df_model, feature_cols, device = load_model()
print("Model loaded!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["relax", "happy"], ["sad", "workout"], ["focus", "party"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    
    await update.message.reply_text(
        "Hi! I’m your personalized playlist bot!\n\n"
        "What mood do you want?",
        reply_markup=reply_markup
    )
    return MOOD

async def mood_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mood'] = update.message.text.lower()
    
    keyboard = [["study", "walking"], ["running", "relaxing"], ["party"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    
    await update.message.reply_text(
        "What activity are you doing?",
        reply_markup=reply_markup
    )
    return ACTIVITY

async def activity_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['activity'] = update.message.text.lower()
    
    keyboard = [["morning", "afternoon"], ["evening", "night"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    
    await update.message.reply_text(
        "What time of day is it?",
        reply_markup=reply_markup
    )
    return TIME

async def time_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['time'] = update.message.text.lower()
    
    await update.message.reply_text(
        "How old are you?",
        reply_markup=ReplyKeyboardRemove()
    )
    return AGE

async def age_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    age = update.message.text
    
    #Generate playlist
    await update.message.reply_text("I’m creating your playlist…")
    
    try:
        playlist = recommend_playlist(
            df_model,
            context.user_data['mood'],
            context.user_data['activity'],
            context.user_data['time'],
            age
        )
        
        response = "*Here’s your playlist!*\n\n"
        for i, (_, song) in enumerate(playlist.iterrows(), 1):
            response += f"{i}. *{song['track_name']}*\n    {song['artist_name']}\n\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f" Error: {str(e)}")
    
    await update.message.reply_text(
        "Do you want another playlist? Use /start"
    )
    
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operation cancelled. Use /start to start again.")
    return ConversationHandler.END

def main():
    TOKEN = "7765818401:AAHbRJxR-lmY-OsYo-2jQWBwnM8SAU2E4vo"  
    
    app = Application.builder().token(TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, mood_choice)],
            ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, activity_choice)],
            TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, time_choice)],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    app.add_handler(conv_handler)
    
    print(" Bot started!")
    app.run_polling()

if __name__ == '__main__':
    main()