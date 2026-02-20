import os
import logging
import re
import tempfile
from collections import defaultdict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import PyPDF2
import speech_recognition as sr
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = os.environ.get('BOT_TOKEN')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
MISTRAL_MODEL = 'mistral-small'

# Хранилище данных пользователей
# user_data[user_id] = {
#     'words': list,
#     'translations': list,
#     'current_index': int,
#     'wrong_words': set,          # индексы слов с ошибками
#     'review_mode': bool,         # режим повторения ошибок
#     'review_indices': list,      # список индексов для повторения
#     'review_index': int          # текущий индекс в режиме повторения
# }
user_data = defaultdict(dict)

def detect_language(word: str) -> str:
    if re.search('[а-яА-Я]', word):
        return 'russian'
    else:
        return 'english'

def translate_word(word: str, source_lang: str, target_lang: str) -> str:
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    prompt = f"Translate the word '{word}' from {source_lang} to {target_lang}. Give only the translation, no extra text."
    data = {
        'model': MISTRAL_MODEL,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    try:
        response = requests.post('https://api.mistral.ai/v1/chat/completions',
                                 headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        translation = result['choices'][0]['message']['content'].strip()
        return translation
    except Exception as e:
        logger.error(f"Ошибка перевода слова '{word}': {e}")
        return None

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_words_from_pdf(file_path: str) -> list:
    words = []
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    words.extend(lines)
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        raise
    return words

async def voice_to_text(voice_file_path: str) -> str:
    """Конвертирует голосовое сообщение в текст с помощью Google Speech Recognition."""
    try:
        # Конвертируем ogg (формат от Telegram) в wav
        audio = AudioSegment.from_ogg(voice_file_path)
        wav_path = voice_file_path + '.wav'
        audio.export(wav_path, format='wav')

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language='ru-RU')  # Можно автоматически определять язык
        return text
    except Exception as e:
        logger.error(f"Ошибка распознавания речи: {e}")
        return None
    finally:
        # Удаляем временные файлы
        if os.path.exists(voice_file_path):
            os.unlink(voice_file_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для изучения слов. Отправь мне файл в формате .txt или .pdf со словами, "
        "по одному слову в строке. Я определю язык, переведу их и начну опрос.\n"
        "Ты можешь отвечать текстом или голосовым сообщением. После основного списка я предложу повторить ошибочные слова."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    file_name = document.file_name or ""

    if not (file_name.endswith('.txt') or file_name.endswith('.pdf')):
        await update.message.reply_text("Пожалуйста, отправь файл с расширением .txt или .pdf")
        return

    file = await context.bot.get_file(document.file_id)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        if file_name.endswith('.txt'):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            lines = extract_words_from_pdf(tmp_path)
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        await update.message.reply_text("Не удалось прочитать файл. Проверьте его содержимое.")
        return
    finally:
        os.unlink(tmp_path)

    if not lines:
        await update.message.reply_text("Файл не содержит текста или не удалось извлечь слова.")
        return

    first_word = lines[0]
    src_lang = detect_language(first_word)
    tgt_lang = 'russian' if src_lang == 'english' else 'english'

    await update.message.reply_text(f"Определён язык слов: {src_lang}. Получаю переводы...")

    translations = []
    failed = []
    for word in lines:
        trans = translate_word(word, src_lang, tgt_lang)
        if trans:
            translations.append(trans)
        else:
            failed.append(word)

    if not translations:
        await update.message.reply_text("Не удалось перевести ни одного слова. Проверьте API-ключ или попробуйте позже.")
        return

    user_data[user_id] = {
        'words': lines,
        'translations': translations,
        'current_index': 0,
        'wrong_words': set(),
        'review_mode': False,
        'review_indices': [],
        'review_index': 0
    }

    msg = f"Загружено слов: {len(lines)}. Успешно переведено: {len(translations)}."
    if failed:
        msg += f"\nНе удалось перевести: {', '.join(failed[:5])}" + ("..." if len(failed) > 5 else "")
    await update.message.reply_text(msg)

    first_word = user_data[user_id]['words'][0]
    await update.message.reply_text(f"Переведи слово: {first_word}")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстового ответа."""
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Сначала отправь файл со словами.")
        return

    user_answer = update.message.text
    await process_answer(update, user_id, user_answer)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосового ответа."""
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Сначала отправь файл со словами.")
        return

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    await update.message.reply_text("Распознаю речь...")
    text = await voice_to_text(tmp_path)

    if text is None:
        await update.message.reply_text("Не удалось распознать голос. Попробуй ещё раз или напиши текстом.")
        return

    await update.message.reply_text(f"Распознано: {text}")
    await process_answer(update, user_id, text)

async def process_answer(update: Update, user_id: int, user_answer: str):
    """Общая логика проверки ответа (используется и для текста, и для голоса)."""
    data = user_data[user_id]
    review_mode = data['review_mode']

    if not review_mode:
        idx = data['current_index']
        if idx >= len(data['words']):
            await update.message.reply_text("Ты уже прошёл все слова. Начинаю повторение ошибок...")
            await start_review(update, user_id)
            return

        correct_word = data['words'][idx]
        correct_translation = data['translations'][idx]
    else:
        idx = data['review_index']
        if idx >= len(data['review_indices']):
            await update.message.reply_text("🎉 Ты повторил все ошибочные слова! Молодец! Можешь загрузить новый файл.")
            # Очищаем данные (опционально)
            # del user_data[user_id]
            return

        word_index = data['review_indices'][idx]
        correct_word = data['words'][word_index]
        correct_translation = data['translations'][word_index]

    # Проверка ответа
    if normalize(user_answer) == normalize(correct_translation):
        await update.message.reply_text("✅ Верно!")
    else:
        await update.message.reply_text(f"❌ Неверно. Правильный перевод: {correct_translation}")
        # Запоминаем ошибку, если ещё не в режиме повторения
        if not review_mode:
            data['wrong_words'].add(idx)

    # Переход к следующему слову
    if not review_mode:
        data['current_index'] += 1
        if data['current_index'] < len(data['words']):
            next_word = data['words'][data['current_index']]
            await update.message.reply_text(f"Следующее слово: {next_word}")
        else:
            # Основной список закончен
            if data['wrong_words']:
                await update.message.reply_text("Основной список пройден. Переходим к повторению ошибочных слов...")
                await start_review(update, user_id)
            else:
                await update.message.reply_text("🎉 Поздравляю! Ты ответил на все слова правильно! Можешь загрузить новый файл.")
    else:
        data['review_index'] += 1
        if data['review_index'] < len(data['review_indices']):
            next_word_index = data['review_indices'][data['review_index']]
            next_word = data['words'][next_word_index]
            await update.message.reply_text(f"Следующее слово для повторения: {next_word}")
        else:
            await update.message.reply_text("🎉 Ты повторил все ошибочные слова! Молодец! Можешь загрузить новый файл.")

async def start_review(update: Update, user_id: int):
    """Запускает режим повторения ошибочных слов."""
    data = user_data[user_id]
    wrong_indices = sorted(list(data['wrong_words']))
    if not wrong_indices:
        return

    data['review_mode'] = True
    data['review_indices'] = wrong_indices
    data['review_index'] = 0

    first_word_index = wrong_indices[0]
    first_word = data['words'][first_word_index]
    await update.message.reply_text(f"Режим повторения. Переведи слово: {first_word}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Исключение при обработке обновления:", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text("Произошла внутренняя ошибка. Попробуйте позже.")

def main():
    if not BOT_TOKEN or not MISTRAL_API_KEY:
        logger.error("Не заданы переменные окружения BOT_TOKEN и MISTRAL_API_KEY")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_error_handler(error_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
