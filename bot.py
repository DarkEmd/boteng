import os
import logging
import re
import tempfile
import difflib
from collections import defaultdict
from typing import List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, CallbackQueryHandler
)
import requests
import PyPDF2
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import pytesseract

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
user_data = defaultdict(lambda: {
    'words': [],
    'translations': [],
    'current_index': 0,
    'wrong_words': set(),
    'review_mode': False,
    'review_indices': [],
    'review_index': 0,
    'target_lang': None,        # целевой язык, выбранный пользователем
    'source_lang': None,         # исходный язык, определённый из файла
    'failed_words': []
})

# Кэш для переводов и проверки синонимов
translation_cache = {}

# ----------------------------------------------------------------------
# Функции очистки и обработки текста (исправленные)
# ----------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Удаляет мусор из извлечённого текста (номера страниц, лишние символы).
    Восклицательные и вопросительные знаки заменяются на пробелы (чтобы не склеивали слова).
    """
    if not text:
        return ""
    # Удаляем номера страниц (часто встречаются в PDF)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Заменяем нежелательные символы на пробелы.
    # Оставляем: буквы, цифры, пробелы, дефис, точку, запятую, ; : | • / — (тире)
    # Восклицательные и вопросительные знаки удаляем (заменяем на пробел)
    text = re.sub(r'[^\w\s\-.,;:|•/—]', ' ', text)
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_translation(text: str) -> str:
    """
    Очищает перевод от лишних знаков препинания в конце.
    """
    text = re.sub(r'[!?.]+$', '', text.strip())
    text = ' '.join(text.split())
    return text

def split_into_words(text: str) -> List[str]:
    """
    Разбивает текст на отдельные слова, используя пробелы и знаки-разделители.
    Дефис оставляется как часть слова (для составных слов).
    """
    text = clean_text(text)
    if not text:
        return []
    
    # Разбиваем по пробельным символам и разделителям (запятая, точка с запятой, |, /, —)
    parts = re.split(r'[\s,;|/—]+', text)
    # Фильтруем пустые строки
    words = [p.strip() for p in parts if p.strip()]
    return words

def extract_words_from_pdf(file_path: str) -> List[str]:
    """
    Извлекает слова из PDF с добавлением пробелов между страницами.
    """
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "  # пробел между страницами
            if not full_text.strip():
                return []
            return split_into_words(full_text)
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        raise

def extract_words_from_txt(file_path: str) -> List[str]:
    """
    Извлекает слова из текстового файла.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return split_into_words(content)

def extract_words_from_image(file_path: str) -> List[str]:
    """
    Извлекает текст из изображения с помощью Tesseract OCR.
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='rus+eng')
        return split_into_words(text)
    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        raise

def detect_language(word: str) -> str:
    """
    Определяет язык слова по соотношению кириллицы и латиницы.
    """
    if not word:
        return 'unknown'
    cyrillic = len(re.findall('[а-яА-Я]', word))
    latin = len(re.findall('[a-zA-Z]', word))
    if cyrillic > latin:
        return 'russian'
    elif latin > cyrillic:
        return 'english'
    else:
        if word and word[0].isalpha():
            return 'russian' if re.match('[а-яА-Я]', word[0]) else 'english'
    return 'unknown'

def normalize_answer(text: str) -> str:
    """
    Приводит ответ к стандартному виду для сравнения.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join(text.split())
    return text

def are_similar_meaning(answer: str, correct: str, threshold: float = 0.85) -> bool:
    """
    Проверяет, являются ли ответ и правильный перевод синонимами (через Mistral).
    При ошибке API использует расстояние Левенштейна как fallback.
    """
    if normalize_answer(answer) == normalize_answer(correct):
        return True
    
    cache_key = f"similar_{answer}_{correct}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    try:
        headers = {
            'Authorization': f'Bearer {MISTRAL_API_KEY}',
            'Content-Type': 'application/json'
        }
        prompt = (f"Are these two phrases translations of each other or synonyms? "
                  f"Answer only 'yes' or 'no'.\n"
                  f"Phrase 1: '{answer}'\n"
                  f"Phrase 2: '{correct}'")
        
        data = {
            'model': MISTRAL_MODEL,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        
        response = requests.post('https://api.mistral.ai/v1/chat/completions',
                                 headers=headers, json=data, timeout=5)
        response.raise_for_status()
        result = response.json()
        ai_answer = result['choices'][0]['message']['content'].strip().lower()
        
        is_similar = 'yes' in ai_answer
        translation_cache[cache_key] = is_similar
        return is_similar
        
    except Exception as e:
        logger.error(f"Ошибка проверки синонимов: {e}")
        # Fallback: сравнение по Левенштейну
        similarity = difflib.SequenceMatcher(None,
                                           normalize_answer(answer),
                                           normalize_answer(correct)).ratio()
        return similarity > threshold

def translate_word(word: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Получает перевод через Mistral API. Результат очищается и кэшируется.
    """
    cache_key = f"{source_lang}_{target_lang}_{word}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    prompt = (f"Translate the word or phrase '{word}' from {source_lang} to {target_lang}. "
              f"Give only the most common translation, no extra text. "
              f"If there are multiple possible translations, give the most common one.")
    
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
        translation = clean_translation(translation)
        translation_cache[cache_key] = translation
        return translation
    except Exception as e:
        logger.error(f"Ошибка перевода слова '{word}': {e}")
        return None

# ----------------------------------------------------------------------
# Обработчики команд и сообщений
# ----------------------------------------------------------------------

async def show_mode_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🇬🇧 Английский → Русский 🇷🇺", callback_data="target_ru")],
        [InlineKeyboardButton("🇷🇺 Русский → Английский 🇬🇧", callback_data="target_en")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Выберите, **на какой язык** переводить слова.\n"
        "Язык исходных слов определится автоматически.",
        reply_markup=reply_markup
    )

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    target = query.data  # "target_ru" или "target_en"
    
    if target == "target_ru":
        user_data[user_id]['target_lang'] = 'russian'
        await query.edit_message_text("✅ Выбран режим: переводим **на русский** язык.")
    elif target == "target_en":
        user_data[user_id]['target_lang'] = 'english'
        await query.edit_message_text("✅ Выбран режим: переводим **на английский** язык.")
    
    await query.message.reply_text(
        "Теперь отправьте файл (.txt/.pdf) или фотографию со словами.\n"
        "Я сам определю их язык и переведу на выбранный вами язык."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Привет! Я бот для изучения слов.\n\n"
        "📌 **Как пользоваться:**\n"
        "1. Выберите, **на какой язык** переводить\n"
        "2. Отправьте файл .txt, .pdf или фотографию со словами\n"
        "3. Отвечайте текстом или голосом\n"
        "4. Я запомню ошибки и предложу повторить\n\n"
        "✅ **Возможности:**\n"
        "• Распознаю текст на фото (OCR)\n"
        "• Понимаю синонимы\n"
        "• Игнорирую лишние символы\n"
        "• Разбиваю строки с разделителями (•, |, / и др.)\n"
        "• Поддерживаю голосовой ввод"
    )
    await update.message.reply_text(welcome_text)
    await show_mode_selection(update, context)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает загруженные файлы (TXT, PDF)."""
    user_id = update.effective_user.id
    document = update.message.document
    
    if user_data[user_id]['target_lang'] is None:
        await update.message.reply_text("Сначала выберите режим перевода командой /start")
        return
    
    file_name = document.file_name or ""
    if not (file_name.endswith('.txt') or file_name.endswith('.pdf')):
        await update.message.reply_text("Пожалуйста, отправьте файл с расширением .txt или .pdf")
        return
    
    file = await context.bot.get_file(document.file_id)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    try:
        if file_name.endswith('.txt'):
            lines = extract_words_from_txt(tmp_path)
        else:  # PDF
            lines = extract_words_from_pdf(tmp_path)
        
        if not lines:
            await update.message.reply_text(
                "Не удалось извлечь слова из файла. Возможно, это сканированный PDF без текстового слоя.\n"
                "Попробуйте другой файл или используйте фотографию."
            )
            return
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        await update.message.reply_text("Ошибка при чтении файла. Проверьте его содержимое.")
        return
    finally:
        os.unlink(tmp_path)
    
    await process_words(update, user_id, lines)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает фотографии (OCR)."""
    user_id = update.effective_user.id
    photo = update.message.photo[-1]  # берём самое большое фото
    
    if user_data[user_id]['target_lang'] is None:
        await update.message.reply_text("Сначала выберите режим перевода командой /start")
        return
    
    file = await context.bot.get_file(photo.file_id)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    await update.message.reply_text("🖼️ Распознаю текст на фото...")
    
    try:
        lines = extract_words_from_image(tmp_path)
        if not lines:
            await update.message.reply_text(
                "Не удалось распознать текст на фото. Убедитесь, что фото чёткое и текст читаемый."
            )
            return
    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        await update.message.reply_text("Ошибка при распознавании текста.")
        return
    finally:
        os.unlink(tmp_path)
    
    await process_words(update, user_id, lines)

async def process_words(update: Update, user_id: int, lines: List[str]):
    """Общая обработка списка слов: определение языка, перевод, сохранение."""
    # Определяем язык первого слова
    first_word_lang = detect_language(lines[0])
    target_lang = user_data[user_id]['target_lang']
    
    if first_word_lang == 'unknown':
        await update.message.reply_text(
            "❌ Не удалось определить язык слов. Попробуйте другой файл или фото."
        )
        return
    
    if first_word_lang == target_lang:
        await update.message.reply_text(
            f"❌ Язык слов ({first_word_lang}) совпадает с целевым языком ({target_lang}).\n"
            f"Невозможно перевести. Проверьте выбранный режим или загрузите слова на другом языке."
        )
        return
    
    source_lang = first_word_lang
    user_data[user_id]['source_lang'] = source_lang  # сохраняем для информации
    
    await update.message.reply_text(
        f"📝 Найдено слов: {len(lines)}. Язык: {source_lang}. Перевожу на {target_lang}..."
    )
    
    translations = []
    failed = []
    
    for i, word in enumerate(lines):
        if i > 0 and i % 10 == 0:
            await update.message.reply_text(f"⏳ Прогресс: {i}/{len(lines)} слов...")
        
        trans = translate_word(word, source_lang, target_lang)
        if trans:
            translations.append(trans)
        else:
            failed.append(word)
            translations.append("")
    
    # Сохраняем
    user_data[user_id].update({
        'words': lines,
        'translations': translations,
        'current_index': 0,
        'wrong_words': set(),
        'review_mode': False,
        'review_indices': [],
        'review_index': 0,
        'failed_words': failed
    })
    
    msg = f"✅ Загружено: {len(lines)} слов\n"
    msg += f"✅ Переведено: {len(translations) - len(failed)}\n"
    if failed:
        msg += f"❌ Не удалось перевести: {len(failed)} слов\n"
        msg += f"📋 Первые 5: {', '.join(failed[:5])}"
    
    if translations and translations[0]:
        msg += f"\n\n🎯 Первое слово: {lines[0]}"
    
    await update.message.reply_text(msg)
    
    # Начинаем опрос
    await send_next_word(update, user_id)

async def send_next_word(update: Update, user_id: int):
    data = user_data[user_id]
    
    if data['review_mode']:
        if data['review_index'] >= len(data['review_indices']):
            await update.message.reply_text(
                "🎉 Поздравляю! Вы повторили все ошибочные слова!\n"
                "Отправьте новый файл или фото, чтобы продолжить."
            )
            return
        word_idx = data['review_indices'][data['review_index']]
        word = data['words'][word_idx]
        await update.message.reply_text(f"🔄 Повторение: {word}")
    else:
        if data['current_index'] >= len(data['words']):
            if data['wrong_words']:
                await update.message.reply_text(
                    f"📊 Основной список пройден!\n"
                    f"❌ Ошибок: {len(data['wrong_words'])}\n"
                    f"🔄 Переходим к повторению..."
                )
                start_review(user_id)
                await send_next_word(update, user_id)
            else:
                await update.message.reply_text(
                    "🎉 Идеально! Все слова правильные!\n"
                    "Отправьте новый файл или фото, чтобы продолжить."
                )
            return
        
        word = data['words'][data['current_index']]
        progress = f"[{data['current_index'] + 1}/{len(data['words'])}]"
        await update.message.reply_text(f"{progress} Переведи: {word}")

def start_review(user_id: int):
    data = user_data[user_id]
    wrong_indices = sorted(list(data['wrong_words']))
    data['review_mode'] = True
    data['review_indices'] = wrong_indices
    data['review_index'] = 0

async def process_answer(update: Update, user_id: int, user_answer: str):
    data = user_data[user_id]
    
    if not data['words']:
        await update.message.reply_text("Сначала отправьте файл или фото со словами.")
        return
    
    if data['review_mode']:
        if data['review_index'] >= len(data['review_indices']):
            await update.message.reply_text("Режим повторения завершён!")
            return
        word_idx = data['review_indices'][data['review_index']]
    else:
        if data['current_index'] >= len(data['words']):
            return
        word_idx = data['current_index']
    
    current_word = data['words'][word_idx]
    correct_answer = data['translations'][word_idx]
    
    if not correct_answer:
        await update.message.reply_text(
            f"⚠️ Для слова '{current_word}' нет перевода (возможно, ошибка API).\n"
            f"Пропускаем..."
        )
        if data['review_mode']:
            data['review_index'] += 1
        else:
            data['current_index'] += 1
        await send_next_word(update, user_id)
        return
    
    is_correct = are_similar_meaning(user_answer, correct_answer)
    
    if is_correct:
        await update.message.reply_text("✅ Верно!")
    else:
        await update.message.reply_text(f"❌ Неверно. Правильный перевод: {correct_answer}")
        if not data['review_mode']:
            data['wrong_words'].add(word_idx)
    
    if data['review_mode']:
        data['review_index'] += 1
    else:
        data['current_index'] += 1
    
    await send_next_word(update, user_id)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await process_answer(update, user_id, update.message.text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not user_data[user_id]['words']:
        await update.message.reply_text("Сначала отправьте файл или фото со словами.")
        return
    
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    await update.message.reply_text("🎤 Распознаю речь...")
    
    try:
        audio = AudioSegment.from_ogg(tmp_path)
        wav_path = tmp_path + '.wav'
        audio.export(wav_path, format='wav')
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            # Язык распознавания = целевой язык (на который переводим)
            lang = 'ru-RU' if user_data[user_id]['target_lang'] == 'russian' else 'en-US'
            text = recognizer.recognize_google(audio_data, language=lang)
        
        await update.message.reply_text(f"📝 Распознано: {text}")
        await process_answer(update, user_id, text)
        
    except sr.UnknownValueError:
        await update.message.reply_text("😕 Не удалось распознать речь. Попробуйте ещё раз или напишите текстом.")
    except Exception as e:
        logger.error(f"Ошибка распознавания: {e}")
        await update.message.reply_text("⚠️ Ошибка при обработке голоса.")
    finally:
        for path in [tmp_path, wav_path]:
            if os.path.exists(path):
                os.unlink(path)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    data = user_data[user_id]
    
    if not data['words']:
        await update.message.reply_text("Нет данных. Сначала отправьте файл или фото.")
        return
    
    total = len(data['words'])
    learned = total - len(data['wrong_words'])
    failed = len(data['failed_words'])
    
    stats = (
        f"📊 **Статистика:**\n"
        f"📚 Всего слов: {total}\n"
        f"✅ Выучено: {learned}\n"
        f"❌ Ошибок: {len(data['wrong_words'])}\n"
        f"⚠️ Не переведено: {failed}\n"
        f"📈 Прогресс: {learned}/{total} ({learned/total*100:.1f}%)"
    )
    await update.message.reply_text(stats)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "😵 Произошла внутренняя ошибка. Попробуйте позже."
        )

# ----------------------------------------------------------------------
# Запуск бота
# ----------------------------------------------------------------------

def main():
    if not BOT_TOKEN or not MISTRAL_API_KEY:
        logger.error("Не заданы переменные окружения BOT_TOKEN и MISTRAL_API_KEY")
        return
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^target_"))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_error_handler(error_handler)
    
    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == '__main__':
    main()
