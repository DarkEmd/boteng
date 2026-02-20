import os
import logging
import re
import tempfile
import difflib
from collections import defaultdict
from typing import List, Tuple, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, CallbackQueryHandler
)
import requests
import PyPDF2
import speech_recognition as sr
from pydub import AudioSegment
import json

# Настройка логирования
logging.basicConfig(
    format='%(astime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = os.environ.get('BOT_TOKEN')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
MISTRAL_MODEL = 'mistral-small'

# Режимы перевода
class TranslationMode:
    EN_RU = "en_ru"  # английский → русский
    RU_EN = "ru_en"  # русский → английский

# Хранилище данных пользователей
user_data = defaultdict(lambda: {
    'words': [],
    'translations': [],
    'current_index': 0,
    'wrong_words': set(),
    'review_mode': False,
    'review_indices': [],
    'review_index': 0,
    'mode': None,  # будет установлен после загрузки
    'source_lang': None,
    'target_lang': None,
    'failed_words': []  # слова, которые не удалось перевести
})

# Кэш для переводов (чтобы не дёргать API повторно)
translation_cache = {}

def clean_text(text: str) -> str:
    """
    Очистка текста от лишних символов, номеров страниц, артефактов PDF.
    """
    if not text:
        return ""
    
    # Удаляем номера страниц (часто встречаются в PDF)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Удаляем странные символы, оставляем буквы, цифры, пробелы и базовые знаки препинания
    text = re.sub(r'[^\w\s\-.,!?;:]', ' ', text)
    
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Разбиваем на отдельные слова, если текст слипся (может быть при распознавании PDF)
    # Но не разбиваем слишком агрессивно
    
    return text.strip()

def split_into_words(text: str) -> List[str]:
    """
    Интеллектуальное разбиение текста на отдельные слова/фразы.
    Пытается определить, где одно слово, а где несколько.
    """
    # Сначала очищаем текст
    text = clean_text(text)
    
    # Если текст пустой
    if not text:
        return []
    
    # Пробуем разбить по строкам, точкам с запятой, запятым (если это список)
    # Но сохраняем фразы из нескольких слов как единое целое
    lines = []
    
    # Сначала пробуем разбить по переводу строки
    if '\n' in text:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
    else:
        # Если нет перевода строки, пробуем разбить по точке с запятой или запятой
        if ';' in text:
            lines = [part.strip() for part in text.split(';') if part.strip()]
        elif ',' in text:
            lines = [part.strip() for part in text.split(',') if part.strip()]
        else:
            # Если ничего не помогло, считаем всё одной фразой
            lines = [text]
    
    # Дополнительная обработка: если строка слишком длинная (> 50 символов),
    # возможно, это несколько слов без разделителей
    final_words = []
    for line in lines:
        if len(line) > 50 and ' ' not in line:
            # Слипшийся текст - пробуем разбить по заглавным буквам
            # Например: "AppleBananaCherry" -> ["Apple", "Banana", "Cherry"]
            parts = re.findall(r'[A-Z][a-z]*|[a-z]+', line)
            if len(parts) > 1:
                final_words.extend(parts)
            else:
                final_words.append(line)
        else:
            final_words.append(line)
    
    return final_words

def extract_words_from_pdf(file_path: str) -> List[str]:
    """
    Улучшенное извлечение слов из PDF с очисткой от артефактов.
    """
    words = []
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Собираем весь текст со всех страниц
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            if not full_text.strip():
                # Если текст не извлёкся, возможно это сканированный PDF
                return []
            
            # Разбиваем на слова с помощью улучшенного парсера
            words = split_into_words(full_text)
            
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        raise
    
    return words

def extract_words_from_txt(file_path: str) -> List[str]:
    """
    Извлечение слов из текстового файла с очисткой.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return split_into_words(content)

def detect_language(word: str) -> str:
    """
    Определение языка с учётом смешанных случаев.
    """
    if not word:
        return 'unknown'
    
    # Считаем количество кириллических символов
    cyrillic_count = len(re.findall('[а-яА-Я]', word))
    latin_count = len(re.findall('[a-zA-Z]', word))
    
    if cyrillic_count > latin_count:
        return 'russian'
    elif latin_count > cyrillic_count:
        return 'english'
    else:
        # Если поровну или оба нули, смотрим на первый символ
        if word and word[0].isalpha():
            if re.match('[а-яА-Я]', word[0]):
                return 'russian'
            else:
                return 'english'
    return 'unknown'

def normalize_answer(text: str) -> str:
    """
    Нормализация ответа для сравнения.
    Удаляет все лишние символы, приводит к нижнему регистру.
    """
    # Удаляем всё, кроме букв, цифр и пробелов
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text

def are_similar_meaning(answer: str, correct: str, threshold: float = 0.85) -> bool:
    """
    Проверка, являются ли ответы синонимичными с помощью Mistral API.
    """
    # Сначала пробуем прямое сравнение
    if normalize_answer(answer) == normalize_answer(correct):
        return True
    
    # Если простое сравнение не прошло, спрашиваем у Mistral
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
        # Если API недоступен, используем расстояние Левенштейна как fallback
        similarity = difflib.SequenceMatcher(None, 
                                           normalize_answer(answer), 
                                           normalize_answer(correct)).ratio()
        return similarity > threshold

def translate_word(word: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Получение перевода через Mistral API с кэшированием.
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
        
        # Кэшируем результат
        translation_cache[cache_key] = translation
        return translation
        
    except Exception as e:
        logger.error(f"Ошибка перевода слова '{word}': {e}")
        return None

async def show_mode_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Показывает клавиатуру для выбора режима перевода.
    """
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 Английский → Русский 🇷🇺", callback_data="mode_en_ru"),
        ],
        [
            InlineKeyboardButton("🇷🇺 Русский → Английский 🇬🇧", callback_data="mode_ru_en"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Выберите режим перевода:",
        reply_markup=reply_markup
    )

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка выбора режима.
    """
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    mode = query.data
    
    if mode == "mode_en_ru":
        user_data[user_id]['mode'] = TranslationMode.EN_RU
        user_data[user_id]['source_lang'] = 'english'
        user_data[user_id]['target_lang'] = 'russian'
        await query.edit_message_text("Выбран режим: Английский → Русский")
    elif mode == "mode_ru_en":
        user_data[user_id]['mode'] = TranslationMode.RU_EN
        user_data[user_id]['source_lang'] = 'russian'
        user_data[user_id]['target_lang'] = 'english'
        await query.edit_message_text("Выбран режим: Русский → Английский")
    
    await query.message.reply_text("Теперь отправьте файл со словами.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /start.
    """
    welcome_text = (
        "👋 Привет! Я бот для изучения слов.\n\n"
        "📌 **Как пользоваться:**\n"
        "1. Выберите режим перевода\n"
        "2. Отправьте файл .txt или .pdf со словами\n"
        "3. Отвечайте текстом или голосом\n"
        "4. Я запомню ошибки и предложу повторить\n\n"
        "✅ **Улучшения:**\n"
        "• Распознаю синонимы\n"
        "• Игнорирую лишние символы\n"
        "• Обрабатываю PDF с картинками\n"
        "• Запоминаю непереведённые слова"
    )
    
    await update.message.reply_text(welcome_text)
    await show_mode_selection(update, context)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка загруженного файла.
    """
    user_id = update.effective_user.id
    document = update.message.document
    
    # Проверяем, выбран ли режим
    if user_data[user_id]['mode'] is None:
        await update.message.reply_text("Сначала выберите режим перевода командой /start")
        return
    
    file_name = document.file_name or ""
    if not (file_name.endswith('.txt') or file_name.endswith('.pdf')):
        await update.message.reply_text("Пожалуйста, отправь файл с расширением .txt или .pdf")
        return
    
    # Скачиваем файл
    file = await context.bot.get_file(document.file_id)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    # Извлекаем слова
    try:
        if file_name.endswith('.txt'):
            lines = extract_words_from_txt(tmp_path)
        else:  # PDF
            lines = extract_words_from_pdf(tmp_path)
            
        if not lines:
            await update.message.reply_text(
                "Не удалось извлечь слова из файла. Возможно, это сканированный PDF без текстового слоя.\n"
                "Попробуйте другой файл или текстовый формат."
            )
            return
            
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        await update.message.reply_text("Ошибка при чтении файла. Проверьте его содержимое.")
        return
    finally:
        os.unlink(tmp_path)
    
    # Определяем язык первого слова для проверки
    first_word_lang = detect_language(lines[0])
    expected_source = user_data[user_id]['source_lang']
    
    if first_word_lang != expected_source and first_word_lang != 'unknown':
        await update.message.reply_text(
            f"⚠️ Внимание! Язык первого слова ({first_word_lang}) "
            f"не соответствует выбранному режиму ({expected_source}).\n"
            f"Продолжаем, но возможны неточности."
        )
    
    # Получаем переводы
    await update.message.reply_text(f"📝 Найдено слов: {len(lines)}. Получаю переводы...")
    
    translations = []
    failed = []
    
    for i, word in enumerate(lines):
        # Показываем прогресс каждые 10 слов
        if i > 0 and i % 10 == 0:
            await update.message.reply_text(f"⏳ Прогресс: {i}/{len(lines)} слов...")
        
        trans = translate_word(word, 
                             user_data[user_id]['source_lang'],
                             user_data[user_id]['target_lang'])
        if trans:
            translations.append(trans)
        else:
            failed.append(word)
            translations.append("")  # плейсхолдер
    
    # Сохраняем данные
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
    
    # Отчёт
    msg = f"✅ Загружено: {len(lines)} слов\n"
    msg += f"✅ Переведено: {len(translations) - len(failed)}\n"
    
    if failed:
        msg += f"❌ Не удалось перевести: {len(failed)} слов\n"
        msg += f"📋 Первые 5: {', '.join(failed[:5])}"
    
    if translations and translations[0]:  # если первое слово перевелось
        msg += f"\n\n🎯 Первое слово: {lines[0]}"
    
    await update.message.reply_text(msg)
    
    # Начинаем опрос
    await send_next_word(update, user_id)

async def send_next_word(update: Update, user_id: int):
    """
    Отправляет следующее слово пользователю.
    """
    data = user_data[user_id]
    
    if data['review_mode']:
        if data['review_index'] >= len(data['review_indices']):
            await update.message.reply_text(
                "🎉 Поздравляю! Вы повторили все ошибочные слова!\n"
                "Отправьте новый файл, чтобы продолжить."
            )
            return
        
        word_idx = data['review_indices'][data['review_index']]
        word = data['words'][word_idx]
        await update.message.reply_text(f"🔄 Повторение: {word}")
        
    else:
        if data['current_index'] >= len(data['words']):
            # Основной список закончен
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
                    "Отправьте новый файл, чтобы продолжить."
                )
            return
        
        word = data['words'][data['current_index']]
        progress = f"[{data['current_index'] + 1}/{len(data['words'])}]"
        await update.message.reply_text(f"{progress} Переведи: {word}")

def start_review(user_id: int):
    """
    Запускает режим повторения.
    """
    data = user_data[user_id]
    wrong_indices = sorted(list(data['wrong_words']))
    data['review_mode'] = True
    data['review_indices'] = wrong_indices
    data['review_index'] = 0

async def process_answer(update: Update, user_id: int, user_answer: str):
    """
    Обработка ответа пользователя.
    """
    data = user_data[user_id]
    
    if not data['words']:
        await update.message.reply_text("Сначала отправьте файл со словами.")
        return
    
    # Определяем текущее слово
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
            f"⚠️ Для слова '{current_word}' нет перевода.\n"
            f"Правильный ответ: [перевод не получен]"
        )
        # Пропускаем это слово
        if data['review_mode']:
            data['review_index'] += 1
        else:
            data['current_index'] += 1
        await send_next_word(update, user_id)
        return
    
    # Проверяем ответ
    is_correct = are_similar_meaning(user_answer, correct_answer)
    
    if is_correct:
        await update.message.reply_text("✅ Верно!")
    else:
        await update.message.reply_text(f"❌ Неверно. Правильный перевод: {correct_answer}")
        if not data['review_mode']:
            data['wrong_words'].add(word_idx)
    
    # Переходим к следующему слову
    if data['review_mode']:
        data['review_index'] += 1
    else:
        data['current_index'] += 1
    
    await send_next_word(update, user_id)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка текстовых сообщений.
    """
    user_id = update.effective_user.id
    await process_answer(update, user_id, update.message.text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка голосовых сообщений.
    """
    user_id = update.effective_user.id
    
    if not user_data[user_id]['words']:
        await update.message.reply_text("Сначала отправьте файл со словами.")
        return
    
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    await update.message.reply_text("🎤 Распознаю речь...")
    
    try:
        # Конвертируем ogg в wav
        audio = AudioSegment.from_ogg(tmp_path)
        wav_path = tmp_path + '.wav'
        audio.export(wav_path, format='wav')
        
        # Распознаём
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            # Определяем язык для распознавания
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
        # Чистим временные файлы
        for path in [tmp_path, wav_path]:
            if os.path.exists(path):
                os.unlink(path)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /stats - показывает статистику.
    """
    user_id = update.effective_user.id
    data = user_data[user_id]
    
    if not data['words']:
        await update.message.reply_text("Нет данных. Сначала отправьте файл.")
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
    """
    Глобальный обработчик ошибок.
    """
    logger.error(f"Ошибка: {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "😵 Произошла внутренняя ошибка. Попробуйте позже."
        )

def main():
    """
    Запуск бота.
    """
    if not BOT_TOKEN or not MISTRAL_API_KEY:
        logger.error("Не заданы переменные окружения BOT_TOKEN и MISTRAL_API_KEY")
        return
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))
    
    # Обработчики сообщений
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^mode_"))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == '__main__':
    main()
