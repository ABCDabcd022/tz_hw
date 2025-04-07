# Импорт необходимых библиотек
import torch  # Основной фреймворк для работы с нейронными сетями
import time  # Для замера времени выполнения
import json  # Для работы с JSON-файлами (сохранение логов)
import os  # Для работы с файловой системой (создание папок)
from datetime import datetime  # Для временных меток в логах

# Импорт компонентов из transformers
from transformers import (
    RobertaForSequenceClassification,  # Модель RoBERTa для классификации
    RobertaTokenizer,  # Токенизатор для RoBERTa
    Trainer,  # Класс для обучения моделей
    TrainingArguments,  # Конфигурация обучения
    default_data_collator  # Функция для формирования батчей
)

# Импорт других необходимых компонентов
from datasets import load_dataset  # Для загрузки датасетов
from peft import LoraConfig, get_peft_model  # Для работы с LoRA
from transformers.trainer_callback import TrainerCallback  # Для кастомного логгирования

# 1. Конфигурация эксперимента
CONFIG = {
    "model": "FacebookAI/roberta-large",  # Имя предобученной модели
    "dataset": "glue",  # Имя датасета (GLUE benchmark)
    "task": "mrpc",  # Конкретная задача (Microsoft Research Paraphrase Corpus)
    "lora_rank": 8,  # Ранг матриц в LoRA
    "batch_size": 4,  # Размер батча (уменьшен для экономии памяти)
    "max_steps": 1000,  # Количество шагов обучения (как в задании)
    "eval_steps": 100,  # Оценивать модель каждые 100 шагов
    "learning_rate": 1e-4,  # Скорость обучения
    "output_dir": "./lora_results",  # Директория для сохранения результатов
    "log_file": "./training_log.json"  # Файл для сохранения логов
}

# 2. Инициализация системы логирования
def init_logging():
    """Создает директорию для результатов и возвращает структуру для логов"""
    # Создаем директорию, если она не существует
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Возвращаем словарь с начальными данными для логов
    return {
        "config": CONFIG,  # Сохраняем конфигурацию эксперимента
        "start_time": datetime.now().isoformat(),  # Время начала эксперимента
        "metrics": []  # Список для хранения метрик во время обучения
    }

# Инициализируем систему логирования
log_data = init_logging()

# 3. Загрузка и настройка модели с LoRA
print("Loading model and tokenizer...")
# Загружаем токенизатор для RoBERTa
tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model"])
# Загружаем саму модель для классификации текста
model = RobertaForSequenceClassification.from_pretrained(CONFIG["model"], num_labels=2)

# Конфигурация LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=CONFIG["lora_rank"],  # Ранг матриц разложения
    lora_alpha=16,  # Коэффициент масштабирования
    target_modules=["query", "key", "value"],  # Какие слои модифицируем
    lora_dropout=0.1,  # Вероятность dropout для LoRA-слоев
    bias="none",  # Не обучаем параметры смещения
    task_type="SEQ_CLS"  # Тип задачи - классификация последовательностей
)

# Применяем LoRA к модели
model = get_peft_model(model, peft_config)

# 4. Анализ параметров модели
# Считаем количество обучаемых параметров
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Считаем общее количество параметров
total_params = sum(p.numel() for p in model.parameters())

# Выводим информацию о параметрах модели
print(f"\nПараметры модели:")
print(f"• Обучаемые: {trainable_params:,}")  # Форматированный вывод с разделителями тысяч
print(f"• Всего: {total_params:,}")
print(f"• % обучаемых: {100*trainable_params/total_params:.2f}%\n")  # Процент обучаемых параметров

# 5. Подготовка данных
print("Loading dataset...")
# Загружаем датасет MRPC из библиотеки Hugging Face
dataset = load_dataset(CONFIG["dataset"], CONFIG["task"])

# Функция для токенизации текста
def tokenize_fn(examples):
    """Принимает примеры с парами предложений и возвращает токенизированный результат"""
    return tokenizer(
        examples["sentence1"],  # Первое предложение
        examples["sentence2"],  # Второе предложение
        truncation=True,  # Обрезаем текст если он длиннее max_length
        padding="max_length",  # Дополняем до максимальной длины
        max_length=128  # Максимальная длина в токенах
    )

# Применяем токенизацию ко всему датасету (параллельно)
tokenized_data = dataset.map(tokenize_fn, batched=True)

# 6. Настройка метрик и логирования
def compute_metrics(p):
    """Вычисляет accuracy по результатам предсказаний модели"""
    # Получаем предсказания (индекс класса с максимальной вероятностью)
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    # Вычисляем accuracy (среднее количество верных предсказаний)
    acc = (preds == p.label_ids).float().mean().item()
    return {"accuracy": acc}  # Возвращаем словарь с метриками

# Кастомный класс для логирования (наследуется от TrainerCallback)
class CustomLogger(TrainerCallback):
    """Кастомный обработчик событий обучения"""
    
    def on_init_end(self, args, state, control, **kwargs):
        """Вызывается после инициализации обучения"""
        print("Training initialized!")
        return control  # Возвращаем контрольные параметры
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании метрик"""
        # Логируем только на главном процессе (для распределенного обучения)
        if state.is_local_process_zero and logs:
            # Формируем запись лога
            log_entry = {
                "step": state.global_step,  # Номер шага обучения
                "time": datetime.now().isoformat(),  # Временная метка
                **logs  # Все метрики из logs
            }
            # Добавляем запись в общий лог
            log_data["metrics"].append(log_entry)
            
            # Сохраняем логи в файл
            with open(CONFIG["log_file"], "w") as f:
                json.dump(log_data, f, indent=2)  # Красивое форматирование
            
            # Выводим информацию в консоль
            print(f"Step {state.global_step}:", logs)
        return control

# 7. Настройка параметров обучения
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],  # Директория для сохранения результатов
    evaluation_strategy="steps",  # Стратегия оценки (по шагам)
    eval_steps=CONFIG["eval_steps"],  # Оценивать каждые 100 шагов
    logging_steps=CONFIG["eval_steps"],  # Логировать каждые 100 шагов
    save_steps=CONFIG["eval_steps"],  # Сохранять модель каждые 100 шагов
    per_device_train_batch_size=CONFIG["batch_size"],  # Размер батча для обучения
    per_device_eval_batch_size=CONFIG["batch_size"],  # Размер батча для оценки
    max_steps=CONFIG["max_steps"],  # Максимальное количество шагов
    learning_rate=CONFIG["learning_rate"],  # Скорость обучения
    report_to="none",  # Отключаем интеграцию со сторонними сервисами
    load_best_model_at_end=True,  # Загружать лучшую модель в конце
    metric_for_best_model="accuracy",  # Метрика для определения лучшей модели
)

# 8. Запуск процесса обучения
print("\nStarting training...")
start_time = time.time()  # Засекаем время начала обучения

# Инициализируем тренер (объект для обучения)
trainer = Trainer(
    model=model,  # Наша модель
    args=training_args,  # Параметры обучения
    train_dataset=tokenized_data["train"],  # Обучающая выборка
    eval_dataset=tokenized_data["validation"],  # Валидационная выборка
    compute_metrics=compute_metrics,  # Функция для вычисления метрик
    callbacks=[CustomLogger()],  # Наш кастомный обработчик событий
)

# Запускаем обучение
trainer.train()
# Вычисляем общее время обучения
training_time = time.time() - start_time

# 9. Завершение эксперимента и сохранение результатов
# Формируем финальные результаты
results = {
    "final_accuracy": trainer.evaluate()["eval_accuracy"],  # Финальная точность
    "training_time_sec": training_time,  # Общее время обучения в секундах
    "end_time": datetime.now().isoformat()  # Время завершения
}

# Обновляем логи финальными результатами
log_data.update(results)
# Сохраняем логи в файл
with open(CONFIG["log_file"], "w") as f:
    json.dump(log_data, f, indent=2)

# Выводим итоговую информацию
print("\n" + "="*50)
print(f"Training completed in {training_time/60:.2f} minutes")  # Время в минутах
print(f"Final accuracy: {results['final_accuracy']:.4f}")  # Финальная точность
print("="*50)

# Сохраняем финальную модель
model.save_pretrained(os.path.join(CONFIG["output_dir"], "final_model"))