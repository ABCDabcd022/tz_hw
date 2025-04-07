# Импорт необходимых библиотек
import torch  # Основной фреймворк для работы с нейронными сетями
from transformers import RobertaForSequenceClassification, RobertaTokenizer  # Модель и токенизатор RoBERTa
from datasets import load_dataset  # Для загрузки датасетов
from peft import LoraConfig, get_peft_model  # Для работы с LoRA
from torch.utils.data import Dataset, DataLoader  # Для создания датасетов и загрузчиков данных

# 1. Инициализация модели и токенизатора
model = RobertaForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-large",  # Загрузка предобученной модели RoBERTa-large
    num_labels=2  # Для задачи бинарной классификации (MRPC)
)
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")  # Загрузка токенизатора

# 2. Настройка LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=8,  # Ранг разложения матрицы адаптации
    lora_alpha=16,  # Коэффициент масштабирования
    target_modules=["query", "key", "value"],  # Применяем LoRA к этим слоям attention-механизма
    lora_dropout=0.1,  # Dropout для LoRA-слоев
    bias="none",  # Не добавляем смещение
    task_type="SEQ_CLS"  # Тип задачи - классификация последовательностей
)
model = get_peft_model(model, peft_config)  # Применяем LoRA к модели

# Определяем устройство для вычислений (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Переносим модель на выбранное устройство

# 3. Подготовка данных
class MRPCDataset(Dataset):
    """Кастомный класс датасета для MRPC"""
    
    def __init__(self, dataset, tokenizer):
        """Инициализация датасета"""
        self.dataset = dataset  # Исходный датасет
        self.tokenizer = tokenizer  # Токенизатор
        
    def __len__(self):
        """Возвращает количество примеров в датасете"""
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """Возвращает один элемент датасета по индексу"""
        item = self.dataset[idx]  # Получаем элемент по индексу
        
        # Токенизируем пару предложений
        encoding = self.tokenizer(
            item["sentence1"],  # Первое предложение
            item["sentence2"],  # Второе предложение
            padding='max_length',  # Дополняем до максимальной длины
            truncation=True,  # Обрезаем, если превышает max_length
            max_length=128,  # Максимальная длина последовательности
            return_tensors="pt"  # Возвращаем тензоры PyTorch
        )
        
        # Возвращаем словарь с данными
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Токены (удаляем лишнюю размерность)
            'attention_mask': encoding['attention_mask'].squeeze(),  # Маска внимания
            'labels': torch.tensor(item['label'], dtype=torch.long)  # Метки класса
        }

# Загрузка датасета MRPC
dataset = load_dataset("glue", "mrpc")  # Загружаем датасет из библиотеки Hugging Face

# Создаем объекты датасета и загрузчика данных
train_dataset = MRPCDataset(dataset["train"], tokenizer)  # Обучающий датасет
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # Размер батча
    shuffle=True  # Перемешивание данных
)

# 4. Обучение с использованием MeZO (Memory-efficient Zeroth-order Optimization)
epsilon = 1e-3  # Величина возмущения для оценки градиента
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)  # Оптимизатор SGD

for step, batch in enumerate(train_loader):
    if step >= 100:  # Ограничиваем обучение 100 шагами для демонстрации
        break
        
    # Переносим данные на устройство (GPU/CPU)
    inputs = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device)
    }
    
    # Сохраняем исходные параметры модели
    original_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # Генерируем случайные возмущения для каждого параметра
    perturbations = {n: torch.randn_like(p) for n, p in model.named_parameters() if p.requires_grad}
    
    # Прямой проход с положительным возмущением (θ + ε·z)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.add_(epsilon * perturbations[n])
    outputs = model(**inputs)
    loss_plus = outputs.loss  # Потери с положительным возмущением
    
    # Прямой проход с отрицательным возмущением (θ - ε·z)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.sub_(2 * epsilon * perturbations[n])  # Вычитаем 2ε·z чтобы получить θ - ε·z
    outputs = model(**inputs)
    loss_minus = outputs.loss  # Потери с отрицательным возмущением
    
    # Восстанавливаем исходные параметры модели
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(original_params[n])
    
    # Вычисляем градиенты по методу MeZO
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:  # Инициализируем градиент, если его нет
                p.grad = torch.zeros_like(p)
            # Формула MeZO: ∇L ≈ z·(L(θ + ε·z) - L(θ - ε·z)) / (2ε)
            p.grad.add_(perturbations[n] * (loss_plus - loss_minus) / (2 * epsilon))
    
    optimizer.step()  # Обновляем параметры модели
    optimizer.zero_grad()  # Обнуляем градиенты
    
    # Выводим информацию о потерях каждые 10 шагов
    if step % 10 == 0:
        print(f"Step {step}: Loss+={loss_plus.item():.4f}, Loss-={loss_minus.item():.4f}")

# Сохраняем обученную модель
print("Обучение завершено!")
model.save_pretrained("./mezo_lora_results")  # Сохраняем модель в указанную директорию