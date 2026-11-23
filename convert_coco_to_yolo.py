
import json
import random
import shutil
from pathlib import Path
from collections import Counter

# ==========================================
# НАСТРОЙКИ
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = BASE_DIR / "result.json"

OUTPUT_DIR = BASE_DIR / "dataset"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASS_ID = 0

# ==========================================
# 1. ЗАГРУЗКА COCO JSON
# ==========================================

print("Загружаем аннотации...")

with open(JSON_PATH, "r", encoding="utf-8") as file:
    coco = json.load(file)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

print(f"Изображений: {len(images)}")
print(f"Аннотаций: {len(annotations)}")
print(f"Категории: {categories}")

# ==========================================
# 2. ПРОВЕРКА ДАННЫХ
# ==========================================

if len(categories) != 1:
    raise ValueError("Ожидался ровно один класс!")

if categories[0]["name"] != "wagon_number":
    raise ValueError("Неожиданное название класса!")

if categories[0]["id"] != CLASS_ID:
    raise ValueError("ID класса не совпадает с CLASS_ID!")

if len(images) != 312:
    raise ValueError("Количество изображений отличается от 312!")

if len(annotations) != len(images):
    raise ValueError("Количество аннотаций не совпадает с количеством изображений!")

image_ids = [image["id"] for image in images]

if len(set(image_ids)) != len(image_ids):
    raise ValueError("Обнаружены повторяющиеся ID изображений!")

annotations_by_image = {}

for annotation in annotations:
    image_id = annotation["image_id"]

    if image_id not in image_ids:
        raise ValueError(f"Аннотация ссылается на неизвестное изображение: {image_id}")

    annotations_by_image.setdefault(image_id, []).append(annotation)

for image in images:
    image_id = image["id"]

    if len(annotations_by_image.get(image_id, [])) != 1:
        raise ValueError(
            f"Для изображения {image_id} найдено не ровно одно обозначение!"
        )

print("Проверка количества объектов пройдена!")

# ==========================================
# 3. РАЗДЕЛЕНИЕ TRAIN / VAL
# ==========================================

random.seed(RANDOM_SEED)

shuffled_images = images.copy()
random.shuffle(shuffled_images)

train_count = int(len(shuffled_images) * TRAIN_RATIO)

train_images = shuffled_images[:train_count]
val_images = shuffled_images[train_count:]

print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")

# ==========================================
# 4. СОЗДАНИЕ СТРУКТУРЫ ДАТАСЕТА
# ==========================================

for split in ("train", "val"):
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# ==========================================
# 5. КОНВЕРТАЦИЯ ПОЛИГОНОВ
# ==========================================


def convert_annotation(image, annotation):

    width = image["width"]
    height = image["height"]

    if width <= 0 or height <= 0:
        raise ValueError("Некорректные размеры изображения!")

    if annotation["category_id"] != CLASS_ID:
        raise ValueError("Обнаружен неизвестный класс!")

    segmentation = annotation["segmentation"]

    if len(segmentation) != 1:
        raise ValueError("Ожидался один полигон!")

    polygon = segmentation[0]

    if len(polygon) != 8:
        raise ValueError(
            f"Ожидалось 8 координат, получено {len(polygon)}"
        )

    normalized = []

    tolerance = 1e-6

    for index in range(0, len(polygon), 2):

        x = float(polygon[index])
        y = float(polygon[index + 1])

        # Проверяем, не вышли ли координаты
        # за границы изображения больше допустимой погрешности
        if (
            x < -tolerance or x > width + tolerance
            or y < -tolerance or y > height + tolerance
        ):
            raise ValueError(
                f"Координата выходит за границы изображения: ({x}, {y})"
            )

        # Устраняем микроскопические погрешности
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))

        # Нормализация координат
        x_norm = x / width
        y_norm = y / height

        normalized.extend([x_norm, y_norm])

    return " ".join(
        [str(CLASS_ID)] + [f"{coord:.8f}" for coord in normalized]
    )



# ==========================================
# 6. СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ И РАЗМЕТКИ
# ==========================================

def process_split(image_list, split):

    for image in image_list:

        image_id = image["id"]

        annotation = annotations_by_image[image_id][0]

        # Путь к исходной фотографии
        relative_path = Path(
            image["file_name"].replace("\\", "/")
        )

        source_image = BASE_DIR / relative_path

        if not source_image.is_file():
            raise FileNotFoundError(
                f"Не найдено изображение: {source_image}"
            )

        # Сохраняем имя фотографии
        filename = source_image.name

        destination_image = (
            OUTPUT_DIR / "images" / split / filename
        )

        shutil.copy2(source_image, destination_image)

        # Преобразуем полигон
        label = convert_annotation(image, annotation)

        # Создаем TXT-файл
        label_path = (
            OUTPUT_DIR / "labels" / split /
            f"{source_image.stem}.txt"
        )

        label_path.write_text(label + "\n", encoding="utf-8")


print("Обрабатываем обучающую выборку...")
process_split(train_images, "train")

print("Обрабатываем проверочную выборку...")
process_split(val_images, "val")

# ==========================================
# 7. СОЗДАНИЕ DATA.YAML
# ==========================================

yaml_content = """path: dataset
train: images/train
val: images/val

names:
  0: wagon_number
"""

(OUTPUT_DIR / "data.yaml").write_text(
    yaml_content, encoding="utf-8"
)

# ==========================================
# 8. ИТОГ
# ==========================================

print("\n================================")
print("КОНВЕРТАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
print("================================")

print(f"Всего фотографий: {len(images)}")
print(f"Обучающая выборка: {len(train_images)}")
print(f"Проверочная выборка: {len(val_images)}")

print(f"\nДатасет сохранен в: {OUTPUT_DIR}")
print("Класс: wagon_number")
print("ID класса: 0")
