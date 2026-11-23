
import cv2
import random
from pathlib import Path

# ==========================================
# НАСТРОЙКИ
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

IMAGES_DIR = DATASET_DIR / "images" / "train"
LABELS_DIR = DATASET_DIR / "labels" / "train"

OUTPUT_DIR = DATASET_DIR / "preview"

NUM_IMAGES = 10
RANDOM_SEED = 42

# ==========================================
# 1. ПОДГОТОВКА
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    images.extend(IMAGES_DIR.glob(extension))

if not images:
    raise FileNotFoundError("В папке train не найдены изображения!")

random.seed(RANDOM_SEED)
selected_images = random.sample(
    images, min(NUM_IMAGES, len(images))
)

print(f"Всего изображений: {len(images)}")
print(f"Проверяем: {len(selected_images)}")

# ==========================================
# 2. ВИЗУАЛИЗАЦИЯ
# ==========================================

for image_path in selected_images:

    label_path = LABELS_DIR / (image_path.stem + ".txt")

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Не найдена разметка: {label_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Не удалось открыть {image_path}")

    height, width = image.shape[:2]

    lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    for line in lines:

        values = list(map(float, line.split()))

        if len(values) < 7 or len(values) % 2 == 0:
            raise ValueError(
                f"Некорректная строка разметки: {label_path}"
            )

        class_id = int(values[0])
        coordinates = values[1:]

        if class_id != 0:
            raise ValueError("Обнаружен неизвестный класс!")

        points = []

        for i in range(0, len(coordinates), 2):

            x_norm = coordinates[i]
            y_norm = coordinates[i + 1]

            if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                raise ValueError("Координаты вне диапазона 0–1!")

            x = round(x_norm * (width - 1))
            y = round(y_norm * (height - 1))

            points.append([x, y])

        polygon = points

        # Рисуем контур
        cv2.polylines(
            image,
            [__import__("numpy").array(polygon, dtype="int32")],
            isClosed=True,
            color=(0, 255, 0),
            thickness=max(2, width // 600)
        )

        # Отмечаем вершины
        for point in polygon:
            cv2.circle(
                image,
                tuple(point),
                radius=max(4, width // 300),
                color=(0, 0, 255),
                thickness=-1
            )

        cv2.putText(
            image,
            "wagon_number",
            (polygon[0][0], max(30, polygon[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), image)

    print(f"Сохранено: {output_path.name}")

print("\nВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
print(f"Результаты находятся в: {OUTPUT_DIR}")
