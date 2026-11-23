
import cv2
import torch
import easyocr
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 1. Загружаем YOLO
model = YOLO(
    r"models\best.pt"
)

# 2. Загружаем OCR
reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available()
)

# 3. Открываем фотографию
image = cv2.imread("wag_test_OCR.jpg")

if image is None:
    raise FileNotFoundError("Не удалось открыть фотографию!")

# Создаём копию для рисования
annotated = image.copy()

# 4. Ищем номерные области
results = model.predict(
    source=image,
    conf=0.25,
    verbose=False
)

result = results[0]

# Считаем обнаруженные объекты
object_count = len(result.boxes)

print("Количество обнаруженных объектов :", object_count)

recognized_count = 0

if result.masks is not None:

    for i, polygon in enumerate(result.masks.xy):

        # Координаты контура
        x_min = max(0, int(polygon[:, 0].min()))
        y_min = max(0, int(polygon[:, 1].min()))
        x_max = min(image.shape[1], int(polygon[:, 0].max()))
        y_max = min(image.shape[0], int(polygon[:, 1].max()))

        # Вырезаем номерную область
        crop = image[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            continue

        # Увеличиваем изображение
        crop = cv2.resize(
            crop,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # 5. Распознаём цифры
        text_results = reader.readtext(
            crop,
            allowlist='0123456789',
            detail=1
        )

        # Собираем цифры слева направо
        text_results.sort(
            key=lambda item: min(point[0] for point in item[0])
        )

        number = ''.join(item[1] for item in text_results)
        
        if number:
            recognized_count += 1
            confidence = sum(item[2] for item in text_results) / len(text_results)
            label = f"{number} ({confidence:.0%})"
            
            print(f"\nОбъект №{i + 1}:")
            print("Распознанный номер:", number)
            print(f"Уверенность OCR: {confidence:.2%}")
        else:
            label = "Не распознано"
            print(f"\nОбъект №{i + 1}: цифры не распознаны")

        # 6. Рисуем контур YOLO
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(
            annotated,
            [points],
            True,
            (0, 255, 0),
            3
        )

        # Рисуем подпись
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        text_y = max(y_min - 10, text_height + 10)

        cv2.rectangle(
            annotated,
            (x_min, text_y - text_height - 8),
            (x_min + text_width + 8, text_y + baseline),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x_min + 4, text_y - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

# 7. Сохраняем результат
output_dir = Path("preview")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "wagon_result.jpg"

cv2.imwrite(str(output_path), annotated)

print("\nКоличество распознанных объектов:", recognized_count)
print("Результат сохранён:", output_path)
print("\nПроверка завершена!")

