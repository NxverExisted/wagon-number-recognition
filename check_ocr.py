import torch
import easyocr

print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Видеокарта:", torch.cuda.get_device_name(0))

print("\nЗагружаем OCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available()
)

print("\nOCR успешно запущен!")