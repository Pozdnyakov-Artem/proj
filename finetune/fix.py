import torch
import sys


def fix_checkpoint(input_path, output_path):
    """Конвертирует 'голый' state_dict в формат RF-DETR"""

    print(f"📥 Загрузка: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # Проверяем, уже ли в правильном формате
    if "model" in checkpoint:
        print("✅ Файл уже в правильном формате")
        return

    # Если это просто state_dict — оборачиваем
    if isinstance(checkpoint, dict) and any(k.endswith(".weight") or k.endswith(".bias") for k in checkpoint.keys()):
        print("🔧 Обнаружен 'голый' state_dict. Конвертируем...")
        fixed = {
            "model": checkpoint,
            "epoch": -1,
            "source": "converted"
        }
        torch.save(fixed, output_path)
        print(f"💾 Сохранено: {output_path}")
        return

    print("❌ Неизвестный формат чекпоинта")
    print(f"Ключи: {list(checkpoint.keys())[:10]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python fix_checkpoint.py input.pth output.pth")
        sys.exit(1)
    fix_checkpoint(sys.argv[1], sys.argv[2])