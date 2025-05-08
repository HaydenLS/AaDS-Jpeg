import os
import numpy as np
from PIL import Image, ImageOps

def convert_png_to_raw(
    input_dir: str,
    output_dir: str,
    mode: str = "color",
    raw_format: str = ".raw"
) -> None:
    """
    Конвертирует PNG изображения из input_dir в собственный RAW формат и сохраняет в output_dir.
    
    Параметры:
        input_dir (str): Путь к папке с PNG изображениями
        output_dir (str): Путь для сохранения RAW изображений
        mode (str): Режим преобразования:
            - "color": обычное цветное изображение
            - "grayscale": в оттенках серого
            - "bw": чёрно-белое без дизеринга
            - "bw_dither": чёрно-белое с дизерингом
        raw_format (str): Расширение для RAW файлов (по умолчанию ".raw")
    """
    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список PNG файлов
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    for png_file in png_files:
        # Открываем изображение
        img_path = os.path.join(input_dir, png_file)
        img = Image.open(img_path)
        
        # Применяем выбранное преобразование
        if mode == "grayscale":
            img = ImageOps.grayscale(img)
        elif mode == "bw":
            img = ImageOps.grayscale(img)
            img = img.point(lambda x: 255 if x > 127 else 0, '1')
        elif mode == "bw_dither":
            img = ImageOps.grayscale(img)
            img = img.convert('1')  # Автоматически применяет дизеринг Флойда-Стейнберга
        
        # Преобразуем в numpy array
        img_array = np.array(img)
        
        # Генерируем имя выходного файла
        base_name = os.path.splitext(png_file)[0]
        output_path = os.path.join(output_dir, base_name + raw_format)
        
        # Сохраняем в RAW формате (простая бинарная запись массива)
        with open(output_path, 'wb') as f:
            f.write(img_array.tobytes())
        
        # Дополнительно сохраняем метаданные о формате
        meta_path = os.path.join(output_dir, base_name + ".meta")
        with open(meta_path, 'w') as f:
            f.write(f"width:{img_array.shape[1]}\n")
            f.write(f"height:{img_array.shape[0]}\n")
            f.write(f"channels:{1 if len(img_array.shape) == 2 else img_array.shape[2]}\n")
            f.write(f"dtype:{img_array.dtype}\n")
            f.write(f"mode:{mode}\n")


### "grayscale", "bw", "bw_dither"
def convert_png_to_special_modes(input_dir: str, output_dir: str, mode: str) -> None:
    """
    Преобразует PNG изображения в специальные режимы и сохраняет как PNG.

    Параметры:
        input_dir (str): Путь к папке с исходными PNG изображениями
        output_dir (str): Путь для сохранения результатов
        mode (str): Режим преобразования:
            - "grayscale" - оттенки серого
            - "bw" - черно-белое без дизеринга (пороговое)
            - "bw_dither" - черно-белое с дизерингом
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    for png_file in png_files:
        # Открываем изображение
        img_path = os.path.join(input_dir, png_file)
        img = Image.open(img_path)
        
        # Применяем выбранное преобразование
        if mode == "grayscale":
            converted_img = ImageOps.grayscale(img)
        elif mode == "bw":
            # Конвертируем в grayscale, затем применяем пороговое преобразование
            gray_img = ImageOps.grayscale(img)
            converted_img = gray_img.point(lambda x: 255 if x > 127 else 0, '1')
        elif mode == "bw_dither":
            # Конвертируем в grayscale, затем применяем дизеринг
            gray_img = ImageOps.grayscale(img)
            converted_img = gray_img.convert('1')  # Автоматически применяет дизеринг Флойда-Стейнберга
        
        # Сохраняем результат
        output_path = os.path.join(output_dir, f"{png_file.split('.')[0]}_{mode}.png")
        converted_img.save(output_path)



if __name__ == "__main__":
    # # Пример использования
    # input_directory = "input_images"
    # output_directory = "raw_output"
    
    # "color", "grayscale", "bw", "bw_dither"
    # conversion_mode = "grayscale"
    
    # convert_png_to_raw(input_directory, output_directory, mode=conversion_mode)

    for mode in ["grayscale", "bw", "bw_dither"]:
        convert_png_to_special_modes("original", "test", mode)
        
    print("Изображения созданы успешно.")