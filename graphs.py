from PIL import Image
import matplotlib.pyplot as plt
import os

path_test = "images/test"
images_names = ['lenna', 'city']
images_formats = ['grayscale', 'bw', 'bw_dither']

# Создаем фигуру: 2 строки (по одной для каждого изображения) и 3 столбца (по форматам)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
fig.tight_layout(pad=3.0)

for i, name in enumerate(images_names):  # строки (lenna, city)
    for j, format in enumerate(images_formats):  # столбцы (grayscale, bw, bw_dither)
        # Формируем путь к файлу
        img_path = os.path.join(path_test, f"{name}_{format}.png")
        
        # Загружаем изображение
        img = Image.open(img_path)
        
        # Отображаем изображение на соответствующей позиции
        ax = axes[i, j]
        ax.imshow(img, cmap='gray' if format != 'color' else None)
        ax.set_title(f"{name} ({format})")
        ax.axis('off')

plt.suptitle("Изображения в разных форматах", fontsize=16, y=1.02)
plt.show()