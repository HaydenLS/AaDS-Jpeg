from main import Jpeg
from PIL import Image
import matplotlib.pyplot as plt
import os

# Пути к папкам
path_test = "images/test"
path_encoded = "images/encoded"
path_decoded = "images/decoded"
path_graphs = "graph_results"

# Убедимся, что нужные папки существуют
os.makedirs(path_encoded, exist_ok=True)
os.makedirs(path_decoded, exist_ok=True)
os.makedirs(path_graphs, exist_ok=True)

# Исходные изображения и их форматы
images_names = ['lenna', 'city']
images_formats = ['', 'grayscale', 'bw', 'bw_dither']

# Уровни качества
qualities = [0, 20, 40, 60, 80, 100]

for name in images_names:
    for format in images_formats:
        base_name = f"{name}_{format}" if format else name
        full_input_path = f"{path_test}/{base_name}.png"

        decoded_images = []

        for quality in qualities:
            output_path = f"{path_encoded}/{base_name}_q{quality}.jpeg"
            result_path = f"{path_decoded}/{base_name}_q{quality}.png"

            # JPEG кодирование и декодирование
            print(f"[INFO] Кодирование {base_name} с качеством {quality}")
            jpeg = Jpeg()
            jpeg.encode(full_input_path, output_path, quality)
            jpeg.decode(output_path, result_path)

            # Загрузка декодированного изображения
            img = Image.open(result_path)
            decoded_images.append((quality, img))

        # Отображение изображений в сетке 2x3
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for idx, (quality, img) in enumerate(decoded_images):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f"Quality {quality}")
            ax.axis('off')

        plt.suptitle(f"JPEG-декодирование: {base_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        graph_path = f"{path_graphs}/images_{base_name}.png"
        plt.savefig(graph_path, dpi=300)
        plt.show()

        print(f"[DONE] График сохранён: {graph_path}\n")
