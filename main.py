# imports
import numpy as np
from PIL import Image
import struct

# Color Transforms
from color_transforms import ColorTransforms
# Huffman Tabels and quatizaton matrices
from huffman_tables import *

# DCT + Quantization
from dct_quantization import DCT_Quantization

# ZIGZAG + RLE + AC DC ECODING + HUFFMAN
from coder import Huffman, BitReader

cb_coded_global = []


class Jpeg:
    def __init__(self, input_file = None, output_file = None):
        self.input_file = input_file
        self.output_file = output_file

    def encode(self, file, output_file, quality):
        img_original = Image.open(file)
        img = np.array(img_original)
        h, w = img.shape[:2]

        # Color Transforms + Downsampling + деление на блоки
        color_transforms = ColorTransforms(img.shape)
        y, cb, cr = color_transforms.transform_forward(img)
        print(y.shape, cb.shape, cr.shape)
        
        # DCT + Квантование
        dct_quantization = DCT_Quantization(quality)
        y_dct, cb_dct, cr_dct = dct_quantization.dct2d(y, cb, cr)

        # Кодирование
        huffman_encoder = Huffman()
        y_dc, y_ac = huffman_encoder.encode(y_dct, 'lum')
        cb_dc, cb_ac = huffman_encoder.encode(cb_dct, 'chrom')
        cr_dc, cr_ac = huffman_encoder.encode(cr_dct, 'chrom')

        
        # Сохраняем всё в файл 
        with open(output_file, "wb") as f:
            # Header: размеры
            f.write(struct.pack("HH", h, w))  # 4 байта

            # Матрицы квантования
            f.write(dct_quantization.Q_Y_Quality.astype(np.uint8).tobytes())   # 64 байта
            f.write(dct_quantization.Q_C_Quality.astype(np.uint8).tobytes())   # 64 байта

            # Длины и данные Huffman-кодированных блоков
            def write_bytes(data):
                if isinstance(data, list) and isinstance(data[0], (bytes, bytearray)):
                    # Это список массивов байт (по блокам)
                    f.write(struct.pack("I", len(data)))  # количество блоков
                    for block in data:
                        f.write(struct.pack("I", len(block)))  # длина блока
                        f.write(block)
                elif isinstance(data, (bytes, bytearray)):
                    f.write(struct.pack("I", 1))  # один блок
                    f.write(struct.pack("I", len(data)))
                    f.write(data)
            
            write_bytes(y_dc)
            write_bytes(y_ac)
            write_bytes(cb_dc)
            write_bytes(cb_ac)
            write_bytes(cr_dc)
            write_bytes(cr_ac)


    def decode(self, file):
        with open(file, "rb") as f:
            h, w = struct.unpack("HH", f.read(4))

            q_y = np.frombuffer(f.read(64), dtype=np.uint8).reshape((8, 8))
            q_c = np.frombuffer(f.read(64), dtype=np.uint8).reshape((8, 8))

            def read_bytes():
                count = struct.unpack("I", f.read(4))[0]
                blocks = []

                for _ in range(count):
                    size = struct.unpack("I", f.read(4))[0]
                    blocks.append(f.read(size))
                    
                if count == 1: # dc
                    return blocks[0]
                else: # ac
                    return blocks

            y_dc = read_bytes()
            y_ac= read_bytes()
            cb_dc = read_bytes()
            cb_ac = read_bytes()
            cr_dc = read_bytes()
            cr_ac = read_bytes()


        # Декодирование
        huffman_decoder = Huffman()

        y_dct = huffman_decoder.decode(y_dc, y_ac, h, w, 'lum')
        cb_dct = huffman_decoder.decode(cb_dc, cb_ac, h, w, 'chrom')
        cr_dct = huffman_decoder.decode(cr_dc, cr_ac, h, w, 'chrom')
        
        

        # DCT Inverse
        dct_quant = DCT_Quantization(quality=100)  # используется только для IDCT
        dct_quant.q_y = q_y
        dct_quant.q_c = q_c
        # Преобразуем
        y_img = dct_quant.idct2d_channel(y_dct, q_y)
        cb_img = dct_quant.idct2d_channel(cb_dct, q_c)
        cr_img = dct_quant.idct2d_channel(cr_dct, q_c)

        # Обратное преобразование цвета
        ct = ColorTransforms((h, w))
        print(y_img.shape, cb_img.shape, cr_img.shape)
        img_recon = ct.transform_backward((y_img, cr_img, cb_img), (h, w))


        Image.fromarray(np.clip(img_recon, 0, 255).astype(np.uint8)).save("images/results/decoded.png")





# main
if __name__ == "__main__":
    filename_in = "images/test/again.png"
    ilename_out = "images/output/again.myjpeg"
    jpeg = Jpeg()
    jpeg.encode(filename_in, ilename_out, 100)
    print("Файл успешно закодирован")

    jpeg = Jpeg()
    jpeg.decode(ilename_out)
    print("Файл успешно декодирован")


    