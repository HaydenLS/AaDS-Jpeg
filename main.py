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

class Jpeg:
    def __init__(self, input_file = None, output_file = None):
        self.input_file = input_file
        self.output_file = output_file

    def encode(self, file, quality):
        img_original = Image.open(file)
        img = np.array(img_original)
        h, w = img.shape[:2]

        # Color Transforms + Downsampling + блокировка
        color_transforms = ColorTransforms(img.shape)
        y, cb, cr = color_transforms.transform_forward(img)

        # DCT + Квантование
        dct_quantization = DCT_Quantization(quality)
        y_dct, cb_dct, cr_dct = dct_quantization.dct2d(y, cb, cr)

        # Кодирование
        huffman_encoder = Huffman()
        y_dc, y_ac = huffman_encoder.encode(y_dct, 'lum')
        cb_dc, cb_ac = huffman_encoder.encode(cb_dct, 'chrom')
        cr_dc, cr_ac = huffman_encoder.encode(cr_dct, 'chrom')

        # Сохраняем всё в файл 
        with open("images/output/question.myjpeg", "wb") as f:
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
        from coder import DC, AC, inverse_zigzag
        from dct_quantization import DCT_Quantization
        from color_transforms import ColorTransforms

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
                return blocks if count > 1 else blocks[0]

            y_dc = read_bytes()
            y_ac = read_bytes()
            cb_dc = read_bytes()
            cb_ac = read_bytes()
            cr_dc = read_bytes()
            cr_ac = read_bytes()

        blocks_h = h // 8
        blocks_w = w // 8

        # Декодирование
        dc_decoder = DC()
        ac_decoder = AC()

        y_dc_vals = dc_decoder.decode(BitReader(y_dc), blocks_h * blocks_w, 'lum')
        y_ac_vals = ac_decoder.decode(BitReader(y_ac), blocks_h * blocks_w, 'lum')
        cb_dc_vals = dc_decoder.decode(BitReader(cb_dc), blocks_h * blocks_w // 4, 'chrom')
        cb_ac_vals = ac_decoder.decode(BitReader(cb_ac), blocks_h * blocks_w // 4, 'chrom')
        cr_dc_vals = dc_decoder.decode(BitReader(cr_dc), blocks_h * blocks_w // 4, 'chrom')
        cr_ac_vals = ac_decoder.decode(BitReader(cr_ac), blocks_h * blocks_w // 4, 'chrom')

        def build_channel(dc_vals, ac_vals, blocks_h, blocks_w):
            channel = np.zeros((blocks_h, blocks_w, 8, 8))
            for i in range(blocks_h):
                for j in range(blocks_w):
                    idx = i * blocks_w + j
                    zz = [dc_vals[idx]] + ac_vals[idx]
                    channel[i, j] = inverse_zigzag(zz)
            return channel

        y_dct = build_channel(y_dc_vals, y_ac_vals, blocks_h, blocks_w)
        cb_dct = build_channel(cb_dc_vals, cb_ac_vals, blocks_h//2, blocks_w//2)
        cr_dct = build_channel(cr_dc_vals, cr_ac_vals, blocks_h//2, blocks_w//2)

        # DCT Inverse
        dct_quant = DCT_Quantization(quality=50)  # используется только для IDCT
        dct_quant.q_y = q_y
        dct_quant.q_c = q_c
        y_img, cb_img, cr_img = dct_quant.idct2d(y_dct, cb_dct, cr_dct)

        # Обратное преобразование цвета
        ct = ColorTransforms((h, w, 3))
        img_recon = ct.transform_backward((y_img, cb_img, cr_img))


        Image.fromarray(np.clip(img_recon, 0, 255).astype(np.uint8)).save("images/result/decoded.png")





# main
if __name__ == "__main__":
    filename = "images/test/question.png"
    jpeg = Jpeg()
    jpeg.encode(filename, 80)
    print("Файл успешно закодирован")

    filename = "images/output/question.myjpeg"
    jpeg = Jpeg()
    jpeg.decode(filename)
    print("Файл успешно декодирован")


    