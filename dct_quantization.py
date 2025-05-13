import numpy as np

# Матрицы квантования
Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


class DCT_Quantization:
    def __init__(self, quality):
        self.C = self.dct_matrix()
        self.quality = quality

        self.Q_Y_Quality = self.get_quantization_matrix(Q_Y)
        self.Q_C_Quality = self.get_quantization_matrix(Q_C)
    
    def dct2d(self, y, cb, cr):
        """
        Применияет преобразование dct + квантование
        """
        y_dct = self.dct2d_channel(y, self.Q_Y_Quality)

        
        cb_dct = self.dct2d_channel(cb, self.Q_C_Quality)
        cr_dct = self.dct2d_channel(cr, self.Q_C_Quality)

        return y_dct, cr_dct, cb_dct
    
    def idct2d(self, y_dct, cb_dct, cr_dct):
        """
        Применияет обратное преобразование dct + деквантование
        """
        y_idct = self.idct2d_channel(y_dct, self.Q_Y_Quality)

        cb_idct = self.idct2d_channel(cb_dct, self.Q_C_Quality)
        cr_idct = self.dct2d_channel(cr_dct, self.Q_C_Quality)

        return y_idct, cb_idct, cr_idct

    
    def dct_matrix(self, N=8):
        C = np.zeros((N, N))
        for u in range(N):
            for x in range(N):
                if u == 0:
                    C[u, x] = np.sqrt(1/N)
                else:
                    C[u, x] = np.sqrt(2/N) * np.cos((2*x + 1) * u * np.pi / (2*N))
        return C
    
    def get_quantization_matrix(self, base_matrix):
        if self.quality <= 0:
            scale = 5000
        elif self.quality < 50:
            scale = 5000 / self.quality
        else:
            scale = 200 - self.quality * 2

        scaled_matrix = np.clip(np.round(base_matrix * scale / 100), 1, 255)
        return scaled_matrix

    def quantize(self, dct_coeffs, q_matrix):
        return np.round(dct_coeffs / q_matrix).astype(np.int32)

    def dequantize(self, quantized_coeffs, q_matrix):
        return quantized_coeffs * q_matrix

    

    # Функция dct2d для работы с каналами.
    def dct2d_channel(self, channel, q_matrix):
        block_h, block_w = channel.shape[0], channel.shape[1]
    
        dct = []
        for i in range(block_h):
            dct_row = []
            for j in range(block_w):
                block = channel[i, j]
                # Центрирование
                block = block.astype(np.float32) - 128
                # Матричное умножение
                coef = self.C @ block @ self.C.T

                # квантование
                quantized = self.quantize(coef, q_matrix)

                # Добавляем блок в матрицу блоков
                dct_row.append(quantized)

            dct.append(dct_row)
        
        return np.array(dct)

    def idct2d_channel(self, block_dct_coefs, q_matrix):
        """
        Обратное квантование + дкт
        """

        block_h, block_w = block_dct_coefs.shape[0], block_dct_coefs.shape[1]

        channel = []
        for i in range(block_h):
            channel_row = []
            for j in range(block_w):
                # Берем блок коэффициентов
                dct_coeffs = block_dct_coefs[i, j]

                # Обратное квантование
                dequantized = self.dequantize(dct_coeffs, q_matrix)

                # Рассчитываем сам блок
                block = self.C.T @ dequantized @ self.C
                # Добавляем 128 и преобразовываем в int
                channel_block = np.clip(block + 128, 0, 255).astype(np.uint8)
                # Добавляем обратно в матрицу
                channel_row.append(channel_block)

            channel.append(channel_row)

        return np.array(channel)

