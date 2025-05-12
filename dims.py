import math
def get_block_dimensions(h, w):
        # Размеры в пикселях, округляем вверх до ближайшего кратного 8
        h_padded = math.ceil(h / 8) * 8
        w_padded = math.ceil(w / 8) * 8

        # Кол-во блоков для Y (без downsampling)
        y_blocks_h = h_padded // 8
        y_blocks_w = w_padded // 8

        # Для Cb и Cr даунсемплируем в 2 раза по высоте и ширине
        cbcr_blocks_h = (h_padded // 2) // 8
        cbcr_blocks_w = (w_padded // 2) // 8

        return (y_blocks_h, y_blocks_w), (cbcr_blocks_h, cbcr_blocks_w)