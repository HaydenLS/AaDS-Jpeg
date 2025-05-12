import numpy as np
from dims import get_block_dimensions

class ColorTransforms:
    def __init__(self, img_shape=None):
        self.img_shape = img_shape
    
    def transform_forward(self, img):
        """
        RGB->YCBCR->DOWNSAMPLE->BLOCKS
        return: (y, cb, cr): tuple
        """
        # Перевод в YcBcR
        ycbcr = RGB2YCBCR(img)

        # Получим каналы
        y, cb, cr = ycbcr.transpose(2, 0, 1)
        # Даунсемплинг цветовых каналов.
        cb_ds = downsample(cb)
        cr_ds = downsample(cr)
        
        # Разбиение каналов на блоки 8x8
        
        y_blocks = block_partioning(y)
        cb_blocks = block_partioning(cb_ds)
        cr_blocks = block_partioning(cr_ds)

        return y_blocks, cb_blocks, cr_blocks

    def transform_backward(self, ycbcr_tuple, shape=None):
        self.img_shape = shape
        h, w = self.img_shape[:2]

        y_blocks, cb_blocks, cr_blocks = ycbcr_tuple

        # Собираем каналы из блоков
        y = merge_blocks(y_blocks, (h, w))

        # Для cb/cr сначала восстанавливаем без обрезки
        padded_h_cbcr = cb_blocks.shape[0] * 8
        padded_w_cbcr = cb_blocks.shape[1] * 8

        cb = merge_blocks(cb_blocks, (padded_h_cbcr, padded_w_cbcr))
        cr = merge_blocks(cr_blocks, (padded_h_cbcr, padded_w_cbcr))

        # Апсемплируем
        cb_up = bilinear_upsample(cb)
        cr_up = bilinear_upsample(cr)

        # Теперь обрезаем до точного размера Y
        cb_up = cb_up[:y.shape[0], :y.shape[1]]
        cr_up = cr_up[:y.shape[0], :y.shape[1]]

        print(f"img shape: y {y.shape}, cb {cb_up.shape}, cr {cr_up.shape}")

        ycbcr = np.stack((y, cb_up, cr_up), axis=-1)
        rgb = YCBCR2RGB(ycbcr)

        return rgb



def RGB2YCBCR(img):
    """
    Переводит изображение в формат YCbCr
    retrun: [H, W, C] матрицу
    """
    img = img.astype(np.float32)
    # take channels
    tr = img.transpose(2, 0, 1)
    # take r g b
    r, g, b = tr
        
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        
    ycbcr = np.dstack((y, cb, cr))
    return np.clip(ycbcr, 0, 255).astype(np.uint8)

def YCBCR2RGB(img):
    """
    Переводит изображение в формат RGB
    retrun: [H, W, C] матрицу
    """
    img = img.astype(np.float32)
    # get components
    tr = img.transpose(2, 0, 1)
    # take r g b + normalize
    y, cb, cr = tr

    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)

    rgb = np.dstack((r, g, b))
    return np.clip(rgb, 0, 255).astype(np.uint8)

def downsample(img_channel, factor: int = 2):
    h, w = img_channel.shape
    # Вычисляем новые размеры (округляя вниз)
    new_h = h // factor
    new_w = w // factor
    
    # Создаём массив для уменьшенного канала
    downsampled = np.zeros((new_h, new_w), dtype=np.float32)
    
    # Усредняем блоки factor x factor
    for i in range(new_h):
        for j in range(new_w):
            block = img_channel[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            downsampled[i,j] = np.mean(block)
    
    return downsampled.astype(np.uint8)

def bilinear_upsample(img_channel, factor: int = 2):
    h, w = img_channel.shape
    new_h, new_w = h * factor, w * factor
    
    # Создаём координатные сетки
    x = np.linspace(0, w-1, new_w)
    y = np.linspace(0, h-1, new_h)
    
    # Билинейная интерполяция
    upsampled = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x0, y0 = int(np.floor(x[j])), int(np.floor(y[i]))
            x1, y1 = min(x0 + 1, w-1), min(y0 + 1, h-1)
            
            # Доли для интерполяции
            x_alpha = x[j] - x0
            y_alpha = y[i] - y0
            
            # Интерполяция по 4 точкам
            val = (1-x_alpha)*(1-y_alpha)*img_channel[y0,x0] + \
                  x_alpha*(1-y_alpha)*img_channel[y0,x1] + \
                  (1-x_alpha)*y_alpha*img_channel[y1,x0] + \
                  x_alpha*y_alpha*img_channel[y1,x1]
            
            upsampled[i,j] = np.clip(val, 0, 255)
    
    return upsampled


def block_partioning(img, size: int = 8, fill: int = 0):
    """
    return: 5d массив из блоков (h_blocks, w_blocks, h_block, w_block, channels)
    """

    h, w = img.shape[0], img.shape[1]

    # Вычисляем количество блоков
    blocks_h = (h + size - 1) // size
    blocks_w = (w + size - 1) // size

    # Дополняем изображение
    padded_h = blocks_h * size
    padded_w = blocks_w * size

    if img.ndim == 2:
        padded = np.full((padded_h, padded_w), fill, dtype=img.dtype)
        padded[:h, :w] = img
    else:  # 3D (цветное)
        padded = np.full((padded_h, padded_w, img.shape[2]), fill, dtype=img.dtype)
        padded[:h, :w, :] = img

    blocks = []
    # Разбиение на блоки
    for i in range(blocks_h):
        row_blocks = []
        for j in range(blocks_w):
            block = padded[i*size:(i+1)*size, j*size:(j+1)*size]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return np.array(blocks)


def merge_blocks(blocks, original_shape: int):

    num_blocks_h, num_blocks_w, size = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    h, w = original_shape[0], original_shape[1]

    # Создаём пустое изображение с дополнением
    padded_h = num_blocks_h * size
    padded_w = num_blocks_w * size


    if blocks.ndim == 5:  # Цветное
        merged = np.zeros((padded_h, padded_w, blocks.shape[4]), dtype=blocks.dtype)
    else:  # (H, W)
        merged = np.zeros((padded_h, padded_w), dtype=blocks.dtype)
    
    # Заполняем блоки
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            merged[i*size:(i+1)*size, 
                  j*size:(j+1)*size] = blocks[i, j]
    
    return merged[:h, :w]