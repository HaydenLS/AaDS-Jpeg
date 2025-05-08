import numpy as np
from huffman_tables import *

# Класс для работы с битовой строкой
class BitStream:
    def __init__(self):
        self.bits = []
        self.current_byte = 0
        self.bit_pos = 7
    
    def add_bits(self, bit_str):
        for bit in bit_str:
            if bit == '1':
                self.current_byte |= (1 << self.bit_pos)
            self.bit_pos -= 1
            
            if self.bit_pos < 0:
                self.bits.append(self.current_byte)
                self.current_byte = 0
                self.bit_pos = 7
    
    def get_bytes(self):
        # Добавляем неполный байт (если есть)
        if self.bit_pos != 7:
            self.bits.append(self.current_byte)
        return bytes(self.bits)
class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            raise EOFError("End of bitstream reached")
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_huffman(self, table):
        code = ""
        while True:
            bit = str(self.read_bit())
            code += bit
            for k, (huff_code, _) in table.items():
                if huff_code == code:
                    return k


# Определение категории для значения
def get_category(value):
    if value == 0:
        return 0
    abs_val = abs(value)
    return int(np.floor(np.log2(abs_val))) + 1


## Класс для работы с DC коэффициентами
class DC:
    def encode(self, dc_components, component_type='lum'):
        dc_diff = self.encode_dc_coefficients(dc_components)

        """Кодирование DC с выбором таблицы"""
        table = huffman_dc if component_type == 'lum' else huffman_dc_chrominance
        bitstream = BitStream()

        for diff in dc_diff:
            category = get_category(diff)
            bitstream.add_bits(table[category][0])  # Добавляем код Хаффмана
            
            if category != 0:
                # Кодирование значения (дополнительный код)
                val = diff if diff > 0 else (abs(diff) ^ ((1 << category) - 1))
                bitstream.add_bits(f"{val:0{category}b}")  # Добавляем значение

        return bitstream.get_bytes()
    
    # Разностное кодирование DC коэффициентов
    def encode_dc_coefficients(self, dc_components):
        differences = [dc_components[0]]  # Первый DC сохраняем как есть
        for i in range(1, len(dc_components)):
            differences.append(dc_components[i] - dc_components[i-1])
        return differences
    
    # Декодирование
    def decode(self, bitreader, count, component_type='lum'):
        table = huffman_dc if component_type == 'lum' else huffman_dc_chrominance
        dc = []
        prev = 0
        for _ in range(count):
            category = bitreader.read_huffman(table)
            if category == 0:
                diff = 0
            else:
                bits = bitreader.read_bits(category)
                # Преобразование из дополнительного кода
                if bits < (1 << (category - 1)):
                    bits -= (1 << category) - 1
                diff = bits
            value = prev + diff
            dc.append(value)
            prev = value
        return dc

    
## Класс для работы с AC коэффициентами
class AC:
    def encode(self, ac_coefs, type: str):
        bits = []
        for ac_block in ac_coefs:
            rle_encoded_ac = self.rle_encode(ac_block)
            bit_stream = self.encode_ac_rle_str(rle_encoded_ac, type)
            bits.append(bit_stream)

        return bits


    def rle_encode(self, ac):
        n = len(ac)
        encoded = []
        zero_counter = 0
        
        for i in range(n):
            if ac[i] == 0:
                zero_counter+=1
                if zero_counter == 16:
                    encoded.append((15, 0)) # ZRL
                    zero_counter = 0
            else:
                category = get_category(ac[i])
                encoded.append((zero_counter, category))
                encoded.append(ac[i])  # Сохраняем значение
                zero_counter = 0

        encoded.append((0,0))  # EOB
        return encoded

    def encode_ac_rle_str(self, block, component_type: str):
        """RLE + Кодирование AC коэффициентов"""
        ac_table = huffman_ac if component_type == 'lum' else huffman_ac_chrominance
        bitstream = BitStream()
            
        for item in block:
            if isinstance(item, tuple):  # (Run, Size)
                bitstream.add_bits(ac_table[item][0])
            else:  # Значение коэффициента
                coeff = item
                category = get_category(coeff)
                if coeff > 0:
                    bitstream.add_bits(f"{coeff:0{category}b}")
                else:
                    bitstream.add_bits(f"{(abs(coeff)^((1<<category)-1)):0{category}b}")
            
        return bitstream.get_bytes()
    
    # Декодирование
    def decode(self, bitreader, count, component_type='lum'):
        table = huffman_ac if component_type == 'lum' else huffman_ac_chrominance
        blocks = []
        for _ in range(count):
            ac = []
            k = 0
            while k < 63:
                run_size = bitreader.read_huffman(table)
                run, size = run_size
                if run == 0 and size == 0:
                    # EOB
                    while len(ac) < 63:
                        ac.append(0)
                    break
                elif run == 15 and size == 0:
                    # ZRL
                    ac.extend([0]*16)
                    k += 16
                    continue
                else:
                    ac.extend([0]*run)
                    val_bits = bitreader.read_bits(size)
                    if val_bits < (1 << (size - 1)):
                        val_bits -= (1 << size) - 1
                    ac.append(val_bits)
                    k += run + 1
            blocks.append(ac)
        return blocks

# Метод зигзаг обхода
def zigzag_indices(n=8):
    indices = []
    for d in range(2 * n - 1):  # Все диагонали
        if d < n:
            for i in range(d + 1):
                j = d - i
                if d % 2 == 0:
                    indices.append((i, j))
                else:
                    indices.append((j, i))
        else:
            for i in range(d - n + 1, n):
                j = d - i
                if d % 2 == 0:
                    indices.append((i, j))
                else:
                    indices.append((j, i))
    return indices

def zigzag_scan(block):
    indices = zigzag_indices(len(block))
    return [block[i][j] for (i,j) in indices]

def inverse_zigzag(sequence, n=8):
    block = np.zeros((n, n))
    indices = zigzag_indices(n)
    for k, (i, j) in enumerate(indices):
        block[i][j] = sequence[k]
    return block


class Huffman:
    def encode(self, channel, type: str):
        """
        Кодирование AC и DC коэффициентов одного канала. (y/cb/cr)
        """
        
        y_h, y_w = channel.shape[0], channel.shape[1]

        dc_coefs = []
        ac_coefs = []
        for i in range(y_h):
            for j in range(y_w):
                block = channel[i, j]
                # 1. Zig Zag
                z_block = zigzag_scan(block)

                # Add to arrays
                dc_coefs.append(z_block[0])
                ac_coefs.append(z_block[1:])
        

        # 4. Кодирование Хаффманом
        dc = DC(); ac = AC()
        dc_bytes = dc.encode(dc_coefs, type)
        ac_bytes = ac.encode(ac_coefs, type)
        
        return dc_bytes, ac_bytes