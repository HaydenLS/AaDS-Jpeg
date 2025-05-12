import numpy as np
from huffman_tables import *
import math
from dims import get_block_dimensions

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
    def __init__(self, data, table):
        self.data = data
        self.bit_pos = 0
        self.current_byte = 0
        self.current_byte_pos = 0

        self.reverse_table = {v: k for k, v in table.items()}

    def read_bits(self, n):
        result = 0
        for _ in range(n):
            if self.current_byte_pos == 0:
                self.current_byte = self.data[self.bit_pos]
                self.bit_pos += 1
                self.current_byte_pos = 8
            
            result <<= 1
            result |= (self.current_byte >> (self.current_byte_pos - 1)) & 1
            self.current_byte_pos -= 1
        
        return result

    def read_huffman_code(self, table):
        code = ""
        while True:
            bit = self.read_bits(1)
            code += str(bit)
            if code in self.reverse_table:
                return self.reverse_table[code]

class ZigZag:
    def __init__(self, n=8):
        self.indicies = self._zigzag_indices(n)

    def zigzag_scan(self, block): 
            return [block[i][j] for (i,j) in self.indicies]
    
    def inverse_zigzag(self, sequence, n=8):
        block = np.zeros((n, n))
        for k, (i, j) in enumerate(self.indicies):
            block[i][j] = sequence[k]
        return block

    # Метод зигзаг обхода (indicies)
    def _zigzag_indices(self, n=8):
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

# Определение категории для значения
def get_category(value):
    if value == 0:
        return 0
    abs_val = abs(value)
    return int(np.floor(np.log2(abs_val))) + 1

## Класс для работы с DC коэффициентами
class DC:
    def encode(self, dc_components, component_type='lum'):
        
        # Получаем разности коэффициентов
        dc_diff = [dc_components[0]]
        for i in range(1, len(dc_components)):
            dc_diff.append(dc_components[i] - dc_components[i-1])
        

        """Кодирование DC с выбором таблицы"""
        table = huffman_dc if component_type == 'lum' else huffman_dc_chrominance
        bitstream = BitStream()

        for diff in dc_diff:
            category = get_category(diff)
            code = table[category]
            bitstream.add_bits(code)
            
            if category != 0:
                if diff >= 0:
                    val = diff
                else:
                    val = (1 << category) + diff - 1
                bitstream.add_bits(f"{val:0{category}b}")
    
        return bitstream.get_bytes()
    
    # Декодирование
    def decode(self, dc_bytes, count, component_type='lum'):
        # Получаем таблицу Хаффмана
        table = huffman_dc if component_type == 'lum' else huffman_dc_chrominance

        bitreader = BitReader(dc_bytes, table)


        # Декодируем
        dc_values = []
        for i in range(count):
            category = bitreader.read_huffman_code(table)
            if category == 0:
                val = 0
            else:
                val = bitreader.read_bits(category)
                if val < (1 << (category - 1)):
                    val = val - (1 << category) + 1

            if len(dc_values) > 0:
                val += dc_values[-1]

            dc_values.append(val)
                
        return dc_values

    
## Класс для работы с AC коэффициентами
class AC:
    def encode(self, ac_coefs, type: str):
        bits = []
        for ac_block in ac_coefs:
            rle_encoded_ac = self._rle_encode(ac_block)
            bit_stream = self._encode_ac(rle_encoded_ac, type)
            bits.append(bit_stream)
        return bits


    def _rle_encode(self, ac):
        encoded = []
        zero_counter = 0
        for coeff in ac:
            if coeff == 0:
                zero_counter += 1
                if zero_counter == 16:
                    encoded.append((15, 0, 0))  # ZRL
                    zero_counter = 0
            else:
                category = get_category(coeff)
                encoded.append((zero_counter, category, coeff))
                zero_counter = 0
        encoded.append((0, 0, 0))  # EOB
        return encoded


    def _encode_ac(self, block, component_type: str):
        ac_table = huffman_ac if component_type == 'lum' else huffman_ac_chrominance
        bitstream = BitStream()

        for run, size, coeff in block:
            bitstream.add_bits(ac_table[(run, size)])  # Хаффман-код (run, size)

            if size != 0:
                if coeff >= 0:
                    bitstream.add_bits(f"{coeff:0{size}b}")
                else:
                    coeff_bits = (1 << size) + coeff - 1
                    bitstream.add_bits(f"{coeff_bits:0{size}b}")
        
        return bitstream.get_bytes()
    
    def decode(self, ac_bytes, component_type: str):
        
        decoded_blocks = []
        for block in ac_bytes:
            decoded_block = self._decode_block(block, component_type)
            decoded_blocks.append(decoded_block)
        return decoded_blocks


    def _decode_block(self, block, component_type: str):
        # Получаем таблицу Хаффмана
        table = huffman_ac if component_type == 'lum' else huffman_ac_chrominance

        ac_values = np.zeros(63, dtype=np.int16)
        ac_index = 0
        # Декодируем
        bitreader = BitReader(block, table)
        while True:
            run, size = bitreader.read_huffman_code(table)
            if run == 0 and size == 0:
                break

            coeff = 0
            if size != 0:
                coeff = bitreader.read_bits(size)
                if coeff < (1 << (size - 1)):
                    coeff = coeff - (1 << size) + 1
                
            # Получаем пару (run, size, coeff)
            ac_values[ac_index + run] = coeff
            ac_index += run+1



        return ac_values


orig = []

class Huffman:
    def encode(self, channel, type: str):
        
        if type == 'lum':
            global orig
            orig = channel

        y_h, y_w = channel.shape[0], channel.shape[1]
        dc_coefs = []
        ac_coefs = []

        # Зигзаг обход блоков
        zz = ZigZag()
        for i in range(y_h):
            for j in range(y_w):
                block = channel[i, j]
                z_block = zz.zigzag_scan(block)
                dc_coefs.append(z_block[0])
                ac_coefs.append(z_block[1:])

        # Кодирование DC и AC коэффициентов
        dc = DC()
        ac = AC()

        dc_bytes = dc.encode(dc_coefs, type)
        ac_bytes = ac.encode(ac_coefs, type)

        return dc_bytes, ac_bytes


    def decode(self, dc_bytes, ac_bytes, h, w, component_type='lum'):
        dc_decoder = DC()
        ac_decoder = AC()


        if component_type == 'lum':
            blocks_h, blocks_w = get_block_dimensions(h, w)[0]
        else:
            blocks_h, blocks_w = get_block_dimensions(h, w)[1]

        total_blocks = blocks_h * blocks_w
        # Декодирование DС и AC коэффициентов
        dc_vals = dc_decoder.decode(dc_bytes, total_blocks, component_type)
        ac_vals = ac_decoder.decode(ac_bytes, component_type)


        # Получаем массивы обратно.
        blocks = []
        zz = ZigZag()
        for dc, ac in zip(dc_vals, ac_vals):
            block = [0] + list(ac)
            block[0] = dc
            block = zz.inverse_zigzag(block)
            blocks.append(block)

        blocks = np.array(blocks)
        blocks = blocks.reshape((blocks_h, blocks_w, 8, 8))

        if component_type == 'lum':
            global orig
            print(np.array_equal(orig, blocks))

        return np.array(blocks)




if __name__ == "__main__":
    dc_test = [23, 22, 20, 19, 22, 30, 28, 2, -25, -18, -16, -16, -16, -10, -5, -1, 1, 2, 1, 2]
    dc_encoder = DC()
    dc_bytes = dc_encoder.encode(dc_test, 'lum')
    dc_decoder = DC()
    decoded = dc_decoder.decode(dc_bytes, len(dc_test), 'lum')
    print("Test:", dc_test, "->\n", decoded)