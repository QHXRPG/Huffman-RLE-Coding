# -*- coding: utf-8 -*-
import struct


def _counter(data: bytes):
    """
    计算每个字节的频率
    :param data:
    :return:
    """
    frequency = {}
    for byte in data:
        if byte in frequency:
            frequency[byte] += 1
        else:
            frequency[byte] = 1
    return frequency

def reshapeProto(lines: list) -> list:
    """
    重组proto信息，使其成为标准处理格式
    :param lines: 直接读取 proto 文件所产生的列表
    :return:
    """
    result = []
    tmp = ''
    for line in lines:
        line = line.strip()
        tmp = tmp + line
    for block in tmp.split('}'):
        block = block.strip()
        if block and '{' in block:
            block = block + '}'
            if block[0] == ';':
                result.append(block[1:])
            else:
                result.append(block)
    return result

def isSquareBracketsMatch(s: str) -> (bool, str, str, str):
    """
    检查该变量字符串的中括号是否匹配
    :param s: 变量字符串
    :return: (中括号是否匹配，变量类型，变量数量，变量名)
    """
    left_bracket_index = s.find('[')
    right_bracket_index = s.find(']')

    # 没有中括号
    if left_bracket_index == -1 and right_bracket_index == -1:
        return False, None, None, None

    # 缺少一个括号或多余括号或多个括号组
    if left_bracket_index == -1 or right_bracket_index == -1 or s.count('[') != 1 or s.count(']') != 1:
        return False, None, None, None

    # 检查中括号的顺序
    if left_bracket_index > right_bracket_index:
        return False, None, None, None

    # 提取中括号内的内容
    content = s[left_bracket_index + 1:right_bracket_index]

    # 中括号内部无字符
    if content == '':
        variable_quantity = -1
    # 中括号内部必须是正整数
    elif content.isdigit() and int(content) > 0:
        variable_quantity = int(content)
    else:
        return False, None, None, None

    # 提取变量类型和变量名
    variable_type = s[:left_bracket_index].strip()
    variable_name = s[right_bracket_index + 1:].strip()

    return True, variable_type, variable_quantity, variable_name


# 最小堆实现
class MinHeap:
    def __init__(self):
        self.heap = []

    # 计算给定索引节点的父节点索引
    @staticmethod
    def _parent(index):
        return (index - 1) // 2

    # 计算给定索引节点的左子节点索引
    @staticmethod
    def _left_child(index):
        return 2 * index + 1

    # 计算给定索引节点的右子节点索引
    @staticmethod
    def _right_child(index):
        return 2 * index + 2

    # 在堆中进行上浮操作
    def _heapify_up(self, index):
        while index > 0:
            parent = self._parent(index)
            if self.heap[index] < self.heap[parent]:
                # 交换当前节点和父节点
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else:
                break

    # 在堆中进行下沉操作
    def _heapify_down(self, index):
        size = len(self.heap)
        while self._left_child(index) < size:
            smallest = index
            left = self._left_child(index)
            right = self._right_child(index)

            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != index:
                # 交换当前节点和最小子节点
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

    # 向堆中插入一个新元素
    def insert(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    # 移除并返回堆中的最小元素
    def extract_min(self):
        if not self.heap:
            raise IndexError("extract_min from an empty heap")
        # 交换根节点和最后一个节点
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        min_value = self.heap.pop()  # 弹出最小值
        self._heapify_down(0)
        return min_value

    # 将一个无序数组转换为最小堆
    def heapify(self, array):
        self.heap = array[:]
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)



# 哈夫曼树节点
class HuffmanNode:
    def __init__(self, byte, freq):
        self.byte = byte    # 节点存储的字节值
        self.freq = freq    # 字节的频率
        self.left = None    # 左子节点
        self.right = None   # 右子节点

    # 定义比较运算符，使得优先队列可以根据频率排序
    def __lt__(self, other):
        return self.freq < other.freq

    def to_preorder_list(self):
        """
        将树转换为前序遍历的列表
        :return:  huffman树的前序列表
        """
        result = []
        result.append((self.byte, self.freq))  # 添加当前节点
        if self.left:  # 遍历左子树
            result.extend(self.left.to_preorder_list())
        if self.right:  # 遍历右子树
            result.extend(self.right.to_preorder_list())
        return result

    def to_bytes(self):
        """
        将前序遍历的列表转换为字节并返回 bytes 类型
        :return: huffman树的前序列表的 bytes
        """
        inorder_list = self.to_preorder_list()
        byte_stream = bytearray()
        zero_position = -1  # 用于记录字节为 0 的位置
        for i, (byte, freq) in enumerate(inorder_list):
            if byte == 0:
                zero_position = i
            byte_value = byte if byte is not None else 0
            byte_stream.extend(struct.pack('BI', byte_value, freq))

        root_bytes = bytes(byte_stream)
        zero_position_bytes = struct.pack('I', zero_position)          # 将 zero_position 转换为字节流
        return root_bytes, zero_position_bytes



class HuffmanCoding:
    def __init__(self):
        self.root: HuffmanNode = None       # huffman树的根节点
        self.codebook = {}                  # 编码本，解压缩过程将依靠编码本进行解码
        self.original_len = 0               # 二进制字符串的原始长度
        self.root_compress_padding = 0      # 压缩哈夫曼树时候是否进行了补0操作

    def build_huffman_tree(self, frequency):
        """
        创建哈夫曼树
        :param frequency: 频率表
        :return:  None
        """
        # 创建一个优先队列（小顶堆），每个元素是一个HuffmanNode
        heap = [HuffmanNode(byte, freq) for byte, freq in frequency.items()]
        minHeap = MinHeap()
        minHeap.heapify(heap)

        # 迭代合并频率最低的两个节点，直到只剩一个节点
        while len(minHeap.heap) > 1:
            left = minHeap.extract_min()    # 取出频率最低的节点
            right = minHeap.extract_min()   # 取出频率次低的节点

            # 合并两个节点为一个新节点，新节点的频率是两个节点频率之和
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            minHeap.insert(merged)  # 将新节点放回堆中

        # 返回哈夫曼树的根节点
        self.root = minHeap.extract_min()

    def build_codes(self, node, prefix=''):
        """
        构建哈夫曼编码表
        :param node: huffman 树节点
        :param prefix: 从 huffman 树根节点到某值的路径
        :return:
        """
        if node.byte is not None:
            self.codebook[node.byte] = prefix
        else:
            self.build_codes(node.left, prefix + '0')
            self.build_codes(node.right, prefix + '1')

    def compress(self, data: bytes) -> bytes:
        """
        使用哈夫曼编码对数据进行压缩
        :param data: 原始字节数据
        :return: 压缩后的字节数据
        """
        frequency = _counter(data)                                          # 统计每个字节的频率
        self.build_huffman_tree(frequency)                                  # 构建哈夫曼树
        self.build_codes(self.root)                                         # 构建哈夫曼编码表
        compressed_data = ''.join(self.codebook[byte] for byte in data)     # 将数据编码为哈夫曼二进制编码的字符串
        self.original_len = len(compressed_data)                            # 记录原始二进制数据长度（用于解压时截断补零）
        padding = (8 - len(compressed_data) % 8) % 8                        # 补齐二进制字符串长度到8的倍数
        compressed_data += '0' * padding

        # 将二进制字符串按每8位分割，转换为字节
        compressed_bytes = bytes(int(compressed_data[i:i + 8], 2) for i in range(0, len(compressed_data), 8))

        return compressed_bytes

    # 哈夫曼解码，解压
    def decompress(self, compressed_data) -> bytes:

        # 将字节转换回二进制字符串
        binary_string_back = ''.join(format(i, '08b') for i in compressed_data)
        compressed_data = binary_string_back[:self.original_len]   # 在压缩阶段加的0在这里消去
        decoded_bytes = []
        node = self.root
        for bit in compressed_data:
            if bit == '0':
                node = node.left
            else:
                node = node.right
            if node.byte is not None:           # 如果到达叶子节点
                decoded_bytes.append(node.byte)
                node = self.root                # 重置回根节点
        return bytes(decoded_bytes)


    @staticmethod
    def huffman_tree_compress(input_string: str, is_root = True) -> (str, int):
        """
        压缩huffman树的字节
        :param input_string: huffman树的 16 进制字符串
        :param is_root: 默认为 True
        :return: 被压缩的huffman树的 16 进制字符串，是否补零（0 或 1）
        """
        left , right, flag, coding_string = 0, 0, 0, ''
        while right < len(input_string):
            if input_string[right] == '0':
                flag += 1
                right += 1
                if flag == 15:
                    coding_string += '0f'
                    flag = 0
                    left = right
            else:
                if flag > 0:
                    if flag == 10: flag = 'a'
                    elif flag == 11: flag = 'b'
                    elif flag == 12: flag = 'c'
                    elif flag == 13: flag = 'd'
                    elif flag == 14: flag = 'e'
                    coding_string += '0' + str(flag)
                    flag = 0
                coding_string += input_string[right]
                right += 1
                left = right

        if flag > 0:
            if flag == 10:flag = 'a'
            elif flag == 11:flag = 'b'
            elif flag == 12:flag = 'c'
            elif flag == 13:flag = 'd'
            elif flag == 14:flag = 'e'
            coding_string += '0' + str(flag)
        if is_root:
            if len(coding_string) % 2 != 0:
                coding_string += '0'
                padding = 1
            else:
                padding = 0
        else:
            padding = 0
        return coding_string, padding

    @staticmethod
    def huffman_tree_decompress(coding_string, padding):
        """
        解压缩 huffman 树的字节
        :param coding_string: 被压缩的huffman树的 16 进制字符串
        :param padding: 是否补零（0 或 1）
        :return: 解压后的huffman树的 16 进制字符串
        """
        if padding == 1:
            coding_string = coding_string[:-padding]
        decompressed = ''
        i = 0

        while i < len(coding_string):
            if coding_string[i] == '0':  # 如果是 '0'，说明是编码部分
                # 获取编码的重复次数
                count = coding_string[i + 1]
                if count.isdigit():     # 如果是数字
                    count = int(count)
                else:                   # 如果是 'a' 到 'f'，转换为 10 到 15
                    count = ord(count) - ord('a') + 10

                # 展开 '0'
                decompressed += '0' * count
                i += 2                      # 跳过当前编码部分
            else:                           # 其他字符直接添加到结果中
                decompressed += coding_string[i]
                i += 1

        return decompressed

    def rebuild_huffman_tree(self, huffman_tree_bytes: bytes, zero_position: int):
        """
        重建 huffman 树
        :param huffman_tree_bytes: 发送端传来的 huffman 树信息
        :param zero_position:  在 huffman 树前序列表中值为 0 的 index
        :return: None
        """
        # 从字节流中解析前序列表
        preorder_list = self._bytes_to_preorder_list(huffman_tree_bytes, zero_position)
        # 重建霍夫曼树
        self.root, _ = self._rebuild_huffman_tree(preorder_list)

    @staticmethod
    def _bytes_to_preorder_list(huffman_tree_bytes: bytes, zero_position: int):
        preorder_list = []
        byte_size = struct.calcsize('BI')
        for i in range(0, len(huffman_tree_bytes), byte_size):
            byte, freq = struct.unpack('BI', huffman_tree_bytes[i:i + byte_size])
            # 在 zero_position 位置将 byte 设置为 0，而不是 None
            byte = 0 if i // byte_size == zero_position else (None if byte == 0 else byte)
            preorder_list.append((byte, freq))
        return preorder_list

    @staticmethod
    def _rebuild_huffman_tree(preorder_list: list, i: int = 0):
        if i >= len(preorder_list):
            return None, i

        root = HuffmanNode(preorder_list[i][0], preorder_list[i][1])

        if root.byte is not None:
            return root, i + 1

        root.left, next_index = HuffmanCoding._rebuild_huffman_tree(preorder_list, i + 1)
        root.right, next_index = HuffmanCoding._rebuild_huffman_tree(preorder_list, next_index)

        return root, next_index


class ProtoParser(object):
    def __init__(self):
        self.desc = {}
        self.huffman = None

    def buildDesc(self, filename: str) -> None:
        """
        从指定的文本文件filename中读取协议描述文本，存为python内部数据，以备解析使用
        :param filename:文件路径
        :return:None
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
        except IOError as e:
            print(f"文件读取错误: {e}")
        except Exception as e:
            print(f"文件读取遇到意外错误: {e}")
        lines = reshapeProto(lines)

        for line in lines:
            struct_begin = line.find('{')
            struct_end = line.find('}')
            current_struct = line[:struct_begin].strip()                # 结构名
            self.desc[current_struct] = []                              # 初始化该结构体的字段列表
            current_block = line[struct_begin + 1:struct_end].strip()   # 结构体
            fields = current_block.split(';')                           # 以分号分割每个变量
            for field in fields:
                if field == '':
                    continue
                else:
                    # 确保字段定义行至少有两个部分：字段类型和字段名称
                    parts = field.split()
                    field_tmp = isSquareBracketsMatch(field)

                    # 非数组变量
                    if len(parts) == 2 and not field_tmp[0]:
                        field_type = parts[0]
                        field_name = parts[1]
                        # 将字段类型和名称添加到当前结构体的字段列表中
                        self.desc[current_struct].append((field_type, 0, field_name))

                    # 数组变量
                    elif field_tmp[0]:
                        field_type = field_tmp[1]
                        field_nums = field_tmp[2]
                        field_name = field_tmp[3]
                        # 将字段类型和名称添加到当前结构体的字段列表中
                        self.desc[current_struct].append((field_type, field_nums, field_name))

    def loads(self, name: str, s: str) -> dict:
        """
        根据先前读取的描述文本，把以16进制表示的字符串s反序列化为字典对象返回
        :param name: 协议体名称
        :param s: 16进制表示的字符串
        :return: 字典对象
        """
        data = bytes.fromhex(s)                 # 将一个十六进制字符串转换为字节对象
        return self._deserialize(name, data)[0]    # 反序列化

    def dumps(self, name: str, d: dict) -> str:
        """
        根据先前读取的描述文本和名为name的协议体内容，把字典d的序列化结果以16进制表示的字符串返回
        :param name: 协议体名称
        :param d: 字典对象
        :return: 16进制表示的字符串
        """
        data = self._serialize(name, d)         # 序列化
        return data.hex()                       # 将字节对象转换为一个十六进制字符串

    def dumpComp(self, name: str, d: dict) -> bytes:
        """
        根据先前读取的描述文本和名为name的协议体内容，把字典d序列化并压缩，返回其结果
        :param name: 协议体名称
        :param d: 字典对象
        :return: 包含长度信息、哈夫曼树和压缩数据的字节流
        """
        serialized_data = self._serialize(name, d)              # 序列化数据

        # 使用哈夫曼编码进行压缩
        huffman = HuffmanCoding()
        compressed_data = huffman.compress(serialized_data)

        # 将 huffman 树的结构信息加入到压缩的数据中一并传给接收端
        root_bytes,  zero_position_bytes = huffman.root.to_bytes()
        root_bytes, padding = huffman.huffman_tree_compress(root_bytes.hex())               # 压缩 huffman 树
        root_bytes = bytes.fromhex(root_bytes)

        original_len = huffman.original_len
        original_len_bytes = struct.pack('<I', original_len)
        padding_byte = struct.pack('<B', padding)                           # padding

        # huffman 树的结构信息长度
        root_bytes_length = len(root_bytes) + len(zero_position_bytes) + len(padding_byte) + len(original_len_bytes)
        root_bytes_length = struct.pack('<I', root_bytes_length)
        result  = root_bytes_length + root_bytes + zero_position_bytes + padding_byte + original_len_bytes + compressed_data
        return result

    def loadComp(self, name: str, s: bytes) -> dict:
        """
        根据先前读取的描述文本，把dumpComp的输出结果恢复为原始对象
        :param name: 协议体名称
        :param s: 包含长度、padding信息、哈夫曼树、压缩数据等字节
        :return: 解压缩并反序列化后的字典对象
        """
        # 解析哈夫曼树的结构信息长度
        root_bytes_length = struct.unpack('<I', s[:4])[0]

        # 计算哈夫曼树结构信息的结束位置
        root_bytes_end = 4 + root_bytes_length

        # 解析哈夫曼树的结构信息（不包括零位置字节、填充字节和原始长度字节）
        root_bytes = s[4:root_bytes_end - 9]  # 9 = 4 (zero_position_bytes) + 1 (padding_byte) + 4 (original_len_bytes)

        # 解析零位置字节
        zero_position_bytes_start = root_bytes_end - 9
        zero_position_bytes = s[zero_position_bytes_start:zero_position_bytes_start + 4]
        zero_position = struct.unpack('<I', zero_position_bytes)[0]

        # 解析填充字节
        padding_byte_position = zero_position_bytes_start + 4
        padding = struct.unpack('<B', s[padding_byte_position:padding_byte_position + 1])[0]

        # 解析原始长度字节
        original_len_bytes_start = padding_byte_position + 1
        original_len_bytes = s[original_len_bytes_start:original_len_bytes_start + 4]
        original_len = struct.unpack('<I', original_len_bytes)[0]

        # 解析压缩数据
        compressed_data = s[root_bytes_end:]

        huffman = HuffmanCoding()

        # 解压缩 huffman 字节并重建huffman 树
        decompress_root_bytes = bytes.fromhex(huffman.huffman_tree_decompress(root_bytes.hex(), padding))
        huffman.rebuild_huffman_tree(decompress_root_bytes, zero_position)
        huffman.original_len = original_len
        decompressed_data = huffman.decompress(compressed_data)
        result, _ = self._deserialize(name, decompressed_data)      # 反序列化数据
        return result

    # 序列化
    def _serialize(self, name: str, d: dict) -> bytes:
        """
        序列化
        :param name: 协议体名称
        :param d: 字典对象
        :return: 字节流
        """
        result = bytearray()
        desc = self.desc[name]

        for field_type, field_nums, field_name in desc:
            value = d[field_name]

            # 非数组字段
            if field_nums == 0:
                result.extend(self._serialize_field(field_type, value))

            # 可变长度数组字段
            elif field_nums == -1:
                result.extend(struct.pack('<H', len(value)))        # 长度作为 uint16
                for item in value:
                    result.extend(self._serialize_field(field_type, item))

            # 固定长度数组字段
            else:
                for item in value:
                    result.extend(self._serialize_field(field_type, item))

        return bytes(result)

    def _serialize_field(self, field_type: str, value) -> bytes:

        # 字符串， 字符串长度+字符串utf-8编码
        if field_type == 'string':
            encoded_str = value.encode('utf-8')
            return struct.pack('<H', len(encoded_str)) + encoded_str
        elif field_type == 'int8':
            return struct.pack('<b', value)
        elif field_type == 'uint8':
            return struct.pack('<B', value)
        elif field_type == 'int16':
            return struct.pack('<h', value)
        elif field_type == 'uint16':
            return struct.pack('<H', value)
        elif field_type == 'int32':
            return struct.pack('<i', value)
        elif field_type == 'uint32':
            return struct.pack('<I', value)
        elif field_type == 'float':
            return struct.pack('<f', value)
        elif field_type == 'double':
            return struct.pack('<d', value)
        elif field_type == 'bool':
            return struct.pack('<?', value)
        else:  # 嵌套结构
            return self._serialize(field_type, value)

    # 反序列化，根据类型名和字节流，反序列化出dict
    def _deserialize(self, name: str, data: bytes) -> (dict, int):
        """
        反序列化，根据类型名和字节流，反序列化出dict
        :param name: 协议体名称
        :param data: 接收端收到的字节流
        :return: 还原后的字典对象
        """
        result = {}
        desc = self.desc[name]
        offset = 0

        for field_type, field_nums, field_name in desc:
            # 非数组字段
            if field_nums == 0:
                value, offset = self._deserialize_field(field_type, data, offset)
                result[field_name] = value

            # 可变长度数组字段
            elif field_nums == -1:
                length, offset = struct.unpack_from('<H', data, offset)[0], offset + 2
                array = []
                for _ in range(length):
                    item, offset = self._deserialize_field(field_type, data, offset)
                    array.append(item)
                result[field_name] = tuple(array)

            # 固定长度数组字段
            else:
                array = []
                for _ in range(field_nums):
                    item, offset = self._deserialize_field(field_type, data, offset)
                    array.append(item)
                result[field_name] = tuple(array)

        return result, offset

    def _deserialize_field(self, field_type: str, data: bytes, offset: int) -> (any, int):
        if field_type == 'string':
            length, offset = struct.unpack_from('<H', data, offset)[0], offset + 2
            value, offset = data[offset:offset + length].decode('utf-8'), offset + length
            return value, offset
        elif field_type == 'int8':
            return struct.unpack_from('<b', data, offset)[0], offset + 1
        elif field_type == 'uint8':
            return struct.unpack_from('<B', data, offset)[0], offset + 1
        elif field_type == 'int16':
            return struct.unpack_from('<h', data, offset)[0], offset + 2
        elif field_type == 'uint16':
            return struct.unpack_from('<H', data, offset)[0], offset + 2
        elif field_type == 'int32':
            return struct.unpack_from('<i', data, offset)[0], offset + 4
        elif field_type == 'uint32':
            return struct.unpack_from('<I', data, offset)[0], offset + 4
        elif field_type == 'float':
            return struct.unpack_from('<f', data, offset)[0], offset + 4
        elif field_type == 'double':
            return struct.unpack_from('<d', data, offset)[0], offset + 8
        elif field_type == 'bool':
            return struct.unpack_from('<?', data, offset)[0], offset + 1
        else:  # 嵌套结构
            nested_value, nested_offset = self._deserialize(field_type, data[offset:])
            new_offset = offset + nested_offset
            return nested_value, new_offset
