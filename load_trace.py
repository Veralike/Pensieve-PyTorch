# 该文件用于预处理数据集。

import os

COOKED_TRACE_FOLDER = './cooked_traces/'                       # 训练集目录位置


def load_trace(cooked_trace_folder):
    cooked_files_list = os.listdir(cooked_trace_folder)        # 列出目录下有哪些文件，组成一个列表

    all_cooked_time = []                                       # 创建三个空列表
    all_cooked_bw = []
    all_file_names = []

    for cooked_file in cooked_files_list:                      # 对目录下每一个文件进行操作
        file_path = cooked_trace_folder + cooked_file          # 组合成文件路径
        cooked_time = []                                       # 记录目录下每一个文件的所有时间戳信息
        cooked_bw = []                                         # 记录目录下每一个文件的所有吞吐量信息

        # 对训练集数据做数据拆分（预处理）：
        with open(file_path, 'rb') as file:                    # 数据集中每一个文件每一行有两个数：时间戳和吞吐量（码率）
            for line in file:
                parse = line.split()                           # 对训练集数据做拆分，默认以空格对语句进行拆分
                cooked_time.append(float(parse[0]))            # 取每一行第一个值作为时间戳
                cooked_bw.append(float(parse[1]))              # 取每一行第二个值作为吞吐量

        all_cooked_time.append(cooked_time)                    # 将目录下每一个文件的时间戳（列表）写入一个列表（得到二维列表）
        all_cooked_bw.append(cooked_bw)                        # 将目录下每一个文件的吞吐量（列表）写入一个列表（得到二维列表）
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names      # all_cooked_time列表中存放的是目录下每个文件的所有时间戳

# 说明一下：Pensieve使用的数据集中，每一行都由两个值组成，
# 一个时间戳timestamp，一个是吞吐量throughput。
