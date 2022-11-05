# 该文件用于创建强化学习中的环境。

import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0                                     # 按时间切割视频块，每个视频块长4000.0ms
BITRATE_LEVELS = 6                                           # 可选码率一共有6个
TOTAL_VIDEO_CHUNK = 48                                       # 这里源码给出的值为48，论文原文也说总数为48
LINK_RTT = 80                                                # RTT时间设置为80ms
DRAIN_BUFFER_SLEEP_TIME = 500.0                              # 暂停500ms后再继续下载

NOISE_LOW = 0.9                                              # 乘性噪声的范围，这个将影响delay的值
NOISE_HIGH = 1.1
PACKET_SIZE = 1500                                           # 这个应该是网络连接中每次发送的数据包的大小为1500bytes
PACKET_PAYLOAD_PORTION = 0.95                                # payload占整个packet的比例
B_IN_MB = 1024.0 * 1024.0                                    # 这里我改为1024*1024
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND                # 缓冲区上限为60s，超过则停止继续下载视频块
VIDEO_SIZE_FILE = './video_size_'                            # 在当前目录下存在多个以'video_size_'开头的记录视频大小的文件


# 这里的环境指和Pensieve模型交互的网络环境
class Environment(object):
    """
    初始化方法
    self.cooked_time：随机抽取的某一数据集的所有时间戳构成的列表
    self.cooked_bw：随机抽取的某一数据集的所有吞吐量构成的列表
    self.video_size：存放每一码率等级的所有视频块的大小
    """
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed):
        assert len(all_cooked_time) == len(all_cooked_bw)    # 首先判断两个值是否相等，长度就是数据集中文件的个数

        np.random.seed(random_seed)                          # 创建一组随机数

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.video_chunk_counter = 0                         # 视频块的计数器，起到遍历的作用，指定这个参数的值可以找到49个视频块中的每一个
        self.buffer_size = 0                                 # 视频缓冲区的大小首先设置为0

        # 随机挑选一个训练数据集，获得了时间和带宽两个参数值：
        self.trace_index = np.random.randint(len(self.all_cooked_time))              # 首先在数据集众多文件中随机挑选一个文件
        self.cooked_time = self.all_cooked_time[self.trace_index]                    # 获得该文件的所有时间戳（列表形式）
        self.cooked_bw = self.all_cooked_bw[self.trace_index]                        # 获得该文件的所有吞吐量（列表形式）这个值已经确定了

        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))                # 在1到总文件数之间生成一个随机整数
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]            # 在随机挑选的这个文件中随机找一个开始时间戳（指上一个）

        self.video_size = {}                                                         # 创建字典video_size，存放每个码率等级的所有视频块大小
        for bitrate in range(BITRATE_LEVELS):                                        # 在字典中，对于每一个码率等级，都对应创建一个列表
            self.video_size[bitrate] = []                                            # 首先对每一个索引创建一个空列表
            with open(VIDEO_SIZE_FILE + str(bitrate)) as file:                       # 不写参数mode时默认只读
                for line in file:
                    self.video_size[bitrate].append(int(line.split()[0]))            # 把'video_size_'文件中每一行的值添加进去

    def get_video_chunk(self, quality):
        assert quality >= 0                                                          # 首先判断想要获取视频的质量是否满足大于0
        assert quality < BITRATE_LEVELS                                              # 其次判断视频质量是否在可选范围以内，quality范围从0~5

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]        # 从字典中获取某一码率等级的某一视频块大小

        delay = 0.0                                                                  # 设置参数delay，单位ms
        video_chunk_counter_sent = 0                                                 # 指定视频块传输了多少数据，单位byte

        """
        循环结束的条件为：有一个视频块被下载完毕。
        需要经历多个duration，每个duration中网络中总吞吐量为packet_payload = duration * throughput * 0.95
        如果packet_payload小于选中的视频块（video_chunk_size），就说明还要至少再经历一轮duration
        在经过多轮duration后，如果总吞吐量大于视频块大小，就跳出循环
        
        经过while循环后，记录下了完成下载某一个视频块的时间戳和下载所需时间delay
        delay：按throughput计算，从开始下载视频块到下载完成，所需要的总时间，最后还需要加上RTT时间和噪声影响
        """
        while True:                                                                  # 通过Linux中的mahimahi下载视频块
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE  # 获取last_mahimahi_time时刻吞吐量，并换算单位
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time # 获取上一个时间戳和这一个时间戳的差值，就是持续时间

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION          # 计算在这段时间内总的吞吐量

            if packet_payload + video_chunk_counter_sent > video_chunk_size:         # 如果这段时间内的吞吐量大于视频块本身的大小
                fractional_time = (video_chunk_size - video_chunk_counter_sent) \
                                  / throughput / PACKET_PAYLOAD_PORTION              # 计算视频块传输完毕还要多长时间
                delay += fractional_time                                             # 加上这段时间
                self.last_mahimahi_time += fractional_time                           # 在上一个时间戳的基础上加上剩余时间
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break                                                                # 跳出循环

            video_chunk_counter_sent += packet_payload                               # 表示视频块未成功下载之前，已经发送了这么多
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]            # 重置上一个时间戳
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):                             # 如果索引值超过了长度，再次从头开始循环
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0                                          # 从头开始是可行的，因为duration计算的是相对时间

        delay *= MILLISECONDS_IN_SECOND                                              # 单位换算
        delay += LINK_RTT                                                            # 加上连接所需要的RTT时间
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)                            # 添加乘性噪声，最终获得下载指定视频块的总时间

        """
        开始比较delay和buffer_size的大小，判断是否有rebuffering事件
        buffer_size指的是缓冲区已有的视频的长度
        当buffer_size小于delay时，说明此时缓冲区内容已经播放完毕，然而新一个视频块还没下载完成，此时进入rebuffering状态
        delay-buffer_size的时间是需要等待的时间，当经过这段时间，新的视频块下载完毕，可以继续观看
        rebuffering状态指视频卡顿，正在缓冲
        """
        rebuf = np.maximum(delay - self.buffer_size, 0.0)                            # 记录rebuffering的时间
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)                 # 重算buffer_size大小，如果进入rebuffering状态则为0
        self.buffer_size += VIDEO_CHUNK_LEN                                          # 下载完毕后，缓冲区增加视频块的时间长度

        """
        当缓冲区长度过大时，需要进入休眠状态，不再下载视频块
        不再下载视频块说明buffer_size不会继续增加，同时由于视频的继续播放还会继续减小
        buffer_size减小的值就是sleep_time的大小，因为在这段时间内没有新的缓冲，同时视频还在继续播放
        休眠结束后，记录下当前的时间戳
        """
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:                                         # 如果缓冲区过大，则进入休眠状态，停止继续下载视频块
            drain_buffer_time = self.buffer_size - BUFFER_THRESH                     # 计算超出缓冲区的时间
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) \
                         * DRAIN_BUFFER_SLEEP_TIME                                   # 休眠时间向上取整（对应论文）
            self.buffer_size -= sleep_time                                           # 超出缓冲区的时间则休眠，保证buffer_size小于阈值

            while True:                                                              # 循环的作用：找到sleep_time结束后的时间戳
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time                                 # 计算从下载完毕的时间到下一个开始时间戳的时间
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND   # 记录下休眠结束的时间戳
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND                      # 到达mahimahi_ptr时刻时，还需要休眠的时间
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]        # 重置时间戳
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):                         # 如果索引值超过了长度，则重新开始循环
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        return_buffer_size = self.buffer_size                                        # 返回缓冲区的长度

        self.video_chunk_counter += 1                                                # 计数器加一
        video_chunk_remain = TOTAL_VIDEO_CHUNK - self.video_chunk_counter            # 计算还剩下多少个视频块

        end_of_video = False                                                         # 接下来判断所有视频块是否发送完毕
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNK:
            end_of_video = True                                                      # 表明所有视频块发送完毕
            self.buffer_size = 0
            self.video_chunk_counter = 0                                             # 缓冲区、计数器全部归零，将视频从头再发一遍

            self.trace_index = np.random.randint(len(self.all_cooked_time))          # 重新随机挑选时间戳和网络吞吐量条件再来一遍
            self.cooked_time = self.all_cooked_time[self.trace_index]
            self.cooked_bw = self.all_cooked_bw[self.trace_index]
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_size = []                                                   # 记录下一个视频块的大小，并返回
        for level in range(BITRATE_LEVELS):
            next_video_chunk_size.append(self.video_size[level][self.video_chunk_counter])

        """
        最后需要返回的值：
        delay：获取某个视频块的下载时间
        sleep_time：缓冲区超过阈值后，需要休眠的时间，是500ms的整数倍
        return_buffer_size：下载了一个视频块后，现在缓冲区的大小
        rebuf：重新缓冲时间（即卡顿的等待时间）
        video_chunk_size：下载的视频块的大小
        next_video_chunk_size：不同码率等级中，下一个视频块的大小
        end_of_video：代表49个视频块是否全部下载完成
        video_chunk_remain：49个视频块中还有多少个没有下载
        """
        return delay, sleep_time, return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, next_video_chunk_size, \
            end_of_video, video_chunk_remain
