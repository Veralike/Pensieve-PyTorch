# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

# 该代码将使用PyTorch复现Pensieve模型
import os
import numpy as np
import torch
import time

from get_video_size import get_video_size
from load_trace import load_trace
from agent import agent

# 声明变量：
S_INFO = 6                                            # 进入到Pensieve网络中的输入一共有6个
S_LEN = 8                                             # 选用过去k=8个frames，测量过去8个frames的下载时间、带宽等信息
A_DIM = 6                                             # Pensieve网络输出动作一共6个，输出的动作就是可选码率其中的一个
GAMMA = 0.99                                          # 折扣因子设置为0.99
ACTOR_LR_RATE = 0.0001                                # actor网络学习率
CRITIC_LR_RATE = 0.001                                # critic网络学习率
VIDEO_BITRATE = [300, 750, 1200, 1850, 2850, 4300]    # 可选码率组成一个列表，单位Kbps
RANDOM_SEED = 42                                      # 随机数种子设置为42
SUMMARY_DIR = './results'                             # 设置结果保存地址
COOKED_TRACE_FOLDER = './cooked_traces/'              # 训练集目录位置


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    get_video_size()                                  # 创建记录视频大小的文件

    np.random.seed(RANDOM_SEED)                       # 首先指定随机数种子，帮助创建随机数组。
    torch.manual_seed(RANDOM_SEED)
    assert len(VIDEO_BITRATE) == A_DIM                # 确认码率个数是否等于输出动作，如果不匹配说明Pensieve网络输出存在问题。

    net_params_queue = []                             # 创建网络参数列表
    exp_queue = []                                    # 创建列表

    # 向两个空队列中添加元素，实现多线程之间的通信：
    # for i in range(NUM_AGENTS):
    # net_params_queue.append(torch.multiprocessing.Queue(1))
    # exp_queue.append(torch.multiprocessing.Queue(1))

    # 由于这里只有一个agent，因此这里我去掉for循环试一下：
    net_params_queue.append(torch.multiprocessing.Queue(1))
    exp_queue.append(torch.multiprocessing.Queue(1))

    # 加载文件和原始数据：
    all_cooked_time, all_cooked_bw, _ = load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER)

    if not os.path.exists(SUMMARY_DIR):               # 判断路径是否存在，如果不存在就创建路径。
        os.makedirs(SUMMARY_DIR)

    # 创建agent，开始执行任务：
    # agent = torch.multiprocessing.Process(target=agent,
    #                                       args=(all_cooked_time, all_cooked_bw, net_params_queue, exp_queue))
    # agent.start()

    # 记录当前时间：
    start_time = time.time()

    # 由于这里只有一个agent，因此这里我不使用多线程启动：
    agent = agent(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, net_params_queue=net_params_queue, exp_queue=exp_queue)

    # 记录当前时间：
    end_time = time.time()
    # 计算总体训练时间：
    print(f"训练网络消耗的时间为：{(end_time-start_time)/60.0}min。\n")


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
