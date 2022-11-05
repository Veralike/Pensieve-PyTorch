# 该文件用于智能体和环境的交互
# 目前我们已经拟合出了Policy-Based和Value-Based函数，接下来我们需要让智能体学习
# 该文件简单实现1个agent的交互过程，多个agent暂不实现

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from env import Environment
from a3c import Actor_Critic

S_INFO = 6
S_LEN = 8
A_DIM = 6
VIDEO_BITRATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
REBUF_PENALTY = 4.3                                   # 卡顿惩罚项，QoE计算需去除
SMOOTH_PENALTY = 1.0                                  # 切换流畅度惩罚项，QoE计算需去除
CHUNK_TIL_VIDEO_END_CAP = 48.0                        # 标志位，用于判断视频块是否全部发送完毕
B_IN_KB = 1024.0                                      # 这里我将原始M_IN_K变量名改为B_IN_KB，并且将参数值改为1024
B_IN_MB = 1024.0 * 1024.0                             # 添加新参数，表示1M之内多少字节
KB_IN_MB = 1024.0                                     # 添加新参数
MILLISECOND_IN_SECOND = 1000.0                        # 添加新参数，表示1s内ms的个数
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENT = 4                                         # agent的个数，这里由于没有设置central_agent，因此个数为1
RANDOM_SEED = 42
DEFAULT_QUALITY = 1                                   # 默认质量选择1，因为video1目录下视频是最好的
TRAIN_SEQ_LEN = 100                                   # 根据论文，强化学习agent和env交互一共100次
TOTAL_EPOCH = 30000
# 由于使用相对路径会出现一定的问题，因此这里我是用绝对路径：
ACTOR_NET_PATH = r'E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Pensieve\selflearning\results\pt\actor.pth'
CRITIC_NET_PATH = r'E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Pensieve\selflearning\results\pt\critic.pth'
TB_LOG_PATH = r'E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Pensieve\selflearning\results\tb_logs'
LOG_FILE_PATH = r'E:\Python_Project\SelfLearning\Deep_Reinforcement_Learning\Pensieve\selflearning\results\log'    # 设置log文件的路径


def agent(all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    """
    agent方法，和环境进行交互
    :param all_cooked_time: 数据集所有文件的时间戳
    :param all_cooked_bw: 数据集所有文件的网络吞吐量
    :param net_params_queue: 网络参数队列
    :param exp_queue: 不知道是啥
    :return: None
    """
    # 限制CPU上进行深度学习训练的线程数，限制PyTorch占用的CPU的数目：
    torch.set_num_threads(1)
    # 创建交互环境，加载数据集中的时间戳、带宽等数据：
    env = Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=RANDOM_SEED)
    # 实例化writer对象：
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=TB_LOG_PATH)

    with open(LOG_FILE_PATH, 'w') as log_file:
        # 创建agent：
        agent_net = Actor_Critic(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)

        time_stamp = 0

        for epoch in tqdm(range(TOTAL_EPOCH)):
            # actor_net_params = net_params_queue.get()
            # agent_net.hard_update_actor_net(actor_net_params)

            last_bitrate = DEFAULT_QUALITY                    # 上一个视频块的码率等级为1
            bitrate = DEFAULT_QUALITY                         # 设置选择的码率等级为1（保证视频质量最好）

            s_batch = []                                      # 创建三个列表，分别保存state、action、reward参数
            a_batch = []
            r_batch = []
            entropy_record = []                               # 记录熵的信息

            state = torch.zeros((1, S_INFO, S_LEN))           # 初始化state数组

            # 开始与环境交互，更新state数组：
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_size, \
                end_of_video, video_chunk_remain = env.get_video_chunk(bitrate)

            time_stamp += delay
            time_stamp += sleep_time                          # 加上下载视频块的时间和休眠时间

            while not end_of_video and len(s_batch) < TRAIN_SEQ_LEN:
                # 迭代码率决策：
                last_bitrate = bitrate

                # 利用env返回的数据更新state：
                state = state.clone().detach()
                state = torch.roll(input=state, shifts=-1, dims=-1)

                state[0, 0, -1] = VIDEO_BITRATE[bitrate] / float(np.max(VIDEO_BITRATE))
                state[0, 1, -1] = buffer_size / BUFFER_NORM_FACTOR
                state[0, 2, -1] = float(video_chunk_size) / B_IN_KB / float(delay)
                state[0, 3, -1] = float(delay) / MILLISECOND_IN_SECOND / BUFFER_NORM_FACTOR
                state[0, 4, :A_DIM] = torch.tensor(next_video_chunk_size) / B_IN_MB
                state[0, 5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # 使用agent预测下一个码率决策：
                bitrate = agent_net.predict(state)

                # 重新交互env，获取新一轮的信息：
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_size, \
                    end_of_video, video_chunk_remain = env.get_video_chunk(bitrate)

                # 计算根据QoE指标reward，同时减去两个惩罚项：
                reward = VIDEO_BITRATE[bitrate] / KB_IN_MB \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BITRATE[bitrate] - VIDEO_BITRATE[last_bitrate]) / KB_IN_MB

                # 将数据添加至列表中：
                s_batch.append(state)
                a_batch.append(bitrate)
                r_batch.append(reward)
                entropy_record.append(3)                      # 不知道这里为什么是3？

                # 写入信息至日志文件：
                log_file.write(
                    str(time_stamp) + '\t' +
                    str(VIDEO_BITRATE[bitrate]) + '\t' +
                    str(buffer_size) + '\t' +
                    str(rebuf) + '\t' +
                    str(video_chunk_size) + '\t' +
                    str(delay) + '\t' +
                    str(reward) + '\n' + '\n'
                )
                log_file.flush()

                # 操作队列：
                # exp_queue.put([
                # s_batch,
                # a_batch,
                # r_batch,
                # end_of_video,
                # {'entropy': entropy_record}
                # ])
                log_file.write('\n')

            print(f"当前epoch：{epoch}。\n")

            # 获取QoE指标参数，QoE指标的值就是奖励值：
            qoe_batch = r_batch
            writer.add_scalar(tag='QoE Metric Value', scalar_value=sum(qoe_batch), global_step=epoch)

        writer.close()

    # 最后保存模型：
    torch.save(agent_net.actor_net, ACTOR_NET_PATH)
    torch.save(agent_net.critic_net, CRITIC_NET_PATH)
    # 使用torch.load()方法加载模型
