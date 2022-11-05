# 该文件用于创建a3c神经网络模型。

import torch
from torch import nn
from torch.nn import functional as func

S_INFO = 6                                               # 为什么源码这里是4？
S_LEN = 8                                                # 选用过去k=8个frames，测量过去8个frames的下载时间、带宽等信息
A_DIM = 6                                                # 可采取动作的数量为6，数值上等于self.action_dim

CONV_OUT_DIM = 128                                       # 一维卷积层输出通道数128（即卷积核filter个数128）
SCA_OUT_DIM = 128                                        # 线性层输出通道数为128
FILTER_SIZE = 4                                          # 卷积核的大小设置为4
STRIDE = 1                                               # 卷积和移动步长设置为1
HIDDEN_DIM = 2 * CONV_OUT_DIM * (S_LEN - FILTER_SIZE + 1) + CONV_OUT_DIM * (A_DIM - FILTER_SIZE + 1) \
    + 3 * SCA_OUT_DIM                                    # 设置中间线性层的输入维数，在此之前需要有flatten步骤


class ActorNetwork(nn.Module):
    """
    创建actor网络：
    state_dim：输入到Pensieve网络中的输入个数和过去测量视频块的个数
    action_dim：输出的动作的个数，输出的动作就是码率的决策
    device：设置神经网络运行设备

    throughput_Conv：past k chunks throughput这个输入值需要经过的卷积层
    downtime_Conv：past k chunks download time这个输入值需要经过的卷积层
    chunksize_Conv：next chunk sizes这个输入值需要经过的卷积层
    buffersize_Linear：current buffer level这个输入值需要经过的线性层
    chunkleft_Linear：chunks left这个输入需要经过的线性层
    chunkbitrate_Linear：last chunk bitrate这个输入需要经过的线性层

    在深度强化学习中，actor网络的最终目标是拟合出策略函数Π(a|s)，即在状态s下采取不同动作a的概率分布
    """
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 定义神经网络层：
        self.throughput_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.downtime_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.chunksize_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.buffersize_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.chunkleft_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.chunkbitrate_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.hidden_layer = nn.Linear(in_features=HIDDEN_DIM, out_features=SCA_OUT_DIM)
        self.output_layer = nn.Linear(in_features=SCA_OUT_DIM, out_features=A_DIM)

        # 神经网络参数初始化：
        nn.init.xavier_normal_(self.buffersize_Linear.weight.data)
        nn.init.constant_(self.buffersize_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunkleft_Linear.weight.data)
        nn.init.constant_(self.chunkleft_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunkbitrate_Linear.weight.data)
        nn.init.constant_(self.chunkbitrate_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.hidden_layer.weight.data)
        nn.init.constant_(self.hidden_layer.bias.data, 0.0)
        nn.init.xavier_normal_(self.throughput_Conv.weight.data)
        nn.init.constant_(self.throughput_Conv.bias.data, 0.0)
        nn.init.xavier_normal_(self.downtime_Conv.weight.data)
        nn.init.constant_(self.downtime_Conv.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunksize_Conv.weight.data)
        nn.init.constant_(self.chunksize_Conv.bias.data, 0.0)

    def forward(self, in_put):
        # 创建前向传播函数：
        chunkbitrate_out = func.relu(self.chunkbitrate_Linear(in_put[:, 0:1, -1]), inplace=True)
        buffersize_out = func.relu(self.buffersize_Linear(in_put[:, 1:2, -1]), inplace=True)
        throughput_out = func.relu(self.throughput_Conv(in_put[:, 2:3, :]), inplace=True)
        downtime_out = func.relu(self.downtime_Conv(in_put[:, 3:4, :]), inplace=True)
        chunksize_out = func.relu(self.chunksize_Conv(in_put[:, 4:5, :self.action_dim]), inplace=True)
        chunkleft_out = func.relu(self.chunkleft_Linear(in_put[:, 5:6, -1]), inplace=True)       # tf源码这里是4:5，最后可以验证一下

        # 将卷积层输出展开：
        throughput_flatten = throughput_out.view(throughput_out.shape[0], -1)
        downtime_flatten = downtime_out.view(downtime_out.shape[0], -1)
        chunksize_flatten = chunksize_out.view(chunksize_out.shape[0], -1)

        # 将多个输出拼成一张大的线性层：
        hidden_layer_input = torch.cat([chunkbitrate_out, buffersize_out, throughput_flatten, downtime_flatten, chunksize_flatten, chunkleft_out], 1)
        hidden_layer_output = func.relu(self.hidden_layer(hidden_layer_input), inplace=True)

        # 最后经过softmax：
        out_put = torch.softmax(self.output_layer(hidden_layer_output), dim=-1)
        return out_put


class CriticNetwork(nn.Module):
    """
    创建critic网络：
    基本和actor_network相同，需要注意的是最后一层输出通道数为1

    在深度强化学习中，critic网络的最终目标是拟合出状态价值函数Q(s,a)，即在状态s下采取动作a的累计奖励回报的期望
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 定义神经网络层：
        self.throughput_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.downtime_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.chunksize_Conv = nn.Conv1d(in_channels=1, out_channels=CONV_OUT_DIM, kernel_size=FILTER_SIZE, stride=STRIDE)
        self.buffersize_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.chunkleft_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.chunkbitrate_Linear = nn.Linear(in_features=1, out_features=SCA_OUT_DIM)
        self.hidden_layer = nn.Linear(in_features=HIDDEN_DIM, out_features=SCA_OUT_DIM)
        self.output_layer = nn.Linear(in_features=SCA_OUT_DIM, out_features=1)

        # 神经网络参数初始化：
        nn.init.xavier_normal_(self.buffersize_Linear.weight.data)
        nn.init.constant_(self.buffersize_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunkleft_Linear.weight.data)
        nn.init.constant_(self.chunkleft_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunkbitrate_Linear.weight.data)
        nn.init.constant_(self.chunkbitrate_Linear.bias.data, 0.0)
        nn.init.xavier_normal_(self.hidden_layer.weight.data)
        nn.init.constant_(self.hidden_layer.bias.data, 0.0)
        nn.init.xavier_normal_(self.throughput_Conv.weight.data)
        nn.init.constant_(self.throughput_Conv.bias.data, 0.0)
        nn.init.xavier_normal_(self.downtime_Conv.weight.data)
        nn.init.constant_(self.downtime_Conv.bias.data, 0.0)
        nn.init.xavier_normal_(self.chunksize_Conv.weight.data)
        nn.init.constant_(self.chunksize_Conv.bias.data, 0.0)

    def forward(self, in_put):
        # 创建前向传播函数：
        chunkbitrate_out = func.relu(self.chunkbitrate_Linear(in_put[:, 0:1, -1]), inplace=True)
        buffersize_out = func.relu(self.buffersize_Linear(in_put[:, 1:2, -1]), inplace=True)
        throughput_out = func.relu(self.throughput_Conv(in_put[:, 2:3, :]), inplace=True)
        downtime_out = func.relu(self.downtime_Conv(in_put[:, 3:4, :]), inplace=True)
        chunksize_out = func.relu(self.chunksize_Conv(in_put[:, 4:5, :self.action_dim]), inplace=True)
        chunkleft_out = func.relu(self.chunkleft_Linear(in_put[:, 5:6, -1]), inplace=True)

        # 将卷积层输出展开：
        throughput_flatten = throughput_out.view(throughput_out.shape[0], -1)
        downtime_flatten = downtime_out.view(downtime_out.shape[0], -1)
        chunksize_flatten = chunksize_out.view(chunksize_out.shape[0], -1)

        # 将多个输出拼成一张大的线性层：
        hidden_layer_input = torch.cat([chunkbitrate_out, buffersize_out, throughput_flatten, downtime_flatten, chunksize_flatten, chunkleft_out], 1)
        hidden_layer_output = func.relu(self.hidden_layer(hidden_layer_input), inplace=True)

        # 最后直接输出：
        out_put = self.output_layer(hidden_layer_output)
        return out_put

# 经过验证，这个模型能够运行
