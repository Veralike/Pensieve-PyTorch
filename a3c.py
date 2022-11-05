# 该文件用于求解网络的梯度等信息

import torch
from network import ActorNetwork, CriticNetwork

GAMMA = 0.99                                             # 折扣因子
ENTROPY_WEIGHT = 0.5                                     # 熵权重为0.5
ENTROPY_EPS = 1e-6


class Actor_Critic(object):
    """
    Actor-Critic方法：
    """
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr):
        # 初始化参数：
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_weight = ENTROPY_WEIGHT
        self.entropy_eps = ENTROPY_EPS
        self.discount = GAMMA

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 实例化网络，并直接移动至GPU：
        self.actor_net = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic_net = CriticNetwork(self.state_dim, self.action_dim).to(self.device)

        # 定义均方差损失函数：
        self.loss_func = torch.nn.MSELoss()

        # 定义优化器：
        # 这里源码使用的是RMSprop优化器，通过计算梯度的平方进行梯度和参数更新：
        self.actor_optim = torch.optim.RMSprop(self.actor_net.parameters(), lr=self.actor_lr, alpha=0.9, eps=1e-10)
        self.critic_optim = torch.optim.RMSprop(self.critic_net.parameters(), lr=self.critic_lr, alpha=0.9, eps=1e-10)
        # 梯度清零：
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # 定义a3c更新参数：
        self.actor_net_params = self.actor_net.parameters()
        self.critic_net_params = self.critic_net.parameters()
        self.td_target = 0

    def get_gradient(self, state_batch, action_batch, reward_batch, terminal):
        """
        get_gradient方法：获取网络的梯度
        :param state_batch: 状态数组batch
        :param action_batch: 动作batch
        :param reward_batch: 奖励batch
        :param terminal: 标志位，是否结束（源码未用到）
        :return: None
        """
        state_batch = torch.cat(state_batch).to(self.device)                   # 状态变量（我感觉这一步没有变化？）
        action_batch = torch.LongTensor(action_batch).to(self.device)          # 转换为长整形torch.tensor数据类型
        reward_batch = torch.tensor(reward_batch).to(self.device)              # 转换为torch.tensor数据类型
        accu_reward_batch = torch.zeros(reward_batch.shape).to(self.device)    # 累计奖励函数

        accu_reward_batch[-1] = reward_batch[-1]                               # 累计奖励最后一个值设置为最新的即时奖励

        for t in reversed(range(reward_batch.shape[0] - 1)):
            accu_reward_batch[t] = reward_batch[t] + self.discount * accu_reward_batch[t+1]

        # 计算TD error，需要验证是否使用.flatten()方法：
        with torch.no_grad():
            value_batch = self.critic_net.forward(state_batch.to(self.device)).flatten().to(self.device)
        td_error_batch = accu_reward_batch - value_batch

        # 计算actor网络损失函数，并更新网络参数：
        action_prob = self.actor_net.forward(state_batch.to(self.device))
        prob_generator = torch.distributions.Categorical(action_prob)
        log_action_prob = prob_generator.log_prob(action_batch)
        actor_loss_base = torch.sum(log_action_prob * (-td_error_batch))
        entropy_regularization = -self.entropy_weight * torch.sum(prob_generator.entropy())
        actor_loss = actor_loss_base + entropy_regularization
        actor_loss.backward()                                                  # 使用backward方法自动计算梯度

        # 计算critic网络损失函数，并更新网络参数：
        critic_loss = self.loss_func(accu_reward_batch, self.critic_net.forward(state_batch.to(self.device)).flatten())
        critic_loss.backward()

    def predict(self, state):
        """
        predict方法：输入一个动作，通过actor网络输出动作（select action）
        actor网络拟合策略函数，输出每个动作的概率，在这些概率中随机抽样
        :param state: 输入Pensieve的状态
        :return: action
        """
        with torch.no_grad():
            prob = self.actor_net.forward(state.to(self.device))       # 测试数据集时也需要移动至GPU上
            dist = torch.distributions.Categorical(prob)
            action = dist.sample().item()                              # 在输出中随机选取一个值作为动作
            return action

    def hard_update_actor_net(self, new_actor_net_params):
        """
        这个函数我也不知道是啥意思
        :param new_actor_net_params: 新的actor网络参数
        :return: None
        """
        for target_params, source_params in zip(self.actor_net_params, new_actor_net_params):
            target_params.data.copy_(source_params.data)

    def update_net(self):
        """
        使用PyTorch API 更新网络参数，单步执行梯度更新，并梯度清零
        :return: None
        """
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        self.critic_optim.step()
        self.critic_optim.zero_grad()

    def get_actor_params(self):
        return list(self.actor_net_params)

    def get_critic_params(self):
        return list(self.critic_net_params)
