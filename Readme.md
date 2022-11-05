# Readme

本工程用于复现Pensieve代码。源码能够实现多个agent的加速训练，而该代码则简单实现一个agent和环境env的交互。

## 配置信息

运行该代码的电脑配置信息如下：

* 操作系统：Windows10
* 内存：16GB
* 显卡：Nvidia GTX 1060 6GB
* 语言：Python 3.9.6
* 深度学习库：PyTorch 1.8.2 LTS
* CUDA：10.2

## 运行方法

直接运行即可。

``` powershell
python main.py
# 或
python3 main.py
```

## 数据集

数据集在清华PiTree网站中寻得，该网站中可以找到FCC18、HSR等多种记录时刻和网络吞吐量的数据集。

网址：[PiTree Traces (transys.io)](https://transys.io/pitree-dataset/traces/index.html)，点击链接即可下载。

## Actor-Critic网络

根据论文给出的图片，Actor-Critic网络如下：

![image-20221020085338297](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221020085338297.png)

Actor网络和Critic网络的形状基本相同，区别仅仅在于最后输出的通道数。

Actor网络拟合策略函数Π(a|s)，表示在s状态下选取动作a的概率；Critic函数拟合动作价值函数Q(s,a)，表示在s状态下选取动作a能够得到的奖励函数的期望。

> 总结：Actor网络只负责选取动作，而Critic网络负责给Actor网络选取的动作打分。

使用PyTorch生成的随机数对神经网络进行测试训练，发现能够正常运行。

## 优化器

这里使用的两个优化器为：RMSprops。

RMSProps优化器结合**梯度平方的指数移动平均数**来调节参数的变化，在不稳定的目标函数中具有较好的收敛效果。

## QoE

作者提出的QoE公式为：
$$
QoE = \sum_{n=1}^{N} q(R_n) - \mu \sum_{n=1}^{N} T_n - \sum_{n=1}^{N-1} |q(R_{n+1}) - q_n|
$$
在强化学习中，**即时奖励**reward就是这个QoE指标（不带求和符号）。

其中R~n~代表第n步选择的码率；T~n~代表`env.py`文件中下载某一个码率等级的视频块，卡住需要加载的时间；u代表针对于卡顿时间T~n~的一个惩罚系数，记为`REBUF_PENALTY`，该值设定为4.3；最后一项表示两个视频块之间的码率差，这里省略了一个平滑度惩罚系数`SMOOTH_PENALTY`，由于这个值设定为1，因此这里省略没写。

#### QoE——绘图思路

为了校验模型的效果，在`agent.py`文件中使用TensorBoard绘制QoE指标的图像。每一轮训练中会通过QoE指标公式计算即时奖励`reward`，并添加至`r_batch`列表中，因此思路为每一轮训练结束后计算`r_batch`列表的总和，观看奖励总和是否呈现减小趋势。

``` python
# 思路：
qoe_batch = r_batch
writer.add_scalar('QoE Metric', sum(qoe_batch), epoch)
```

使用`tensorboard`命令查看运行的QoE指标变化：

``` powershell
tensorboad --logdir=results\tb_logs --port=6007
```

显示如下：

![image-20221105201817347](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221105201817347.png)

可以看出目前模型效果并不是很稳定，有待进一步优化；也有可能绘图思路出现偏差，有待进一步思考。

## 难点与问题

总结一下现在遇到的问题：

* 构建了拟合策略函数、动作价值函数的**神经网络**之后，应该怎么去训练？是使用PyTorch单独进行训练，还是通过强化学习的方法，在agent与env交互的过程中训练？
* 怎么用代码去体现agent和env的交互过程？换句话说，怎么用代码去体现Policy Gradient算法和TD算法？

>  根据我的思考，我的初步回答是：在`a3c.py`文件中首先实例化actor网络和critic网络，然后在`__init__`方法中定义Policy Gradient方法（公式）和TD方法（公式），然后在`agent.py`文件中开始循环，表示强化学习训练开始，每次获得一个action后，就通过`get_gradient`方法计算Policy Gradient和TD，从而达到更新网络的目的。

然而在TensorFlow源码和PyTorch源码中，我并没有看到严格对应Policy Gradient算法和TD算法的代码，也就是说虽然源码虽然也是求动作价值函数的梯度，然而**并没有严格对应于文献中的公式**，这是目前最疑惑的问题。



以上内容是复现Pensieve文献的心得，后续会继续更新
