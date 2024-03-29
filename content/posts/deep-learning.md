---
title: 'Deep Learning'
date: "2024-03-10"
draft: false
tags: ["Learning Note", "Deep Learning"]
categories: ["Learning"]
series: ["Learning Path"]
---

# 第一章 引言

一些前置概念：

-   深度学习（deep learning）：让计算机从**经验**中学习，根据**层次化**的概念体系理解世界。深度学习通过其他较简单的表示来表达复杂表示，解决了表示学习中的核心问题。

-   知识库（knowledge base）：将知识用形式化的语言进行硬编码，使用逻辑推理规则来自动地理解这些形式化语言的声明。

-   机器学习（machine learning）：AI系统需要具备自己获取知识的能力，即**从原始数据中提取模式的能力。**

-   表示学习（represnetation learning）：使用**机器学习**来**发掘表示本身**，而不仅仅把标识映射到输出。

-   变差因素（factors of variation）：“因素”仅指代影响的不同来源，这些因素通常是不能被直接观测到的量，可以被看作**数据的概念或者抽象**。如，分析语音时，变差因素包括说话者的年龄、性别、口音和正在说的词语；分析汽车图像时，变差因素包括汽车的位置、颜色、太阳的角度和亮度。

-   多层感知机（multilayer perceptron, MLP）：是一个将一组输入值映射到输出值的数学函数。该函数由许多较简单的函数复合而成。

    <img src="https://p.ipic.vip/um9m2r.png" alt="image-20230502160406053" style="zoom:50%;" />

    <div align = "center">深度学习模型示意图</div>

    <img src="https://p.ipic.vip/iswd3l.png" alt="image-20230502164426861" style="zoom: 33%;" />

    <div align = "center">深度学习韦恩图</div>

    

# 第二章 机器学习基础

## 学习算法

机器学习算法是一种能够从数据中学习的算法。“对于某类任务T和性能度量P，一个计算机程序被认为可以从经验E中学习是指，通过经验E改进后，它在任务T上由性能度量P衡量的性能有所提升。”

### 任务T

通常机器学习任务定义为机器学习系统应该如何处理样本 （example）。样本是指我们从某些希望机器学习系统处理的对象或事件中收集到的已经量化的特征 （feature）的集合。常见的机器学习任务如下：**分类、输入缺失分类、回归、转录、机器翻译、结构化输出、异常检测、合成和采样、缺失值填补、去噪、密度估计**

### 性能度量P

“通常性能度量P是特定于系统执行的任务T而言的。”

### 经验E

“机器学习算法可以大致分类为无监督 （unsupervised）算法和监督 （supervised）算法。

#### 无监督学习算法

 （unsupervised learning algorithm）训练含有很多特征的数据集，然后学习出这个数据集上有用的结构性质。在深度学习中，我们通常要学习生成数据集的整个概率分布，显式地，比如密度估计，或是隐式地，比如合成或去噪。还有一些其他类型的无监督学习任务，例如聚类，将数据集分成相似样本的集合。

#### 监督学习算法

 （supervised learning algorithm）训练含有很多特征的数据集，不过数据集中的样本都有一个标签 （label）或目标 （target）。例如，Iris数据集注明了每个鸢尾花卉样本属于什么品种。监督学习算法通过研究Iris数据集，学习如何根据测量结果将样本划分为3个不同品种。”

## 容量、过拟合和欠拟合

在先前未观测到的输入上表现良好的能力被称为**泛化** （generalization）。**欠拟合**是指模型不能在训练集上获得足够低的误差，而**过拟合**是指训练误差和测试误差之间的差距太大。模型的**容量**是指其拟合各种函数的能力。容量低的模型可能很难拟合训练集。容量高的模型可能会过拟合。

![image-20230628234605792](https://p.ipic.vip/jq47vd.png)

### 正则化

正则化是指修改学习算法，使其**降低泛化误差**而非训练误差。正则化是机器学习领域的中心问题之一，只有优化能够与其重要性相提并论。

## 超参数和验证集

大多数机器学习算法都有超参数，可以设置来**控制算法行为**。超参数的值不是通过学习算法本身学习出来的。

### 交叉验证

当数据集太小时，最常见的是k-折交叉验证过程，将数据集分成k个不重合的子集。测试误差可以估计为k次计算后的平均测试误差。在第i次测试时，数据的第i个子集用于测试集，其他的数据用于训练集。

## 估计、偏差和方差

### 点估计

为一些感兴趣的量提供单个“最优”预测。点估计也可以指输入和目标变量之间关系的估计，我们将这种类型的点估计称为**函数估计**。

### 偏差

估计的偏差被定义为

![image-20230629204506339](https://p.ipic.vip/z1t2nb.png)

### 方差和标准差

期望的变化程度是多少。

## 监督学习算法

监督学习算法是给定一组输入x和输出y的训练集，学习如何关联输入和输出

## 无监督学习算法

无监督算法只处理“特征”，不操作监督信号。监督和无监督算法之间的区别没有规范严格的定义，因为没有客观的判断来区分监督者提供的值是特征还是目标。通俗地说，无监督学习的大多数尝试是指从不需要人为注释的样本的分布中抽取信息。

# 第三章 深度前馈网络

## 介绍

**深度前馈网络**（deep feedforward network），也叫作**前馈神经网络**（feedforward neural net-work）或者**多层感知机**（multilayer perceptron，MLP）。“这种模型被称为**前向**（feedforward）的，是因为信息流过x的函数，流经用于定义f的中间计算过程，最终到达输出y。在模型的输出和模型本身之间没有**反馈**（feedback）连接。当前馈神经网络被扩展成**包含反馈连接**时，它们被称为**循环神经网络**。

前馈神经网络用许多不同函数复合在一起来表示。该模型与一个有向无环图相关联，而图描述了函数是如何复合在一起的。多个函数连接在一个链上称为**链式结构**，这些链式结构是神经网络中最常用的结构。链的全长被称为**深度**。最后一层被称为**输出层**。

在神经网络训练过程中，训练数据直接指明了输出层在每一点x上必须做什么，但是训练数据并没有说每个单独的层应该做什么，所以这些层被称为**隐藏层**。

## 基于梯度的学习

在于神经网络的非线性导致大多数我们感兴趣的代价函数都变得非凸。这意味着神经网络的训练通常使用迭代的、基于梯度的优化，仅仅使得代价函数达到一个非常小的值。

### 代价函数

神经网络的代价函数或多或少是和其他的参数模型（例如线性模型的代价函数）相同的。

### 输出单元

代价函数的选择与输出单元的选择紧密相关。任何可用作输出的神经网络单元，也可以被用作隐藏单元。

## 隐藏单元

## 架构设计

更深层的网络通常能够对每一层使用更少的单元数和更少的参数，并且经常容易泛化到测试集，但是通常也更难以优化。

**万能近似定理**表明，一个前馈神经网络如果具有线性输出层和至少一层具有任何一种“挤压”性质的激活函数（例如logistic sigmoid激活函数）的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的Borel可测函数。

## 反向传播和其他的微分算法

# 第四章 正则化

# 第五章 优化

# 第六章 卷积网络

# 第七章 循环网络

# 第八章 实践



