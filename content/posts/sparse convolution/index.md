+++
title = '稀疏卷积'
date = 2024-06-04T10:31:53+08:00
draft = false
tags = ["CNN"]
categories = ["Learning"]
series = ["Learning Path"]
+++

# 稀疏卷积

# TLDR

<aside>
💡 通过对输入输出不为空的数据建立位置哈希表和RuleBook，只对有效数据计算卷积，从而减少计算量！

> Rulebook 储了稀疏卷积过程中需要计算的位置和操作，但是这个过程是高度可并行化的，因此可以有效地利用GPU来加速。

</aside>

# 普通卷积

- 卷积的实际流程
  ![Pasted image 20240602221433.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602221433.png)
- 反卷积的实际流程
  ![Pasted image 20240602222420.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602222420.png)
- 矩阵是如何存储的？
  ![Pasted image 20240602224627.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602224627.png)
  这个矩阵的作用是将原始输入张量中每个像素的特征向量投影到输出通道的特征向量上。可以把它想象成一个坐标变换的过程，将一个3维数据投影到128维空间。
  ![Pasted image 20240602224637.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602224637.png)
- 矩阵是如何优化的？
  对单个像素的优化：
  ![Pasted image 20240602225048.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602225048.png)
- 一次卷积的完整流程
  ![Pasted image 20240602225156.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602225156.png)

# 稀疏卷积

- 普通卷积的推广
  $$
  \mathrm{x_u=\sum W_{i}x_{i+u}}\:for\: u\in\mathbb{C}^{out}
  $$
  其中 i 属于 N，代表卷积核的像素位置相对于当前处理的像素 (u) 的偏移量。
  具体来说，稀疏卷积使用一个被称为“Rulebook”的结构来安排所有原子操作（例如卷积核对应位置的计算）。这个 Rulebook 是一组规则，定义了输入特征的哪些位置需要和卷积核进行计算。
  ![Pasted image 20240602230539.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602230539.png)
- 坐标散列（coordinate hashing）

  1. 要解决的问题

     当稀疏3D点云在大部分空间中是空的，如何在下采样后恢复到原始分辨率是一个很具挑战性的任务。坐标散列将不同层次之间的坐标映射起来，以便在卷积和反卷积操作中有效地追溯和恢复这些点的位置。

  2. 原理

     本质上是一种记录和查询机制，可以将高维空间中的稀疏数据映射到一个紧凑的、易于管理的格式中。通过这个机制，每个点的空间信息都能在不同的层次结构中进行高效的映射和还原，从而使得在下采样和上采样（反卷积）操作之后，数据能够准确恢复到它们原来的空间位置。
     ![不同层的映射是分层链接的。](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Untitled.png)
     不同层的映射是分层链接的。
     ![Pasted image 20240602231004.png](%E7%A8%80%E7%96%8F%E5%8D%B7%E7%A7%AF%2049df8c7a19204e6a835b9496877610f6/Pasted_image_20240602231004.png)

---

# 参考文献

1. [How does sparse convolution work?](https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1)
2. [Sparse Convolution explained with code](https://rancheng.github.io/Sparse-Convolution-Explained/)
