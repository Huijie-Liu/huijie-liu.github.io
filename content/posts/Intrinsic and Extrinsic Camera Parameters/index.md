+++
title = '相机的外参和内参'
date = 2024-06-03T20:31:53+08:00
draft = false
description = "详细讲解计算机视觉中相机的内参和外参概念，包括相机标定、坐标转换、投影矩阵等核心知识点。"
tags = ["Learning Note", "Graphics", "Computer Vision"]
categories = ["Learning"]
series = ["Learning Path"]
+++

# 相机的外参和内参

# TLDR

<aside>
💡 **外参矩阵：**将点从世界坐标系转换到相机坐标系。

**内参矩阵：**将点从相机坐标系转换到像素坐标系。

</aside>

# 关键概念

相机是用于捕捉图像的传感器，它们将世界中的点投影到我们看到的2D平面上。

这种变换通常分为两个部分：外部参数和内部参数。相机的外部参数取决于其位置和方向。相机的内部参数则取决于它如何捕捉图像（焦距、光圈、视场、分辨率）。

### 坐标系

在成像过程中，主要使用三种坐标系：

![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled.png)

1. **世界坐标系（3D）**：这是一个固定的全局坐标系，用于描述场景中物体的绝对位置。
2. **相机坐标系（3D）**：这是以相机为中心的坐标系，用于描述相对于相机的物体位置。
3. **图像坐标系（2D）**：这是投影到2D图像平面上的坐标系，用于描述图像上像素的位置。
4. **像素坐标系（2D）**：这是具体到图像像素的坐标系，通常以图像左上角为原点。

### 外部参数

外部参数包括旋转矩阵 $R$ 和平移向量 $t$ ，它们描述了世界坐标系到相机坐标系的变换。具体来说，给定一个世界坐标系中的点 $P_w$ ，通过外部参数变换可以得到相机坐标系中的点 $P_c$ ：

$P_c = R \cdot P_w + t$

其中 $R$ $$ 是一个 $3 \times 3$ 的旋转矩阵， $t$ 是一个 $3 \times 1$ 的平移向量。

### 内部参数

内部参数描述了相机的成像特性，主要包括焦距 $f$ 、主点 $(c_x, c_y)$ 、像素尺寸等。内部参数矩阵 $K$ 可以表示为：

$K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$

其中 $f_x$ 和 $f_y$ 是水平方向和垂直方向的焦距， $c_x$ 和 $c_y$ 是主点（即光轴与图像平面的交点）的坐标。

### 图像形成

通过将外部和内部参数结合起来，可以将世界坐标系中的点转换到图像坐标系中。首先，使用外部参数将世界坐标系中的点转换到相机坐标系：

$P_c = R \cdot P_w + t$

然后，使用内部参数将相机坐标系中的点投影到图像平面：

$\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \cdot P_c$

其中 $(u, v)$ 是图像平面上的像素坐标。

# 成像步骤

1. **世界坐标系（3D）**

   ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%201.png)

2. **相机坐标系（3D）**

   ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%202.png)

   ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%203.png)

   $\begin{pmatrix}X_C\\Y_C\\Z_C\\1\end{pmatrix}=\underbrace{\begin{pmatrix}R_{3\times3}&t_{3\times1}\\0_{1\times3}&1_{1\times1}\end{pmatrix}}_{(4\times4)}\begin{pmatrix}X_W\\Y_W\\Z_W\\1\end{pmatrix}\\\text{Camera Extrinsic Matrix}$

3. **图像坐标系（2D）：针孔模型**
   1. **针孔模型**：光线通过相机开口的中心，并在另一端投影到二维平面上。相机捕捉到的图像就是这个二维平面。

      > 这是一种有损变换，这意味着将点从相机坐标系投影到二维平面上是不可逆的（深度信息丢失）。

      ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%204.png)

   2. **确定坐标**：点的X和Y坐标被投影到二维平面上。二维平面距离相机为 f（焦距）。投影 $( X_i, Y_i )$ 可以通过相似三角形法则找到（进入和离开相机中心的光线与x轴和y轴的角度相同，分别为 $\alpha$ 和 $\beta$ ）。

      ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%205.png)

      $\begin{pmatrix}X_i\\Y_i\\Z_C\end{pmatrix}=\begin{pmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{pmatrix}\begin{pmatrix}X_C\\Y_C\\Z_C\\1\end{pmatrix}$
4. **像素坐标系（2D）**
   1. 常规情况

      1. 像素坐标系 $[u, v]$ 表示通过将图像坐标系中的点离散化而得到的整数值。
      2. 像素坐标系的原点在左上角，因此除了离散化之外，还需要一个平移操作 $(c_x, c_y)$。

      ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%206.png)

      ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%207.png)

      $\begin{pmatrix}u\\v\\w\end{pmatrix}=\begin{pmatrix}1/\rho_u&0&c_x\\0&1/\rho_v&c_y\\0&0&1\end{pmatrix}\begin{pmatrix}X_i\\Y_i\\Z_C\end{pmatrix}$

   2. 特殊情况

      > 有时，二维图像平面不是矩形，而是倾斜的，即X轴和Y轴之间的角度不是90度。在这种情况下，在进行从图像到像素坐标系的变换之前，需要进行另一个变换将点从理想的矩形平面转换到倾斜平面。

      ![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%208.png)

      $\binom{x_{skew}}{y_{skew}}=\begin{pmatrix}1&-\cot(\theta)\\0&1/_{\sin(\theta)}\end{pmatrix}\binom{x}{y}$

# 总结

## 变换

1. **世界到相机（World-to-Camera）**：3D到3D的投影。包括旋转、缩放和平移操作。
2. **相机到图像（Camera-to-Image）**：3D到2D的投影。有信息丢失，取决于相机模型及其参数（如针孔模型、f-theta等）。
3. **图像到像素（Image-to-Pixel）**：2D到2D的投影。从连续到离散。包括量化和原点移动。

![Untitled](%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%A4%96%E5%8F%82%E5%92%8C%E5%86%85%E5%8F%82%2036ff5bff299449a09b551c0176c76ee6/Untitled%209.png)

## 参数

1. 相机外参矩阵 (World-to-Camera)

   将点从世界坐标系转换到相机坐标系。取决于相机的位置和方向。

2. 相机内参矩阵 (Camera-to-Image, Image-to-Pixel)

   将点从相机坐标系转换到像素坐标系。取决于相机的属性（如焦距、像素尺寸、分辨率等）。

# 参考文献

1. [What are Intrinsic and Extrinsic Camera Parameters in Computer Vision?](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)

