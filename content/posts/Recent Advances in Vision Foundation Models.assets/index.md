+++
title = 'Recent Advances in Vision Foundation Models'
date = 2024-03-28T13:30:42+08:00
draft = false
tags = ["Deep Learning", "Computer Vision"]
categories = ["Learning"]
series = ["Learning Path"]

+++

> [原文链接](https://arxiv.org/pdf/2309.10020.pdf)
>
> ![img](<./assets/(null)-20240401103437163-1938877.(null)#center>)

# Q1: how to learn image representations?

## Overview

![img](<./assets/(null)-20240401103437163.(null)#center>)

![img](<./assets/(null)-20240401103437191.(null)#center>)

## 改进 CLIP

### 数据层面：Data scaling up

![img](<./assets/(null)-20240401103437202.(null)#center>)

### 模型层面：Model design

- image side

  - FLIP（Scaling CLIP training via masking）：是一种改进的训练方法，用于提高 CLIP模型的训练效率。FLIP 的核心思想是在训练过程中随机遮挡图像的部分区域，只对可见的区域进行编码。
  - > Scaling language-image pre-training via masking, CVPR 2023

  - ![img](<./assets/(null)-20240401103437302.(null)#center>)

- language side

  - K-Lite: 将外部知识融入到对比学习预训练中，在 K-Lite 中，实体的维基百科定义（**knowledge**）可以与原始的图像替代文本（alt-text）一起自然地用于对比预训练。
  - > K-lite: Learning transferable visual models with external knowledge, NeurIPS 2022

  - ![img](<./assets/(null)-20240401103437199.(null)#center>)

- improved interpretability
  - STAIR（Learning Sparse Text and Image Representation in Grounded Tokens）：
    - 将图像和文本映射到**高维稀疏嵌入空间**；
    - 每个维度的值是一个非负标量，表示与该维度对应的词或标记的权重；
    - 提供了更好的性能和更清晰地识别图像和文本之间的对应关系；
  - > STAIR: Learning Sparse Text and Image Representation in Grounded Tokens, 2023

![img](<./assets/(null)-20240401103437291.(null)#center>)

- more modalities
  - ImageBind（One embedding space to bind them all）：将不同模态（在这篇论文中有7种模态）链接到一个共同空间的方法。
    - 一个共同的嵌入空间。
    - 使用冻结的预训练 CLIP。
    - 学习其他模态编码器：对于每个非图像和文本的模态，ImageBind 学习一个编码器，将该模态的数据映射到 CLIP 的嵌入空间中。

![img](<./assets/(null)-20240401103437370.(null)#center>)

### 目标函数：Objective function

- fine-grained supervision（细粒度监督）
  - FILIP：通过细粒度监督来改进视觉-语言预训练的方法。即单词-图像块（patch）对齐。
    - 双编码器结构：FILIP 仍然使用双编码器结构，即分别对图像和文本进行编码，而不是使用融合编码器。
    - 细粒度监督：FILIP 的核心在于细粒度的监督。它首先计算单词和图像块之间的相似性，然后通过最大池化（max pooling）聚合这个相似性矩阵来计算损失。这种方法允许模型学习更细粒度的图像-文本对齐。
    - 学习单词-图像块对齐：通过这种细粒度的相似性计算和损失函数，FILIP 能够学习单词和图像块之间的对齐关系。这种对齐对于可视化非常有用，因为它提供了图像和文本之间更明确的关联。
  - > FILIP: Fine-grained Interactive Language-Image Pre-Training, ICLR 2022

![img](<./assets/(null)-20240401103437379.(null)#center>)

- adding a generative branch
  - Coca：通过在预训练阶段同时利用图像-文本对和图像-标签对来学习图像和文本的表征。并增加了一个生成分支，以提升模型性能并赋予其新的能力，如图像描述和视觉问答。
    - 对比学习：CoCa 使用对比学习机制来学习图像和文本之间的对应关系，通过最大化相关图像-文本对的相似度，同时最小化不相关对的相似度。
    - 混合数据预训练：CoCa 利用混合的图像-文本对和图像-标签对进行预训练。这种混合使用使模型能够同时学习丰富的视觉概念和复杂的跨模态关系。
    - 生成分支：CoCa 在传统的对比学习框架上增加了一个生成分支。这个分支可以生成文本描述，从而增强模型的性能，并使模型能够执行图像描述和视觉问答等任务。
    - 学习从零开始的图像编码器：Coca完全从零开始训练，以更深层次地理解视觉内容。
    - ![img](<./assets/(null)-20240401103437347.(null)#center>)
  - > Coca: Contrastive captioners are image-text foundation models, 2022

## CLIP + 其他学习方法

### CLIP + Supervised Learning

Noisy label + text supervision（噪声标签+文本监督）

- UniCL ：它提供了一种原则性的方法来同时使用图像-标签和图像-文本数据。它是一种统一的对比学习（Unified Contrastive Learning）框架，旨在优化图像、文本和标签在同一空间内的表征。
  - ![img](<./assets/(null)-20240401103437620.(null)#center>)
- LiT: Locking the image encoder：使用预训练好的image encoder并将它冻结，并添加文本塔实现开放词汇表。使text model学会从image encoder的结果中读取出好的表示。
  - ![img](<./assets/(null)-20240401103437511.(null)#center>)
- MOFI：从带有噪声的实体标注图像中学习图像表示的方法。它结合了监督学习和对比学习，以提高模型在多任务环境中的性能。

  - ![img](<./assets/(null)-20240401103437540.(null)#center>)

  -

> Unified contrastive learning in image-text-label space, CVPR 2022
>
> Lit: Zero-shot transfer with locked-image text tuning, CVPR 2022
>
> MOFI: Learning Image Representations from Noisy Entity Annotated Images, 2023

### CLIP + Image-Only (Non-) Contrastive Learning

- SimCLR：对于给定的图像，应用两种不同的数据增强方法，生成两个变体。这两个变体被视为正样本对，即它们代表同一图像的不同视角，并进行对比训练。
  - ![img](<./assets/(null)-20240401103437452.(null)#center>)
- DeCLIP：
  - 自监督学习：
    1. 图像模态：使用类似 SimCLR 的方法，通过双重数据增强和对比学习来学习图像的特征表示。
    2. 文本模态：使用掩码语言模型（MLM）的方式来学习文本的特征表示。
  - 多视角监督（Multi-view supervision）：利用来自不同模态（如图像和文本）的信息作为互补的视角，通过对齐这些不同视角的表示来增强学习。
  - 最近邻监督（Nearest-neighbor supervision）：在特征空间中，通过考虑样本的最近邻来引入额外的监督信号，从而促进模型学习更加鲁棒和区分性的特征。
  - ![img](<./assets/(null)-20240401103437543.(null)#center>)
- SLIP：SLIP 的核心思想是将 SimCLR和 CLIP相结合进行模型训练。
  - ![img](<./assets/(null)-20240401103437548.(null)#center>)
- xCLIP： xCLIP = CLIP + nCLIP，是对 CLIP 的一个扩展，它利用图像自监督学习的技术来实现非对比学习，通过正则化和高维投影来确保学到的表示具有足够的区分度和鲁棒性。 引入了**锐度**（sharpness）和**平滑性**（smoothness）正则化。锐度正则化鼓励模型学习到尖锐的、区分度高的特征，而平滑性正则化则确保模型不会对数据的微小变化过度敏感。

  - ![img](<./assets/(null)-20240401103437625.(null)#center>)

  -

> A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020
>
> Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm, ICLR 2022
>
> Slip: Self-supervision meets language-image pretraining, ECCV 2022
>
> Non-Contrastive Learning Meets Language-Image Pre-Training, CVPR 2023

### Image-Only (Non-) Contrastive Learning + Masked Image Modeling

- BEiT：将自然语言处理中的预训练技术引入到图像领域，并通过视觉标记的概念建立起图像和文本之间的联系。
  - 图像分词器（Image Tokenizer）：在预训练之前，首先使用 VQ-VAE或 GAN等方法学习一个“图像分词器”。将图像分割成一系列离散的视觉标记，类似于 DALL-E 和 Parti 等图像生成模型中使用的方法。
  - 随机遮蔽与预测：在预训练阶段，随机遮蔽图像的一些区域（图像块），然后训练 BEiT 模型去预测这些被遮蔽的视觉标记。
  - 知识蒸馏：这个过程可以理解为图像分词器和 BEiT 编码器之间的知识蒸馏，但后者只能看到图像的一部分。BEiT 编码器通过学习重建遮蔽的视觉标记，从而学习到图像的丰富表示。
  - ![img](<./assets/(null)-20240401103437865.(null)#center>)
- MAE：使用像素值作为目标来训练模型。
  - 大规模遮蔽：在预训练阶段，随机遮蔽图像的大部分区域（例如 75%），只留下一小部分可见区域。
  - 编码器和遮蔽标记：将编码器应用于可见的图像块，以提取特征。在编码器之后引入遮蔽标记。
  - 像素级重建：模型的任务是预测被遮蔽区域的像素值，从而重建整个图像。
  - 适用于目标检测和分割：MAE 预训练对于目标检测和分割等任务特别有帮助，因为它强化了模型对图像局部细节和整体结构的理解。
  - ![img](<./assets/(null)-20240401103437772.(null)#center>)

MIM很适合模型微调，但不能学习全局图像表示，也不适用大规模数据缩放。

> BEiT: BERT Pre-Training of Image Transformers, ICLR 2022
>
> Masked Autoencoders Are Scalable Vision Learners, CVPR 2022

### CLIP + Masked Image Modeling （Shallow interaction）

CLIP和 MIM是两种不同的自监督学习方法，它们各自专注于不同方面的特征学习。

- MVP：使用 CLIP 的图像特征作为 MIM 训练的目标，可以捕获在MIM训练中缺失的语义，这种做法能够结合两种方法的优势，提升模型在捕捉图像语义信息方面的能力。
  - ![img](<./assets/(null)-20240401103437778.(null)#center>)
- EVA：推广了这一方法
  - ![img](<./assets/(null)-20240401103437893.(null)#center>)
- Masked Autoencoding Does Not Help Natural Language Supervision at Scale：MAE对大规模的自然语言监督没有帮助。
  - ![img](<./assets/(null)-20240401103437820.(null)#center>)
- BEiT-3：结合了 BERT和 BEiT的思想，进行多模态数据的遮蔽建模。

  - 多模态变换器（Multiway Transformer）：BEiT-3 使用一个多路变换器来处理图像/文本和图像-文本数据。这种架构允许模型同时处理并整合来自不同模态的信息。
  - 遮蔽数据建模：与单一模态的 MIM 类似，BEiT-3 对图像和文本数据进行遮蔽处理，并训练模型去预测被遮蔽的部分。这种方式促使模型学习到深层次的、跨模态的表示。
  - 共享自注意力层：BEiT-3 设计了共享的自注意力层，这些层可以处理来自不同模态的信息。这种共享机制有助于提高模型的参数效率和泛化能力。
  - 模态专家前馈网络（FFN Modality Experts）：模型包含三个前馈网络（FFN）模态专家，分别专注于处理图像、文本和图像-文本数据。这种设计使得模型能够针对不同模态的数据学习特定的特征。
  - ![img](<./assets/(null)-20240401103438009.(null)#center>)

  -

> MVP: Multimodality-guided Visual Pre-training, ECCV 2022
>
> EVA: Exploring the Limits of Masked Visual Representation Learning at Scale, CVPR 2023
>
> Masked Autoencoding Does Not Help Natural Language Supervision at Scale, CVPR 2023
>
> Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks

## 主干网络（Backbones）

![img](<./assets/(null)-20240401103437873.(null)#center>)

## 总结

### 三个高层次的原则

1. 可扩展性（Scaling）：一个好的算法应该简单，但也能够很好地扩展。这意味着算法应该能够处理大规模的数据集和模型，同时保持良好的性能。
2. 对比（Contrasting）：从 SimCLR 到 CLIP，对比学习在自监督学习中扮演着重要角色。通过比较正负样本对，模型学习到区分不同数据点的能力。
3. 遮蔽（Masking）：从 BERT 到 BEiT，遮蔽技术被广泛应用于自监督学习中。通过预测被遮蔽部分的内容，模型能够学习到数据的内在结构和特征。

### 展望

1. 进一步扩展规模：

   1. 1. 数据规模和模型规模的扩展仍是一个挑战。需要探索更有效的方法来处理和学习超大规模的数据集，以及设计更大更强大的模型架构。

2. 新的模型训练范式：

   1. 1. 寻找超越 CLIP 和 MIM 的简单且可扩展的算法。这可能涉及到新的学习机制、训练策略或模型架构。

3. 统一的图像-/区域-/像素级预训练：

   1. 1. 开发能够在不同粒度级别上对图像进行全面理解的模型。这需要模型能够同时捕捉到全局信息和局部细节。

4. 具有更灵活、可提示（promptable）接口的视觉模型：

   1. 1. 将自然语言处理中的概念（如上下文学习、思维链、提示、新兴属性等）引入到计算机视觉中。探索如何使视觉模型能够通过提示进行灵活的交互和学习。

5. 使用更创新的数据训练视觉主干网络：

   1. 1. 探索新的数据来源和类型，以解锁模型的新能力，类似于 GPT-4 展示的能力。例如，训练模型阅读整个扫描的论文，然后用几个要点来总结论文内容。

# Q2: how to extend vision models with more flexible, promptable interfaces?

## 计算机视觉领域独有的挑战

### 模型

1. 输入格式多种多样：图片，视频，多模态（附带文本/语音）
2. 不同粒度的任务：图片级，区域级，像素级
3. 不同类型的输出：空间输出，文本输出

![img](<./assets/(null)-20240401103438016.(null)#center>)

### 数据

![img](<./assets/(null)-20240401103437915.(null)#center>)

### 如何解决

![img](<./assets/(null)-20240401103437965.(null)#center>)

## Bridge Vision with Language

学习原始视觉信号与丰富语义之间的映射，并可以对各种开放词汇视觉识别任务提供动力

![img](<./assets/(null)-20240401103438207.(null)#center>)

![img](<./assets/(null)-20240401103438180.(null)#center>)

- GroupViT：通过从头开始学习图像-文本对来学习对语义相似的区域进行分组
  - 使用分组块进行自下而上的分组
  - 自上而下的图像-文本监督以实现视觉语义对齐
  - ![img](<./assets/(null)-20240401103438234.(null)#center>)
- MaskCLIP：从CLIP中提取自由的密集标签
  - 将注意力池更改为新的适应策略
  - 使用CLIP作为教师模型的伪标签掩码
  - ![img](<./assets/(null)-20240401103438117.(null)#center>)
- OpenSeg：弱监督学习，通过强制文本特征和掩码池特征之间的细粒度对齐。
  - 从图像-文本对和局部描述中学习。
  - 使用预训练的掩码建议网络。
  - ![img](<./assets/(null)-20240401103438497.(null)#center>)
- MaskCLIP (UCSD)：使用CLIP作为初始化的COCO全视分割的监督训练
  - 掩码建议网络训练
  - CLIP模型自适应
  - ![img](<./assets/(null)-20240401103438086.(null)#center>)
- 总结
  - CLIP作为基础对开放词汇训练有很大帮助
  - 将弱注释与黄金注释结合起来以获得更好的性能
  - ![img](<./assets/(null)-20240401103438233.(null)#center>)

## Unified multi-task modeling

希望开发一个统一的视觉模型，该模型可以在许多视觉任务中表现良好。

![img](<./assets/(null)-20240401103438335.(null)#center>)

两种统一化

![img](<./assets/(null)-20240401103438177.(null)#center>)

- Outputs Unification
  - 概述
    - 以一致的格式表示不同输出（一个输出里包含多个任务的结果）
    - ![img](<./assets/(null)-20240401103438596.(null)#center>)
  - UniTab and Pix2Seqv2
  - **统一词表：** 将文本和坐标都进行标记化（tokenization），并放入同一个词表中。
  - **任务前缀：** 需要一个任务前缀来确定模型正在处理的任务
  - ![img](<./assets/(null)-20240401103438974.(null)#center>)
  - Unified-IO：加入了VQVAE
    - **预训练 VQVAE：** 先单独对 VQVAE 进行预训练，让它学习有效地表示和离散化不同类型的视觉信息。
    - **序列到序列的联合训练：** 在预训练的 VQVAE 之上，以序列到序列（Sequence-to-Sequence）的方式对整个 Unified-IO 模型进行联合训练。这样做是为了让模型学习如何处理不同模态的输入，并生成符合不同任务要求的序列形式输出。
    - ![img](<./assets/(null)-20240401103438294.(null)#center>)
  - 总结
    - **优点：** 这个方式可以让一个模型通过自然语言的转换来适应各种视觉任务。
    - **局限性：**
      - 仍然需要有任务特定的步骤来把大语言模型的输出转化成可用的结构化数据。
      - 中间过程涉及自然语言，因此模型内部不同任务的交互关系可能会更难以解释。
      - 相比直接输出结构化数据的模型，VisionLLM 在整合不同任务、提升整体性能方面可能存在局限性。
- Functionality Unification
  - 概述
    - **知识迁移：** 一个模型在特定任务上学习到的能力，可以较容易地迁移到使用相似输出类型的其他任务上。
    - **模型简化：** 我们可以设计更具有通用性的模型架构，通过输出层的适配来应对不同的视觉任务。
    - **多任务学习：** 既然任务之间存在关联，就有机会设计联合学习方案，让模型同时在多个任务上训练，以相互促进，提升整体性能。
    - ![img](<./assets/(null)-20240401103438373.(null)#center>)
  - UniPerceiver-v2
    - ![img](<./assets/(null)-20240401103438644.(null)#center>)
  - X-Decoder：允许图像级和像素级的监督信号相互作用。
    - ![img](<./assets/(null)-20240401103438720.(null)#center>)

## LLM-like promptable interface

- 概述：一个通用的视觉模型应该具有相同的上下文学习能力，在不改变模型参数的情况下将输出与各种用户意图对齐。
  - **上下文记忆 （Context Memory）：** 模型需要能存储和处理之前交互轮次中的信息。这可能是通过一个外部记忆模块，或者大规模参数模型本身的记忆能力来实现。
  - **提示构建（Prompt Construction）：** 系统能够根据：1) 用户的多轮意图输入； 2) 存储的上下文信息来自动构建模型的提示。提示的内容可能包括任务描述、指示、以及之前的示例交互。
- SAM
  - ![img](<./assets/(null)-20240401103438692.(null)#center>)
- SEEM: Segment Everything Everywhere all at Once
  - ![img](<./assets/(null)-20240401103439015.(null)#center>)

# Q3: how to do image generation?

## Overview

- 4大主题
  - ![img](<./assets/(null)-20240401103439153.(null)#center>)
- 研究总结
  - ![img](<./assets/(null)-20240401103438746.(null)#center>)
- T2I模型总览
  - ![img](<./assets/(null)-20240401103439020.(null)#center>)
- DF原理

  - ![img](<./assets/(null)-20240401103438973.(null)#center>)

  - **交叉注意力机制**：K和V代表键（Key）和值（Value），它们是从文本流τ(y)投影来的，而Q是从视觉流投影来的，都具有相同的隐藏维度d。因此，Q和K之间的softmax运算产生了一个大小为(hw×d)·(N×d)转置= hw×N的注意力图M。这个交叉注意力图M表示图像和文本之间的细粒度交互，对于文本中的每一个词N，在所有空间位置hw上都有交互。然后注意力图M与V进行点积运算，以产生一个下采样/上采样块的输出。

## Spatial Controllable Generation

1. **区域绑定的文本描述**：将传统T2I模型中的图像级文本描述扩展到区域绑定的文本描述，使得开放式文本描述能够精确地操作特定的空间区域。

   1. ReCo：其核心思想是扩展文本编码器E的文本词汇，并安排不同的标记以表示绑定的文本输入。该研究通过增加一组位置标记（例如，<687>、<204>、<999>、<833>），这些位置标记与文本标记无缝混合，并作为空间修饰符，指示接下来的文本仅在指定的空间区域操作。

   2. ![img](<./assets/(null)-20240401103439045.(null)#center>)

   3. GLIGEN：采用了一种即插即用的方法，冻结原始T2I模型，并训练额外的门控自注意力层来学习新的定位技能。定位令牌携带两种类型的信息：需要在其中定位的文本词的语义表示和它们的空间配置。然后，通过新增加的门控自注意力层将这些定位令牌添加到预训练的T2I模型中，所有剩余的预训练参数都被冻结。

   4. > Reco: Region-controlled text-to-image generation.
      >
      > Gligen: Open-set grounded text-to-image generation

2. **密集空间条件**：这一类研究从边界框扩展到以2D数组形式表示的密集空间条件，如分割掩码、边缘图、深度图和关键点等。

   1. ControlNet：向文本提示添加了额外的输入条件。这个额外的条件可以是Canny边缘图、霍夫线、HED边界、素描、人体姿势图、分割掩模、深度图像、法线图或线条图，每个条件都有其独特的模型副本。
   2. ![img](<./assets/(null)-20240401103439419.(null)#center>)

   3. > ControlNet：Adding conditional control to text-to-image diffusion models.

3. **推理时指导**：前两类工作需要对T2I模型进行微调以理解扩展的空间条件。第三类技术探讨了在不对模型进行微调的情况下实现空间控制的方法。

   1. 核心思想与分类器指导（Classifier guidance）类似，即使用判别器损失来引导扩散过程。具体来说，就是在扩散过程的每一步，加入一个额外的项，这个项是由判别器计算出的目标检测损失与期望布局有关的梯度，乘以一个指导强度因子，这样可以在不额外训练模型的情况下实现空间控制。

## Text-based Editing

文本到图像编辑（Text-to-image editing）是一种从给定图像和输入的文本描述中合成新图像的技术。可以是之前从文本到图像模型生成的图像，或者是自然图像。目标是保留大部分视觉内容，只修改特定组成部分。

1. 局部图像区域变化：经典的编辑场景之一是改变局部图像区域，例如移除、更换或在某个区域内添加对象。
2. 扩展的空间编辑：语言输入描述了空间区域中期望的外观，语言也可以用作编辑指令告诉机器要做什么，例如“将图像中的对象A更改为对象B”。
3. 综合编辑系统：不是扩展单一的文本到图像（T2I）模型进行编辑，编辑系统集成了不同的专业模块，如分割模型和大型语言模型。

![img](<./assets/(null)-20240401103439401.(null)#center>)

## Text Prompts Following

文本到图像（T2I）模型可能无法遵循文本提示的问题，尤其是当图像描述变得复杂时。例如，某些名词短语可能被遗漏，属性可能应用于错误的对象，生成的图像可能有错误的对象数量、关系、风格等。

1. 推理时操作（Inference-time manipulation）:

   1. 1. 在推理阶段，设计各种方法来重新分配视觉潜在表示或图像-文本交叉注意力，以确保文本提示中的所有名词短语都在生成的图像中得到体现。

   2. StructureDiffusion：使用解析树来提取名词短语和文本提示的语言结构，然后强制模型“关注”所有提取的名词短语。这是通过修改交叉注意力机制来实现的，其中 O = M · V，M 是 softmax 交叉注意力图，V 是句子特征。$$O=\frac1{k+1}\Sigma_{i=0}^k(M\cdot V_i)$$，其中 $$V_0$$ 是句子特征 ，$$V_i$$ 是解析树中的短语特征。这种方法确保视觉流在所有识别的名词短语上保持平衡的注意力，从而促进更准确的图像生成。

      - ![img](<./assets/(null)-20240401103439330.(null)#center>)

   3. Attend-and-Excite：提出了一种正则化损失函数 $$\begin{aligned}\ell=\max_{n=1,...,N_{\mathrm{sub}}}(1-max\left.G(M_t^n)\right)\end{aligned}$$，用于增强最被忽视的主题tokens的最大注意力。该正则化损失函数使用了一个高斯核 _G_ 来平滑注意力地图，$$N_{sub}$$是主题tokens的数量。然后利用这个损失函数更新在推理时间的潜在表示 _zt_，更新公式为$$z_t^{\prime}=z_t-\alpha\nabla_{z_t}\ell$$，$$\alpha$$是步长大小。

   4. ![img](<./assets/(null)-20240401103439425.(null)#center>)

   5.

   6. > Training-free structured diffusion guidance for compositional text-to-image synthesis
      >
      > Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models.

2. 对齐调优（Alignment tuning）:

   1. 1. 学习一个额外的模型学习阶段，通常以图像-文本相似性作为奖励，使得调优后的模型能够更好地遵循文本提示。

   2. DDPO：加入强化学习

      - ![img](<./assets/(null)-20240401103439944.(null)#center>)

   3. > Training diffusion models with reinforcement learning

## Concept Customization

直接扩展T2I模型以通过图像输入理解视觉概念。

1. 单一概念定制: 从单一概念定制开始，该过程涉及测试时微调，将视觉概念的多张图片编码成新的Token嵌入。

   1. Textual Inversion：T2I 模型处理不同狗品种的四个图像，随后学习新标记的嵌入，表示为 [V]。这个 [V] 标记可以用作文本标记来表示这个特定的狗。[V] 令牌可以与其他文本描述无缝集成。文本反转通过前缀调整来学习 [V] 令牌嵌入，即冻结所有 T2I 模型的参数并训练 [V] 令牌嵌入以生成输入图像。

   2. Dreambooth：仅调整输入图像可能会导致对特定概念的 T2I 模型过度拟合的风险，为了解决这个问题，本文提出了特定于类的先验保留损失。这种方法的核心是使用预训练的 T2I 模型来生成与目标定制概念相同的类图像。然后，该模型在输入图像（使用 [V] 令牌）和模型生成的图像（没有 [V] 令牌）上联合微调。确保了模型能够在独特的“[V] 狗”和它最初训练的其他一般狗之间进行区分，同时保持其整体的T2I能力。

      - ![img](<./assets/(null)-20240401103439445.(null)#center>)

      - ![img](<./assets/(null)-20240401103439411.(null)#center>)

   3. > An image is worth one word: Personalizing text-to-image generation using textual inversion
      >
      > Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation.

2. 多概念定制: 允许扩展文本到图像模型的Token词汇表，以包括多个概念Tokens，这使得多个概念能够在生成过程中相互作用以及与剩余的视觉场景交互。

   1. Custom Diffusion

      - ![img](<./assets/(null)-20240401103439746.(null)#center>)

      -

   2. > Multi-concept customization of text-to-image diffusion.

3. 个性化测试微调的简化: 由于测试时微调要求用户为每个新概念定制T2I模型，这可能会变得相当复杂。为简化使用流程，一些研究探索了无需测试时微调的定制方法，采用统一的微调阶段来扩展T2I模型，使其接受图像条件输入。这些模型将视觉概念的图像作为额外的输入条件，并根据文本描述生成包含视觉概念的图像。

   1. SuTI：训练单个模型来模拟微调的主题特定专家，并生成以文本和主题输入图像为条件的图像。
      - ![img](<./assets/(null)-20240401103440055.(null)#center>)
   2. > Subject-driven text-to-image generation via apprenticeship learning

## 展望

- Unified image and text inputs（统一图像文本输入）
- Tuning with alignment-focused loss and rewards（聚焦于对齐的损失和奖励）
- Closed-loop of multimodal content understanding and generation（多模态内容理解和生成的闭环）

# Q4: how to train multimodal LLM?

## Image-to-Text Generative Models

### **模型架构**

- **预训练图像编码器与语言模型**：这是多模态模型的基础。图像编码器（通常为基于 CNN 或 Transformer 的结构）学习提取图像的视觉特征，而语言模型（如 GPT 系列）负责建模文本序列的内在规律。
- **连接两个模态的可训练模块**：为了融合视觉与语言信息，模型需要有专门的可训练模块。常见的有：
  - **Cross-Attentional Mechanisms**： 跨模态注意力机制让图像特征与文本相互影响，赋予模型理解图文联系的能力。
  - **Multimodal Fusion Modules**： 多模态融合模块负责将来自不同模态的信息聚合，形成统一的表示。
  - ![img](<./assets/(null)-20240401103439631.(null)#center>)

### 训练目标

- **Cross-Attended Image-to-Text Generation**：模型通过跨模态注意力，逐步生成与图像对应的文本描述。这是许多图像描述生成模型的核心目标。
- **Autoregressive loss on language output**：自回归损失用于训练语言模型的生成能力。通过最小化预测的下一个词与真实词之间的差异，使模型学习生成流畅、自然的文本序列。

## Large Multimodal Models

- LLaVA: Large Language-and-Vision Assistant
  - ![img](<./assets/(null)-20240401103439681.(null)#center>)

## 总结

![img](<./assets/(null)-20240401103439659.(null)#center>)

# Q5: how to chain vision experts with LLM?

LLM + 各种工具

## Evolution of Modeling Paradigm

![img](<./assets/(null)-20240401103439811.(null)#center>)

![img](<./assets/(null)-20240401103439840.(null)#center>)

## New Paradigm

![img](<./assets/(null)-20240401103439845.(null)#center>)

- MM-ReAct: Prompting ChatGPT for Multimodal Reasoning and Action
- ![img](<./assets/(null)-20240401103439972.(null)#center>)

- ![img](<./assets/(null)-20240401103440030.(null)#center>)
