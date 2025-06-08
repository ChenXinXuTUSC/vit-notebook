# extended.ipynb 分析

此 Markdown 文件提供了对 'extended.ipynb' Jupyter Notebook 的分析，解释了每个单元格的用途以及它们之间的联系。

## 单元格 1：导入

此单元格导入了项目所需的库，包括：

*   `os`: 用于与操作系统交互。
*   `os.path`: 用于处理文件路径。
*   `numpy`: 用于数值运算。
*   `math`: 用于数学函数。
*   `random`: 用于生成随机数。
*   `open3d`: 用于 3D 数据处理。
*   `torch`: 用于使用 PyTorch 进行深度学习。
*   `torch.nn`: 用于定义神经网络模块。
*   `torch.nn.functional`: 用于神经网络函数。
*   `torch.utils.data`: 用于处理数据集和数据加载器。
*   `matplotlib.pyplot`: 用于绘制图形。
*   `utils`: 包含实用函数的自定义模块。
*   `scipy.stats`: 用于统计函数。

这些库对于 Notebook 的其余部分至关重要，提供了数据操作、模型构建和可视化的工具。

## 单元格 2：ModelNet40 数据集

此单元格定义了 `ModelNet40` 类，这是一个用于 ModelNet40 数据集的自定义数据集类。

*   `__init__`: 初始化数据集，从指定的根目录和拆分（训练/测试）加载形状名称和数据样本。 它还设置投影轴和图像形状。
*   `__len__`: 返回数据集中的样本数。
*   `__getitem__`: 返回数据集中的单个样本，包括特征图和 ground truth 标签。
*   `add_proj_dir`: 将投影方向添加到投影轴列表中。
*   `project_list`: 将 3D 点投影到 2D 平面列表上。
*   `project`: 使用指定的轴和图像形状将 3D 点投影到 2D 平面上。

此类负责加载和预处理 ModelNet40 数据集，该数据集用于训练和测试 3D 对象识别模型。

## 单元格 3：数据集初始化和可视化

此单元格初始化用于训练和测试的 `ModelNet40` 数据集，定义图像和补丁形状，并设置投影轴。 然后，它迭代训练数据集并可视化每个视图的特征图。

*   `image_shape`: 定义投影图像的形状 (64x64)。
*   `patch_shape`: 定义用于分割图像的补丁的形状 (4x4)。
*   `proj_axes`: 定义用于将 3D 点投影到 2D 平面上的投影轴。
*   `train_dataset`: 初始化用于训练的 `ModelNet40` 数据集。
*   `valid_dataset`: 将训练数据集拆分为训练集和验证集。
*   `testt_dataset`: 初始化用于测试的 `ModelNet40` 数据集。

可视化部分迭代训练数据集的第一个样本，显示每个视图的 4 个通道。

## 单元格 4：PatchSpliter 和 PosEncoder

此单元格定义了 `PatchSpliter` 和 `PosEncoder` 类，它们用于预处理输入图像，然后再将其馈送到 Transformer 编码器中。

*   `PatchSpliter`: 将输入图像分割成补丁并将它们展平为 token。
*   `PosEncoder`: 将位置编码添加到 token。

这些类对于准备 Transformer 编码器的输入数据至关重要，Transformer 编码器用于特征提取。

## 单元格 5：MViT 模型

此单元格定义了 `MViT` 类，它是 3D 对象识别的主要模型。

*   `__init__`: 初始化模型，包括补丁分割器、位置编码器、Transformer 编码器和输出头。
*   `forward`: 定义模型的前向传递，包括将输入图像分割成补丁、添加位置编码、将 token 传递到 Transformer 编码器，以及预测类标签。

此类结合了补丁分割器、位置编码器和 Transformer 编码器，以创建一个强大的 3D 对象识别模型。

## 单元格 6：训练设置

此单元格设置训练环境，包括记录器、数据加载器、设备、损失函数、优化器和调度器。

*   `logger`: 初始化记录器以保存训练日志。
*   `ckpt_path`: 定义用于保存模型检查点的路径。
*   `train_dataloader`: 初始化训练数据集的数据加载器。
*   `valid_dataloader`: 初始化验证数据集的数据加载器。
*   `testt_dataloader`: 初始化测试数据集的数据加载器。
*   `device`: 设置训练设备（如果可用，则为 CUDA，否则为 CPU）。
*   `loss_fn`: 定义损失函数 (CrossEntropyLoss)。
*   `optimizer`: 定义优化器 (AdamW)。
*   `scheduler`: 定义学习率调度器 (ReduceLROnPlateau)。

此单元格准备训练环境以训练 MViT 模型。

## 单元格 7：训练和验证函数

此单元格定义了 `train` 和 `valid` 函数，它们用于训练和验证模型。

*   `train`: 训练模型一个 epoch，计算训练数据集的损失和准确率。
*   `valid`: 验证模型一个 epoch，计算验证数据集的损失和准确率。

这些函数对于训练和评估 MViT 模型至关重要。

## 单元格 8：训练循环

此单元格定义了 `run` 函数，它是主要的训练循环。

*   `run`: 训练模型指定的 epoch 数，保存最佳模型检查点并记录训练指标。

此函数协调训练过程，为每个 epoch 调用 `train` 和 `valid` 函数，并保存最佳模型检查点。

## 单元格 9：测试设置

此单元格设置测试环境，包括设备和模型。

*   `device`: 设置测试设备（如果可用，则为 CUDA，否则为 CPU）。
*   `model`: 初始化 MViT 模型并加载最佳模型检查点。

此单元格准备测试环境以评估 MViT 模型。

## 单元格 10：测试函数

此单元格定义了 `test` 函数，该函数用于测试模型。

*   `test`: 在测试数据集上测试模型，计算损失、准确率和混淆矩阵。

此函数在测试数据集上评估 MViT 模型并生成结果。

## 单元格 11：混淆矩阵

此单元格生成并显示混淆矩阵，该矩阵可视化模型在测试数据集上的性能。

混淆矩阵显示了每个类的正确和错误分类样本的数量，从而提供了对模型优势和劣势的深入了解。
