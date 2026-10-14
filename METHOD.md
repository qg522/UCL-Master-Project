# 分类方法与推理实现

本文描述随包代码和参数实际执行的方法。模型使用观察MRI和预先生成的健康参考，对五类疾病进行分类；不包含在此包中的参考生成阶段不作为已可复现的端到端实现。

## 1. 输入与配对预处理

观察cine记为Xp，生成参考记为Xh。每例磁盘NIfTI尺寸为H×W×Z×T = 128×128×16×30。dataloader读取float32后执行`transpose(3,2,0,1)`，得到T×Z×H×W；batch后为B×30×16×128×128。

打包数据已经完成对齐：已有生成影像交换H/W；原图双线性缩放至128×128（align_corners=False），深度居中裁切或补零至16层。时间按覆盖整个周期的端点对齐网格线性采样为30相位；两路共同平移到ED开头。原图以空间变换后完整cine的min/max归一化，参考强度保持不变。生成端完整时间预处理代码未取得，时间网格是明确约定，不能当作已验证的原始生成配方。

发布的数据是这些操作的最终结果；推理不会再次缩放或按每路独立归一化。加载时检查形状、affine、有限值及测试患者清单。差异`D = Xp - Xh`在float32中直接计算，保留正负号，不取绝对值。不额外保存差异文件，减少重复数据。

## 2. 双路时空编码

观察流与差异流使用结构相同、参数独立的CineEncoder。每个patch覆盖5相位×2层×32像素×32像素，展开为10,240维，经过线性投影映射至128维。

patch数为(30/5)×(16/2)×(128/32)×(128/32)=768。每路加入可学习位置编码，随后经过2层TransformerEncoder。每层使用4个注意力头、512维前馈层、ReLU前馈激活、dropout=0、norm_first=True，整个编码器末尾另加LayerNorm。权重MLP独立使用GELU。

对应代码：`heartmirror/classification.py:CineEncoder`与`heartmirror/tokenizer.py:transformer`。

## 3. 自适应重加权

对每路编码结果Z∈R^(B×768×128)，使用独立MLP：Linear(128,128)→GELU→Linear(128,1)。在768个token维度计算softmax得到权重w，输出Z'=w⊙Z。这里是逐token缩放，不是加权求和；完整token序列继续进入融合模块。

## 4. 跨注意力融合

差异流作为Query，观察流同时作为Key和Value，执行4头MultiheadAttention。差异残差融合为F=Zd'+Attention(Q=Zd',K=Zp',V=Zp')。

将F与Zp'沿token维拼接，得到1536个token；在最前插入可学习CLS token，形成1537个token。随后通过4层相同配置的TransformerEncoder。取CLS输出，经Linear(128,5)得到logits。softmax给出五类概率，argmax得到预测。

类别顺序始终固定：NOR、DCM、HCM、MINF、RV；修改顺序会改变类别含义，不能与当前权重混用。

## 5. 训练目标的代码定义

模型代码保留objective函数，但发布入口不进行优化。focal目标为Σ_i -α_y(1-p_y)^γ log(p_y)，γ=2，未传入α时权重为1。观察及差异token平均池化后分别通过128→128投影并L2归一化，得到up和ud。

监督跨流对比使用ud·upᵀ/τ，τ=0.1，沿batch中的观察样本维计算log-softmax。相同诊断标签的观察样本均为正样本，包含同一患者；对每个差异样本平均正样本log概率，再对患者求和。总目标=L_focal+0.1 L_contrast。

随参数配置记录的优化设置为AdamW、学习率1e-4、weight_decay=0.01、batch=8。推理不调用objective，也不更新权重。本文件描述目标定义，不以配置中的shots或epochs字段推断当前权重实际训练样本数或轮数。

## 6. 推理流程

1. 读取config.json及固定类别顺序，建立full variant分类器。
2. 以weights_only=True读取纯state_dict，再strict=True加载，检查参数键和形状。
3. 调用eval及torch.inference_mode；不使用随机增强或标签指导前向。
4. 按test.csv顺序读取30例，并核对test_ids.json。
5. 执行双路编码、重加权、跨注意力及分类头。
6. 将完整概率及argmax类别写入CSV，另输出所提供分区的准确率。

默认FP32，不启用混合精度。GPU并行运算可能与CPU产生微小数值差异；比较结果应固定设备、PyTorch版本和batch配置。

## 7. 参数与边界

参数总量4,538,375；127项张量，当前存储为FP32。代码中对比投影参数保留以匹配state_dict，分类输出使用CLS路径；推理同时返回的对比嵌入不影响分类结果。

本包只支持已经对齐的原图及参考配对输入，不包括原图自动分割、临床诊断流程或实时参考生成。当前128×128/16层设置不同于论文240×240/10层设置。本包数据名称是来源清单的test分区，不由名称推断独立性，也不将运行结果声明为论文复现。
