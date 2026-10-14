# HeartMirror ACDC 分类推理

本包包含分类推理入口、纯参数权重、模型源码、dataloader、完整推理方法说明和seed0测试分区的30例配对MRI。每类6例，类别顺序固定为NOR、DCM、HCM、MINF、RV。

## 使用

```bash
python -m pip install -r requirements.txt
python infer.py
```

默认使用可用的CUDA，否则使用CPU。可以明确选择：

```bash
python infer.py --device cpu --batch-size 1
python infer.py --device cuda --batch-size 2 --out outputs/predictions.csv
```

推理只读取`data/test.csv`，并核对`data/test_ids.json`中的患者顺序；不遍历训练或验证分区。输出每例标签、预测类别和五类概率。标签只用于输出及计算分区准确率，不参与模型前向。

| 文件 | 用途 |
| --- | --- |
| `infer.py` | 严格加载参数并执行分类推理 |
| `dataloader.py` | 读取配对MRI并计算有符号差异 |
| `weights/ACDC.pt` | 127项张量组成的纯state_dict |
| `config.json`、`classes.json` | 参数对应的配置与类别顺序 |
| `heartmirror/` | 推理所需模型及Transformer实现 |
| `METHOD.md` | 数据、结构、公式、训练目标与推理方法 |
| `data/` | 测试分区清单、30例输入和数据许可说明 |

当前权重为4,538,375个参数。本包不包含参考生成器权重，使用已提供的生成参考，不能只输入原始MRI运行完整生成流程。测试分区名称和纯参数文件均不能单独证明训练时的数据独立性；本包输出用于执行验证，不作独立泛化或论文成绩声明。

本机验证记录可见`VALIDATION.json`。源码与权重包、测试影像包分开提供；若下载两份ZIP，将测试影像包解压到本包根目录，使`data/patients/`就位。

## 数据与许可

ACDC影像按原数据条款使用，许可和引用见`data/LICENSE_TERMS.md`及`data/MANDATORY_CITATION.md`。
