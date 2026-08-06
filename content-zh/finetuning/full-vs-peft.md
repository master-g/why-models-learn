---
title: "全量微调与参数高效微调：可训练参数、资源和能力边界"
tags: ["why-models-learn"]
---

全量微调与参数高效微调（parameter-efficient fine-tuning，PEFT）都从同一个预训练 checkpoint 出发，用监督或其他下游目标更新模型，但它们允许参数变化的空间不同。全量微调更新原模型参数，PEFT 冻结基础权重，只更新 adapter、低秩因子、prefix 或其他小参数集合。两者的显存、optimizer state、梯度通信、checkpoint 形态和能力回归边界不能用“训练步数相同”代替。

![全量微调与参数高效微调示意图：同一个冻结基础 checkpoint 分别进入全参数更新或小参数 adapter 更新，两条路径拥有不同的显存、通信、checkpoint 和回归合同](/assets/finetuning/svg/full-vs-peft.1.svg)

## 先写出两种参数更新空间

### 全量微调直接移动 checkpoint

设预训练参数向量为 $\theta_0\)，监督微调目标为 $\mathcal L$。全量微调允许：

$$
\theta=\theta_0+\Delta\theta,
\qquad
\Delta\theta\in\mathbb R^{P}.
$$

这里 $P$ 是可训练参数数量。每个参与前向的 trainable tensor 都可以获得梯度、保存 optimizer state，并在 checkpoint 中产生新的参数值。冻结某些 embedding、norm 或输出头后，实验已经不再是“全量”，而是一个有明确冻结集合的部分更新方案。

### PEFT 把变化限制在小参数集合

PEFT 保留基础参数 $\theta_0$，引入可训练参数 $\phi$：

$$
\theta(\phi)=F(\theta_0,\phi),
\qquad
\phi\in\mathbb R^{P_{\mathrm{peft}}},
\qquad
P_{\mathrm{peft}}\ll P.
$$

对 LoRA 这类增量方法，某个线性层的权重可以写成：

$$
W'=W_0+\Delta W,
\qquad
\Delta W=BA.
$$

若 $W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，低秩因子 $B$ 和 $A$ 的秩为 $r$，新增参数数量为：

$$
P_{\mathrm{adapter}}
=
r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

这条公式只计算 adapter 参数，不包括冻结基础权重、激活、workspace 和 optimizer state。LoRA 的初始化、缩放、目标模块和合并方式在后续词条中单独展开。

### 训练空间决定了“能改变什么”

|方案|可训练对象|参数变化空间|典型 checkpoint|
|---|---|---|---|
|full fine-tuning|基础模型参数|接近 $\mathbb R^P$|新的完整权重|
|adapter PEFT|adapter 参数|由插入位置和结构决定|基础权重加 adapter|
|LoRA|低秩矩阵 $A,B$|低秩增量子空间|基础权重加低秩因子|
|prefix/prompt tuning|虚拟 token 或 prefix 参数|输入或 attention 条件子空间|基础权重加 prefix|
|bias/scale-only|偏置或通道尺度|坐标级小参数集合|基础权重加小张量|

“参数少”只说明更新维度小，不说明功能一定弱。更新空间、目标模块、数据质量、训练预算和基础 checkpoint 共同决定结果。

## 显存账要区分冻结权重与训练状态

### 训练显存至少拆成五项

对参数数量 $P$，权重存储字节数 $b_w$，梯度字节数 $b_g$，optimizer state 每个参数的字节数 $b_o$，全量训练的参数状态近似为：

$$
M_{\mathrm{full}}
\approx
P(b_w+b_g+b_o)+M_{\mathrm{act}}+M_{\mathrm{work}}.
$$

PEFT 仍然需要把基础权重放入前向，但只有 $P_{\mathrm{peft}}$ 个参数需要梯度和 optimizer state：

$$
M_{\mathrm{peft}}
\approx
Pb_w
+P_{\mathrm{peft}}(b_w+b_g+b_o)
+M_{\mathrm{act}}+M_{\mathrm{work}}.
$$

两式中的激活和 workspace 可能受 batch、序列长度、checkpointing、kernel 和目标模块影响。PEFT 不会自动消除 activation memory，也不会让长上下文的 attention 或 KV cache 变成常数。

### Adam state 会放大全量训练的差异

如果权重和梯度使用 BF16 或 FP16，各占 2 bytes，Adam 的一阶和二阶矩使用 FP32，各占 4 bytes，则每个 trainable parameter 还需要约 8 bytes 的 optimizer state：

$$
b_o=4+4=8.
$$

全量训练每个参数的权重、梯度和 optimizer state 约为 $2+2+8=12$ bytes，不含临时激活和碎片。PEFT 的基础权重仍占 $2P$ bytes，但额外的 10 bytes 只按 $P_{\mathrm{peft}}$ 计算。

以一个 $P=7\times10^9$ 的模型和 $P_{\mathrm{peft}}=2\times10^7$ 的 adapter 为例：

|项目|全量微调|PEFT|
|---|---:|---:|
|基础权重 BF16|$2P$|$2P$|
|可训练权重|$2P$|$2P_{\mathrm{peft}}$|
|梯度|$2P$|$2P_{\mathrm{peft}}$|
|Adam moments|$8P$|$8P_{\mathrm{peft}}$|
|训练状态近似|$12P$|$2P+12P_{\mathrm{peft}}$|
|激活与 workspace|由 batch、长度和 kernel 决定|同样需要单独测量|

“PEFT 显存很小”通常只描述 trainable state 或 adapter state。若基础模型本来就接近设备容量，冻结权重仍可能成为主要显存项。

### 梯度 checkpoint 与权重 checkpoint 不是同一件事

全量 checkpoint 通常需要保存新的基础权重、optimizer state、scheduler state、随机状态和 tokenizer/template 版本。PEFT checkpoint 可以只保存 adapter，但部署时还需要准确的 base model 标识：

$$
\operatorname{load}(\text{base hash},\text{adapter hash},\text{config}).
$$

缺少 base hash 时，adapter 可能被加载到形状相同但参数不同的模型上。它可以通过文件格式检查，却不再代表训练时的函数。

## 梯度通信和并行训练也会变化

### DDP 主要同步 trainable gradient

在数据并行训练中，每轮通常需要对 trainable gradient 做同步。忽略压缩、分桶和通信拓扑时，单轮梯度 payload 近似为：

$$
C_{\mathrm{grad}}
\approx
P_{\mathrm{train}}b_g.
$$

全量微调取 $P_{\mathrm{train}}=P$，PEFT 取 $P_{\mathrm{train}}=P_{\mathrm{peft}}$。这会影响通信时间、梯度 bucket、网络带宽和多机扩展，但不会改变每张卡加载基础权重的需求。

### 参数分片策略要和方案一起报告

ZeRO、FSDP 或其他 sharding 方案可能分片参数、梯度和 optimizer state。比较全量与 PEFT 时，应说明：

|组件|全量微调|PEFT 要核对|
|---|---|---|
|parameters|基础参数是否分片|基础权重是否冻结和分片|
|gradients|所有 trainable 参数|只同步 adapter 梯度还是仍保留额外路径|
|optimizer|全模型 state|adapter state 是否单独分组|
|communication|全参数梯度 bucket|adapter bucket、频率和大小|
|activation|由前向图决定|冻结权重仍参与前向和反向所需路径|
|checkpoint|shard 合并与恢复|base、adapter、配置是否成对保存|

只看到显存下降，不能推导出训练吞吐按同一比例提高。前向矩阵乘法、激活保存、通信等待和 I/O 仍可能成为瓶颈。

## 选择 PEFT 参数的位置

### 目标模块是能力预算的一部分

常见的目标模块包括 attention 的 Q/K/V/O 投影、MLP 投影、embedding、LM head 和 norm。不同目标模块的可训练参数数量和函数影响不同：

|目标位置|能影响的路径|需要观察|
|---|---|---|
|Q/K projection|token 间读取权重和位置条件|检索、格式和长上下文|
|V/O projection|读取内容与输出混合|答案内容和表示变换|
|MLP projection|token 内部非线性特征|知识、风格和任务组合|
|embedding|输入 token 表示|新 token、拼写和领域词|
|LM head|词表 logits|输出词分布和格式 token|
|norm/bias|尺度和偏移|稳定性、容量和回归|

target modules 不能只写成“attention + MLP”。应保存具体模块名称、层范围、rank、alpha、dropout、初始化和是否共享参数。

### trainable ratio 要按参数数量计算

PEFT 的 trainable ratio 为：

$$
\rho_{\mathrm{train}}
=
\frac{P_{\mathrm{peft}}}{P}.
$$

如果只报告 rank，不报告目标矩阵的输入输出维度、层数和数量，无法得到 $P_{\mathrm{peft}}$。同一个 rank 在不同 hidden size、层数和 target module 集合下代表不同的更新容量。

### 小 adapter 也可能改变整个输出

adapter 插入在每层 hidden path 上，经过多层残差、attention 和 MLP 后，少量参数的变化可以影响整段输出。参数数量与函数影响不是一一对应关系。比较不同 PEFT 配置时，要同时观察：

- trainable parameter count；
- target layer 和 target module；
- adapter 输出相对基础分支的尺度；
- 训练 token 和有效监督 token；
- base、adapter 和 merged checkpoint 的回归。

## checkpoint、合并和部署

### 未合并 adapter 有额外前向路径

以 LoRA 为例，线性层输出可以写成：

$$
y=xW_0^\mathsf T+xA^\mathsf TB^\mathsf T.
$$

未合并时需要计算基础分支和增量分支。若基础权重固定，可以缓存或优化部分路径，但不能假定新增分支没有成本。合并后：

$$
W_{\mathrm{merged}}=W_0+BA,
$$

推理可以回到一个普通线性层的路径，但必须保存合并前的 base hash、adapter hash、dtype、缩放和合并顺序。

### 多 adapter 部署需要独立的资源账

同一基础模型挂载多个任务 adapter 时，可以按请求切换 adapter，也可以合并某一个任务。需要记录：

|部署方式|优势|影响|
|---|---|---|
|未合并单 adapter|切换快、保留独立文件|前向增加增量路径|
|多 adapter 服务|共享基础权重|batch 按 adapter 分桶，缓存更复杂|
|合并后部署|推理图简单|切换和回滚需要重新合并|
|全量独立 checkpoint|任务之间隔离|每份权重和显存都更大|

adapter 的 prefix cache、KV cache 和 tokenizer/template 仍要遵循请求合同。换 adapter 后不能复用语义不一致的前缀状态。

### 基础模型版本是 adapter 的输入

adapter 训练实际学习的是某个 base 函数附近的更新。即使两个 base checkpoint 的参数 shape 相同，词表、位置机制、归一化、权重值或 tokenizer 不同，也可能导致 adapter 结果变化。发布 adapter 时至少绑定：

- base model 名称和内容 hash；
- tokenizer 和 special token；
- chat template；
- target modules、rank、scale 和 dtype；
- 训练数据版本和 loss mask；
- 训练框架、optimizer 和 checkpoint 状态；
- merged 与 unmerged 的验证结果。

## 质量、容量与灾难性遗忘

### 全量微调的自由度更高

全量微调可以调整基础模型的所有可训练层，适合任务分布和基础能力都需要明显变化的场景。自由度提高也意味着：

- 小数据集可能快速过拟合；
- 原有语言分布和知识能力可能回退；
- 学习率或 token 预算过大时，参数偏移难以回滚；
- 多任务混合时，任务之间的梯度可能互相覆盖。

这些结果不是“全量一定更强”的证明，而是说明更新空间更大，回归成本也更高。

### PEFT 的约束可以保留基础能力，但不是质量保证

PEFT 把更新限制在较小的参数集合，通常更容易保留基础模型的通用行为，也更容易为不同任务保存独立 adapter。它可能在以下情况下容量不足：

- 新任务需要大范围改变表示；
- 目标模块没有覆盖需要改变的路径；
- rank 或 adapter width 太小；
- 数据分布与 base 的能力差异较大；
- 多个行为必须同时组合。

“参数少所以不遗忘”和“参数多所以能学会”都不是可直接使用的结论。应在同一 base、同一数据、同一训练 token 和同一评测协议下比较。

### 回归矩阵要包含 base、full 和 PEFT

|评测维度|base|full SFT|PEFT|
|---|---|---|---|
|目标任务|未适配基线|目标 loss 和任务分数|目标 loss 和任务分数|
|原始能力|能力底线|是否回退|是否回退|
|格式协议|base 模板行为|schema、EOS、工具|schema、EOS、工具|
|长度与位置|短、中、长|截断和长答案|截断和长答案|
|鲁棒性|改写、噪声|未见输入|未见输入|
|资源|基础推理账|训练与部署账|训练、adapter 和部署账|

若只比较 full 和 PEFT 的目标分数，却没有 base 回归，无法判断某个方案是在获得目标能力，还是在牺牲原有能力。

## 优化设置不能只复制全量方案

### 学习率和参数尺度不同

全量微调的更新作用在所有基础参数，PEFT 的更新作用在 adapter 参数和其插入路径。两者的初始化尺度、梯度范数、weight decay、学习率和 warmup 不应默认相同。至少记录：

|设置|全量方案|PEFT 方案|
|---|---|---|
|trainable groups|基础参数分组|adapter 与可选 bias 分组|
|learning rate|基础模型更新尺度|adapter 初始化和增量尺度|
|weight decay|按基础参数配置|按 adapter 参数配置|
|gradient clipping|全参数梯度范数|adapter 梯度范数和共享路径|
|scheduler|有效 token 和 update|有效 token 和 update|
|freeze state|哪些参数冻结|base、norm、head 是否冻结|

如果优化器把冻结参数放进 param group，虽不一定产生梯度，仍可能制造额外 state 或让配置含义不清。训练启动时应打印 trainable 参数表和 optimizer group。

### 训练预算要按有效 token 比较

设每个 update 的有效 target token 为 $N_{\mathrm{update}}$，训练 update 数为 $U$，则监督 token 预算为：

$$
N_{\mathrm{train}}=\sum_{u=1}^{U}N_{\mathrm{update},u}.
$$

full 与 PEFT 使用同一个样本数，不代表在相同的优化预算下比较；响应长度、mask、packing 和截断都可能改变 $N_{\mathrm{train}}$。

## 一个可运行的资源账本探针

下面的探针比较一个 7B 规模参数账和一个 2,000 万参数 adapter 的训练状态，并计算 32 层、rank 8、四个 4096×4096 目标矩阵的 LoRA 参数量。它只核对数量级，不包含激活、workspace、分片、量化和显存碎片。

```python
P = 7_000_000_000
Pa = 20_000_000
weight_bytes = 2
grad_bytes = 2
adam_state_bytes = 8
full_bytes = P * (weight_bytes + grad_bytes + adam_state_bytes)
peft_bytes = P * weight_bytes + Pa * (weight_bytes + grad_bytes + adam_state_bytes)

print("full_weight_gib=", f"{P * weight_bytes / 2**30:.3f}")
print("full_train_state_gib=", f"{full_bytes / 2**30:.3f}")
print("peft_train_state_gib=", f"{peft_bytes / 2**30:.3f}")
print("trainable_ratio_percent=", f"{100 * Pa / P:.6f}")
print("optimizer_state_reduction=", f"{(P * adam_state_bytes) / (Pa * adam_state_bytes):.1f}")
print("total_memory_reduction=", f"{full_bytes / peft_bytes:.3f}")
print("full_gradient_transfer_gib=", f"{P * grad_bytes / 2**30:.3f}")
print("peft_gradient_transfer_mib=", f"{Pa * grad_bytes / 2**20:.3f}")

layers = 32
rank = 8
dimension = 4096
target_matrices = 4
adapter_parameters_per_matrix = rank * (dimension + dimension)
lora_parameters = layers * target_matrices * adapter_parameters_per_matrix
print("lora_parameters=", lora_parameters)
print("lora_adapter_fp16_mib=", f"{lora_parameters * weight_bytes / 2**20:.3f}")
```

运行输出：

```text
full_weight_gib= 13.039
full_train_state_gib= 78.231
peft_train_state_gib= 13.262
trainable_ratio_percent= 0.285714
optimizer_state_reduction= 350.0
total_memory_reduction= 5.899
full_gradient_transfer_gib= 13.039
peft_gradient_transfer_mib= 38.147
lora_parameters= 8388608
lora_adapter_fp16_mib= 16.000
```

在这个假设下，PEFT 仍需约 13.039 GiB 的基础 BF16 权重，但训练状态约为 13.262 GiB；全量训练状态约为 78.231 GiB。adapter 的 trainable ratio 为 0.285714%，Adam state 的参数数量相差 350 倍，梯度通信从约 13.039 GiB 降到 38.147 MiB。LoRA 参数计算得到 8,388,608 个参数，FP16 adapter payload 为 16 MiB。

这些数值不包含 activation、workspace、通信 buffer、显存碎片和框架额外副本。它们用于检查公式的分母与字节单位；部署前仍需要实际运行时峰值。

## 运行方法

将上一个 Python 代码块保存为 full-vs-peft-ledger.py，再运行 python3 full-vs-peft-ledger.py。修改模型参数量、adapter 参数量、dtype 或 optimizer state 字节数后，应同步更新正文的比较表。

接入真实训练时，还要从参数对象统计 trainable count、每个 optimizer group 的 state、梯度 bucket、基础权重副本、激活峰值和 checkpoint 文件大小。纸面账本不能替代设备实测。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|PEFT 显存仍然不足|基础权重、激活或 workspace 占主要部分|分项记录峰值显存|
|adapter 训练没有效果|target module、冻结状态或 mask 错误|打印 trainable 参数和梯度范数|
|全量训练 loss 降但原能力回退|学习率、token 预算或数据混合过大|运行 base/full 回归矩阵|
|adapter 加载后结果异常|base hash、tokenizer 或 template 不一致|校验加载合同并对比 base|
|未合并推理变慢|增量分支和多 adapter 分桶|测量基础/增量 kernel 时间|
|多任务 adapter 互相污染|batch、cache 或合并顺序混用|按 adapter ID 记录请求状态|
|参数比率看起来很小但文件很大|包含 optimizer、checkpoint 或重复 base|分别统计 adapter、optimizer 和 base 文件|
|不同 PEFT 配置无法比较|rank、目标层、scale 或有效 token 不同|保存完整 adapter config|
|通信没有按比例下降|仍同步冻结参数或通信由激活主导|检查 gradient bucket 内容和通信 trace|
|量化后 adapter 失效|量化 dtype、反量化路径或合并顺序变化|分别测试 merged、unmerged 和目标 kernel|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|base|模型内容 hash、tokenizer、template|adapter 是否绑定正确基础函数|
|trainable set|参数名、shape、dtype、数量|实际更新了哪些参数|
|optimizer|group、state、LR、weight decay|冻结参数是否进入 state|
|memory|权重、梯度、state、激活、workspace|显存账是否完整|
|communication|梯度 bucket、分片、带宽、时间|同步对象是否只有 trainable 参数|
|adapter|方法、rank、target module、scale|更新空间能否复现|
|checkpoint|base、adapter、optimizer、配置和合并产物|能否恢复或回滚|
|quality|base、full、PEFT 的同协议结果|能力变化发生在哪里|
|deployment|merged/unmerged、batch、cache、延迟|训练方案是否适合运行时|

全量微调与 PEFT 的主要差别不是“一个大、一个小”，而是可训练参数空间、状态账本和 checkpoint 合同不同。结论必须同时给出 trainable parameter count、有效 token 预算、资源分项和 base 回归；单一目标分数不能决定方案。

## 相关词条

- [监督微调](../finetuning/sft/)：固定示范数据、target mask 和 next-token 损失。
- [指令数据](../finetuning/instruction-data/)：记录任务结构、来源、质量、覆盖、去重和切分。
- [LoRA](../finetuning/lora/)：展开低秩增量的初始化、缩放、目标模块和合并。
- [QLoRA](../finetuning/qlora/)：讨论量化基础权重与低秩 adapter 的组合资源账。
- [参数量](../transformer-components/parameter-count/)：核对基础模型、层、投影和存储字节。
- [优化器](../training-nn/optimizers/)：说明 optimizer state、参数分组和更新规则。
- [混合精度](../training-nn/mixed-precision/)：分析权重、梯度、累加器和 optimizer state 的 dtype。
- [分布式训练](../pretraining/distributed-training/)：展开参数、梯度、状态分片和通信。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：评估微调后原有能力的变化。
- [训练稳定性](../pretraining/training-stability/)：检查学习率、梯度、精度和 checkpoint 恢复。
