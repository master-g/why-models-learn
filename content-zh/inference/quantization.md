---
title: "量化：用有限位宽近似权重、激活与 KV cache"
tags: ["why-models-learn"]
---

量化把浮点张量映射为有限位宽的整数或低比特码，再用 scale 和可选的 zero point 恢复近似值。它可以减少权重、激活或 KV cache 的存储字节数，也可以让目标硬件使用整数矩阵乘法；它同时引入舍入误差、截断误差、额外元数据和反量化路径。量化的对象、量化轴、校准数据、计算 dtype、累加 dtype 与误差验收必须共同记录，单独写一个 int4 标签不足以定义推理协议。

![量化示意图：浮点张量经过范围、scale、舍入和截断变成低比特码，再经过反量化进入计算并产生误差](/assets/inference/svg/quantization.1.svg)

## 量化先定义存储合同

### 量化对象不同，资源和误差路径不同

推理链中至少有四类可以量化的对象：

|对象|原始表示|量化后保存的内容|主要影响|
|---|---|---|---|
|权重 $W$|FP32、BF16 或 FP16|int8、int4、NF4 等码与 scale|静态权重 bytes、矩阵乘法路径、输出误差|
|激活 $x$|FP16、BF16 或 FP32|动态或静态 int8/int4 码|中间张量 bytes、校准敏感性、算子兼容性|
|KV cache|按层保存的 K/V 浮点张量|低比特 K/V 与 token/head/channel 级 scale|decode 显存、带宽、长上下文误差|
|logits 或中间状态|计算 dtype 的分数和状态|少数场景下的低精度临时值|排序、softmax、采样边界和数值稳定性|

权重可以在加载时一次量化，激活通常随 batch、请求或校准范围变化，KV cache 则随请求长度不断追加。相同的 bit 数落在不同对象上，带来的收益和风险不同。

### W4A16 说明两种 dtype

常见的 W/A 记法分别描述权重和激活的存储或计算位宽。例如：

|记法|权重路径|激活路径|需要进一步说明的字段|
|---|---|---|---|
|W16A16|FP16 或 BF16|FP16 或 BF16|累加 dtype、权重是否分片|
|W8A16|int8|FP16 或 BF16|int8 dot 的累加 dtype、scale 位置|
|W4A16|int4 或其他 4 bit code|FP16 或 BF16|打包格式、group size、反量化融合位置|
|W8A8|int8|int8|输入和权重 scale、整数累加、输出 requantize|
|W4A4|低比特权重|低比特激活|激活校准、饱和和算子覆盖范围|

W4A16 通常表示权重以 4 bit 形式存储，输入激活以 16 bit 参与计算。它不表示整个矩阵乘法只用 4 bit，也不自动说明累加器、输出投影或 KV cache 的 dtype。

### 存储 dtype 与计算 dtype 分开记录

量化权重可能在内存中是 int4，在 kernel 中解包成寄存器中的低精度值，再用 FP16、BF16 或 FP32 累加。实际合同至少包含：

|字段|需要记录的值|缺失时的影响|
|---|---|---|
|code dtype|int8、int4、NF4 或自定义 code|无法计算 payload bytes 和解包规则|
|scale dtype|FP16、BF16 或 FP32|无法计算 metadata bytes 和误差|
|compute dtype|FP16、BF16、FP8 或 INT8|无法解释 kernel 输入和输出|
|accumulator dtype|FP32、INT32 或其他类型|无法解释归约误差和溢出|
|axis / group|tensor、channel、token、group、block|无法恢复对应 scale|
|clipping rule|min-max、percentile、MSE 或固定范围|无法复现量化范围|
|zero point|是否存在以及保存方式|无法复现非对称量化|

量化配置必须和 checkpoint 一起保存。只保存一个“4 bit”字段，会丢失解码所需的 scale、分组和 zero point。

## 一个均匀量化器的合同

### 先缩放，再舍入和截断

设浮点标量为 $x$，量化码为 $q$，scale 为 $s>0$，zero point 为 $z$。一个非对称均匀量化器可以写成：

$$
q=\operatorname{clip}_{[q_{\min},q_{\max}]}\left(\operatorname{round}\left(\frac{x}{s}\right)+z\right).
$$

反量化把整数码恢复到计算域：

$$
\widehat{x}=s(q-z).
$$

这里的 $\widehat{x}$ 是近似值。原始值与近似值之间的差为：

$$
e=x-\widehat{x}.
$$

完整执行顺序是：

1. 选定量化轴和校准样本。
2. 估计 $x_{\min}$、$x_{\max}$ 或对称范围 $\alpha$。
3. 计算 scale 与 zero point。
4. 对每个值执行缩放和舍入。
5. 把超出整数码范围的结果截断到边界。
6. 按确定的布局打包并保存元数据。
7. 在计算前按相同元数据反量化，或由融合 kernel 直接使用整数码。

截断发生在舍入之后。把浮点值先截断到校准范围、再映射到码域，和直接让整数码饱和，只有在范围和边界定义一致时才具有相同语义。

### 对称量化固定中心

对称量化把 zero point 固定为 0，并让浮点零点对应整数零点。对 signed $b$ bit code，可以取：

$$
q_{\max}=2^{b-1}-1,
\qquad
s=\frac{\alpha}{q_{\max}},
\qquad
\alpha=\max_i\lvert x_i\rvert.
$$

此时：

$$
q=\operatorname{clip}_{[-q_{\max},q_{\max}]}\left(\operatorname{round}\left(\frac{x}{s}\right)\right),
\qquad
\widehat{x}=sq.
$$

对称方案的 metadata 较少，整数乘法路径简单；当正负范围差异很大或均值明显偏离 0 时，它会把一部分码空间分配给较少使用的一侧。

### 非对称量化利用完整码域

给定浮点范围 $[x_{\min},x_{\max}]$ 和整数范围 $[q_{\min},q_{\max}]$，非对称 scale 可以写成：

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}.
$$

zero point 由下式得到：

$$
z=\operatorname{round}\left(q_{\min}-\frac{x_{\min}}{s}\right).
$$

实际实现还要把 $z$ 截断到整数码范围，并重新检查端点的反量化结果。非对称量化可以更充分地利用全范围，代价是每个量化组需要保存或重建 zero point，整数矩阵乘法还要处理零点修正。

### 量化器的边界条件

|边界|合同|需要观察的结果|
|---|---|---|
|全零张量|固定 $s=1$ 或其他非零默认值|量化码保持为零，反量化仍为零|
|$x_{\min}=x_{\max}$|避免 scale 为零|输出范围、zero point 和码值确定|
|正负不对称|选择对称或非对称规则|极值误差与码空间利用率|
|恰好落在边界|明确 round half 的规则|不同语言或 kernel 是否得到相同码|
|NaN 或 Inf|在量化前拒绝或显式处理|不能静默变成普通整数码|
|超出校准范围|clip 后再量化|记录饱和比例和受影响位置|
|空 group|禁止或使用固定默认 scale|避免读取未初始化 metadata|

Python、C++ 和 GPU kernel 对 tie-breaking 的约定可能不同。量化探针必须包含正好落在半格点附近的输入，不能只测一般随机数。

## 量化误差来自舍入与截断

### 舍入误差有一个局部上界

在没有截断且 scale 固定时，最近整数舍入的误差满足：

$$
\lvert e\rvert\leq \frac{s}{2}.
$$

这个结论只约束舍入误差。若 $x$ 超出校准范围，截断误差还包括浮点值与可表达端点之间的差。设 $C$ 是没有截断的值集合，$S$ 是发生饱和的值集合，则总平方误差可以分成：

$$
\operatorname{MSE}
=
\frac{1}{n}
\left(
\sum_{i\in C}e_i^2+
\sum_{i\in S}e_i^2
\right).
$$

减小 bit 数会增大相邻码的间距。扩大范围可以减少饱和，却会让整个码域变粗。范围选择是舍入误差和截断误差之间的折中，需要在代表性校准数据上测量。

### 一个 int4 数值例子

对向量 $x=(-1.7,-0.8,-0.1,0.6,1.25)$ 使用 signed int4，取 $q_{\max}=7$。最大绝对值为 $\alpha=1.7$，所以：

$$
s=\frac{1.7}{7}=0.242857143.
$$

量化码为 $q=(-7,-3,0,2,5)$，反量化结果约为 $\widehat{x}=(-1.7,-0.728571,0,0.485714,1.214286)$。最大绝对误差为 $0.114286$，均方误差为 $0.005888$。最小值没有舍入误差，其他位置仍然携带有限的近似误差。

### 误差指标要和任务指标分层

|层级|指标|回答的问题|
|---|---|---|
|码级|最大绝对误差、均方误差、饱和比例|量化器如何近似输入张量|
|层级|输出 L2、相对误差、余弦相似度|一个算子或 block 的误差如何传播|
|模型级|logits 差异、KL、困惑度、任务准确率|模型输出是否保持|
|系统级|显存、带宽、TTFT、ITL、吞吐|资源收益是否实现|

低 MSE 不能单独证明生成质量保持。一个很小的 logit 扰动也可能改变 top-1 边界；一个较大的中间层误差则可能在后续归一化和残差加法中被重新缩放。报告必须同时保留码级、模型级和系统级结果。

## 量化粒度决定 scale 是否适合数据

### 四种常见粒度

同一个 scale 覆盖的元素越多，metadata 越少；同一组数据的范围差异越大，scale 越难同时适合所有元素。

|粒度|一个 scale 覆盖|优势|影响|
|---|---|---|---|
|per-tensor|整个张量|metadata 最少、kernel 简单|离群值会放大量化间距|
|per-channel|输出 channel 或指定通道|保留不同 channel 的动态范围|scale 需要沿矩阵乘法轴广播|
|per-group|固定大小的连续权重组|平衡误差与 metadata|group size 和打包布局必须固定|
|per-token / per-block|每个 token 或物理 block|适合激活、KV 或局部变化|运行时 scale 计算和读写更频繁|

“per-channel”必须标记轴。对线性层 $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，常见 weight-only 方案按输出行保存 scale，使每个输出 channel 有自己的范围；按输入列保存 scale 会改变 scale 与输入向量的组合方式，不能仅更换 axis 名称。

### group size 是存储合同的一部分

若权重元素数量为 $N$，每组有 $G$ 个元素，每个 scale 占 $s_{\mathrm{scale}}$ bytes，则 scale 元数据近似为：

$$
M_{\mathrm{scale}}
=
\left\lceil\frac{N}{G}\right\rceil s_{\mathrm{scale}}.
$$

若每组还保存 zero point，则：

$$
M_{\mathrm{metadata}}
=
\left\lceil\frac{N}{G}\right\rceil
\left(s_{\mathrm{scale}}+s_{\mathrm{zero}}\right)
+
M_{\mathrm{index}}.
$$

group size 越小，局部范围拟合通常越好，scale 数量、读取和解包成本也越高。group 的边界必须与打包顺序一致；对同一组码使用相邻 group 的 scale，会在边界处制造系统性输出误差。

### 离群值改变 scale

如果一个 channel 中只有一个极大值，per-tensor 或大 group scale 会由该值决定，其他普通值的码分辨率下降。常见处理包括：

|处理|作用|需要验证|
|---|---|---|
|缩小校准范围|减少离群值对 scale 的支配|饱和比例、下游误差|
|per-channel 或 per-group|把离群值限制在局部范围|metadata、kernel 支持|
|离群值保留高精度|只对少量列或权重例外保存|分支路径和额外 bytes|
|平滑激活范围|在权重与激活之间重新分配尺度|变换前后输出等价性|
|混合 bit|给敏感层或 channel 更高 bit|层级收益和部署覆盖|

处理离群值会改变量化算法的定义。必须保存离群值判定阈值、例外列表和复原路径，不能只报告平均 bit 数。

## 权重量化：先减静态 bytes，再核对计算路径

### weight-only 的存储账

忽略 padding 和索引开销时，$N$ 个权重用 $b$ bit 保存的 payload 近似为：

$$
M_{\mathrm{payload}}=N\frac{b}{8}.
$$

加入 scale、zero point 和 metadata 后：

$$
M_{\mathrm{weight}}
\approx
N\frac{b}{8}
+
M_{\mathrm{scale}}
+
M_{\mathrm{zero}}
+
M_{\mathrm{metadata}}.
$$

对 $N=1{,}048{,}576$、group size 为 128、每个 scale 使用 FP16 的例子：

|格式|payload bytes|scale bytes|合计 bytes|
|---|---:|---:|---:|
|FP16|2,097,152|0|2,097,152|
|INT8|1,048,576|0|1,048,576|
|INT4 payload|524,288|0|524,288|
|INT4 + group scale|524,288|16,384|540,672|

INT4 加入 scale 后仍比 FP16 少存储，但收益取决于 group size、scale dtype、zero point、padding、权重分片和运行时 workspace。生产报告应使用实际 checkpoint 的字节数，不要只用参数量乘 bit 数。

### 反量化位置影响延迟

线性层可以写成：

$$
y=Wx+b.
$$

量化权重反量化后计算：

$$
\widehat{y}=\widehat{W}x+b.
$$

如果每个输出 channel 使用 $s_j$，并采用对称量化，则：

$$
\widehat{W}_{j,k}=s_jq_{j,k}.
$$

kernel 可以在读取 q 时乘 scale，也可以先把一小块 q 反量化到寄存器或 shared memory。两种实现的数学近似相同，访存、寄存器压力、融合范围和累加误差可能不同。

|路径|权重存储|主要操作|需要核对|
|---|---|---|---|
|预先反量化|低比特 checkpoint、运行前 FP16 权重|加载时产生完整浮点权重|峰值显存、加载时间|
|块内反量化|低比特权重和 group scale|矩阵乘法 tile 内转换|scale 读取、寄存器和带宽|
|整数乘加|低比特权重、低比特激活|整数 dot、scale 合并|累加范围、零点修正、输出转换|
|混合路径|不同层或矩阵不同格式|按算子选择 kernel|路由、fallback、质量差异|

权重存储减少不等于模型计算自动变快。若加载时完整反量化，推理峰值可能重新接近 FP16；若 kernel 不支持目标形状，运行时可能回退到较慢实现。

### 量化算法名称不能替代合同

GPTQ、AWQ、SmoothQuant 和其他方法可以使用不同的校准目标、误差补偿或权重—激活变换。它们仍需要明确：

- 量化前后的对象；
- bit、group size、axis 和 codebook；
- calibration 数据与随机状态；
- scale、zero point 和例外权重的存储；
- 计算、累加和输出 dtype；
- 目标 kernel 及其 fallback；
- 层级误差与端到端质量。

算法名称用于定位实现，字段合同用于复现结果。

## 激活量化：动态范围跟着输入变化

### 静态与动态激活量化

静态激活量化在校准阶段估计每层范围，部署时直接读取固定 scale。动态激活量化在每个 batch、token 或 block 运行时估计范围，再量化当前输入。

|方案|scale 来源|运行时成本|主要风险|
|---|---|---|---|
|静态|离线 calibration 数据|低|分布漂移导致饱和|
|动态|当前 batch 或 token|需要统计范围|额外归约和 scale 不稳定|
|混合|固定范围加局部修正|中等|路径较多、复现字段增加|

校准数据应覆盖 prompt 长度、语言、格式、领域、batch 形状和目标任务。只用平均激活范围会漏掉长尾；只用极端样本又会增大普通 token 的量化间距。

### 激活量化会影响后续非线性

设量化激活为 $\widehat{x}=x+e_x$，非线性为 $\phi$。线性近似给出：

$$
\phi(\widehat{x})-\phi(x)
\approx
\phi'(x)e_x.
$$

在 GELU、SiLU、softmax 或归一化附近，导数、指数和归约会把小的输入误差重新分配到不同坐标。若激活接近 ReLU 或 mask 边界，少量误差也可能改变分支。

激活量化的验收要包含：

|检查|固定条件|结果|
|---|---|---|
|范围|同一校准集、同一 token mask|scale、饱和比例|
|算子|同一非线性、归一化和 residual 顺序|逐层最大误差|
|批处理|同一有效长度和 padding|动态 scale 是否受 batch 影响|
|分布|短 prompt、长 prompt、异常输入|长尾误差|
|部署|目标 device 和 kernel|输出、延迟和 fallback|

## KV cache 量化：把历史误差带入每个 decode step

KV cache 在 decode 中被反复读取。对 $L$ 层、batch $B$、历史长度 $P$、K/V 宽度 $D_{kv}$、每个元素 $b$ bit 的 cache，忽略 metadata 时：

$$
M_{\mathrm{KV}}
\approx
2LBPD_{kv}\frac{b}{8}.
$$

如果采用 GQA，$D_{kv}=h_{kv}d_h$；不能使用 query width $D$ 代替 K/V width。以 $L=32$、$B=2$、$P=4096$、$D_{kv}=1024$ 为例：

|KV dtype|每元素 bit|主 payload bytes|
|---|---:|---:|
|FP16|16|1,073,741,824|
|INT8|8|536,870,912|
|INT4|4|268,435,456|

scale、zero point、block table、padding 和 allocator 对齐仍需加到实际占用中。

### KV scale 的轴影响读取误差

KV cache 可以按 token、head、channel、group 或 block 保存 scale。不同轴的行为不同：

|粒度|scale 变化|适合的目标|主要代价|
|---|---|---|---|
|per-token|每个历史 token 一组|token 间范围差异大|每步读取更多 scale|
|per-head|每个 K/V head 一组|head 统计差异明显|局部离群值仍影响范围|
|per-channel|每个通道一组|通道尺度稳定|metadata 和广播|
|per-group|固定 channel group|误差与 bytes 折中|group layout|
|per-block|分页 block 一组|paged cache 实现|block 边界和重用状态|

K 和 V 的分布可以不同，必须说明是否共用 scale。若 cache block 被移动、复用或淘汰，scale 的所有权要与 K/V block 同步。

### cache 量化改变的是读取值

每个 decode step 的 query、K/V 读取和 attention reduction 仍按原有 shape 执行。量化 cache 只把历史 K/V 的浮点值替换为近似值：

$$
\widehat{K}=K+E_K,
\qquad
\widehat{V}=V+E_V.
$$

attention 输出的变化同时受到 score 变化和 value 读取变化影响。必须分别测量：

- K 量化对 score 和 attention weight 的影响；
- V 量化对输出向量的影响；
- scale 读取和 dequantize 的时间；
- 长历史、不同 batch 和不同 GQA 配置；
- 首 token 与后续 token 的质量差异。

KV cache bytes 下降可以缓解 memory bandwidth，但实际 ITL 仍由 kernel、batch、cache reuse、通信和 sampling 共同决定。

## 量化与矩阵乘法的计算合同

### 理论 MAC 仍由 shape 决定

矩阵的逻辑维度没有因为存储码变短而改变。若 $W\in\mathbb{R}^{m\times n}$、$x\in\mathbb{R}^{n}$，输出仍有 $m$ 个坐标，每个坐标需要 $n$ 个乘加项：

$$
\operatorname{MACs}=mn.
$$

量化改变每个元素的读取、转换和实际硬件指令。INT8 dot、INT4 unpack、scale multiplication、zero point correction 和 requantize 都要列入实现账本。

### 对称与非对称整数乘法

对称量化可以把浮点乘积近似写成：

$$
Wx
\approx
(s_Wq_W)(s_xq_x)
=
s_Ws_x(q_Wq_x).
$$

如果两侧都有 zero point，则展开后还会出现交叉项和常数项。kernel 可能预先计算某些行和或列和，也可能在 tile 内处理。实现必须验证 zero point 的符号、广播轴和累加范围。

### 量化收益的三层证据

|证据层|需要测量|可支持的结论|
|---|---|---|
|存储|checkpoint bytes、峰值显存、scale metadata|存储减少了多少|
|计算|kernel 类型、带宽、FLOPs、dequant 时间|目标算子是否使用低比特路径|
|模型|logits、困惑度、任务分数、生成稳定性|输出质量变化是否可接受|

只有存储层数据时，只能报告存储变化。只有离线延迟时，还要确认 kernel、batch、序列长度、warmup 和输出协议保持一致。

## 误差如何传播到 logits 和选择器

### 线性层的保守上界

对 $y=Wx+b$ 和近似值 $\widehat{y}=\widehat{W}\widehat{x}+\widehat{b}$，一个粗略的范数上界为：

$$
\lVert\widehat{y}-y\rVert
\leq
\lVert\widehat{W}-W\rVert\lVert x\rVert
+
\lVert\widehat{W}\rVert\lVert\widehat{x}-x\rVert
+
\lVert\widehat{b}-b\rVert.
$$

这个上界通常较松，但它把权重误差、输入误差和偏置误差分开。多层模型还会重复应用局部放大和归一化，不能从单层 MSE 直接推断最终质量。

### logit margin 决定 argmax 稳定性

设最高和第二高 logit 的差值为：

$$
\Delta=\ell_{(1)}-\ell_{(2)}.
$$

若量化扰动使两个位置的相对变化超过 $\Delta$，greedy 的 top-1 token 可能改变。温度不会改变有限 logits 的排序，但 top-k、top-p、重复惩罚、mask 和采样都在不同阶段读取分数。量化误差必须放在完整的 logit processor 链中测量。

### 生成质量要按协议核验

|协议|最小核验|
|---|---|
|greedy|固定输入、tokenizer、tie-breaking，比较逐步 token|
|temperature sampling|固定 request-level RNG，比较分布和重复统计|
|top-k/top-p|比较候选集合、截断边界和重新归一化|
|beam search|比较 parent index、累计 score 和最终序列|
|speculative decoding|比较 target verification、接受率和 correction token|
|KV cache|比较 prefill 后与逐步 decode 的 logits 差异|

量化误差可能只在长上下文、稀有 token、EOS 边界或某个特定 batch 下出现。短 prompt 的平均准确率不能覆盖这些边界。

## 运行方法

下面的探针使用最近整数舍入和对称 signed int4，计算向量误差、per-tensor 与 per-channel 的差异、权重存储账以及 GQA KV cache 在不同 bit 数下的主 payload。代码明确采用 Python 的 round 规则；生产 kernel 需要替换为目标实现的 tie-breaking 和打包逻辑。

```python
from math import prod

def quantize_symmetric(values, bits):
    qmax = 2 ** (bits - 1) - 1
    alpha = max(abs(value) for value in values)
    scale = alpha / qmax if alpha else 1.0
    codes = [
        max(-qmax, min(qmax, int(round(value / scale))))
        for value in values
    ]
    dequantized = [scale * code for code in codes]
    return scale, codes, dequantized

def mse(values, reference):
    return sum((value - ref) ** 2 for value, ref in zip(values, reference)) / len(values)

x = [-1.7, -0.8, -0.1, 0.6, 1.25]
scale, codes, dequantized = quantize_symmetric(x, 4)
errors = [value - ref for value, ref in zip(x, dequantized)]
print("symmetric_int4_scale=", f"{scale:.9f}")
print("symmetric_int4_q=", codes)
print("symmetric_int4_dequant=", [round(value, 6) for value in dequantized])
print("symmetric_int4_max_abs_error=", f"{max(abs(value) for value in errors):.6f}")
print("symmetric_int4_mse=", f"{mse(x, dequantized):.6f}")

weights = [[0.9, -0.7, 0.2], [0.08, -0.12, 0.04]]
flat_weights = [value for row in weights for value in row]
_, _, tensor_dequantized = quantize_symmetric(flat_weights, 4)
channel_dequantized = []
for row in weights:
    _, _, row_dequantized = quantize_symmetric(row, 4)
    channel_dequantized.extend(row_dequantized)
print("per_tensor_mse=", f"{mse(flat_weights, tensor_dequantized):.6f}")
print("per_channel_mse=", f"{mse(flat_weights, channel_dequantized):.6f}")
print(
    "small_channel_max_abs_error=",
    f"{max(abs(a - b) for a, b in zip(weights[1], channel_dequantized[3:])):.6f}",
)

N = 1_048_576
group_size = 128
fp16_bytes = N * 2
int8_bytes = N
int4_payload_bytes = N * 4 // 8
int4_scale_bytes = (N // group_size) * 2
print("memory_fp16_bytes=", fp16_bytes)
print("memory_int8_bytes=", int8_bytes)
print("memory_int4_payload_bytes=", int4_payload_bytes)
print("memory_int4_scale_bytes=", int4_scale_bytes)
print("memory_int4_total_bytes=", int4_payload_bytes + int4_scale_bytes)
print(
    "memory_int4_reduction_vs_fp16=",
    f"{fp16_bytes / (int4_payload_bytes + int4_scale_bytes):.6f}",
)

layers, batch, history, kv_width = 32, 2, 4096, 1024
for bits in (16, 8, 4):
    cache_bytes = 2 * layers * batch * history * kv_width * bits // 8
    print(f"kv_bits_{bits}_bytes=", cache_bytes)
```

运行输出：

```text
symmetric_int4_scale= 0.242857143
symmetric_int4_q= [-7, -3, 0, 2, 5]
symmetric_int4_dequant= [-1.7, -0.728571, 0.0, 0.485714, 1.214286]
symmetric_int4_max_abs_error= 0.114286
symmetric_int4_mse= 0.005888
per_tensor_mse= 0.001761
per_channel_mse= 0.001099
small_channel_max_abs_error= 0.005714
memory_fp16_bytes= 2097152
memory_int8_bytes= 1048576
memory_int4_payload_bytes= 524288
memory_int4_scale_bytes= 16384
memory_int4_total_bytes= 540672
memory_int4_reduction_vs_fp16= 3.878788
kv_bits_16_bytes= 1073741824
kv_bits_8_bytes= 536870912
kv_bits_4_bytes= 268435456
```

探针说明了三件事。对称 int4 的 scale 由最大绝对值决定；per-channel 可以显著降低小动态范围 channel 的误差；int4 的 scale metadata 会增加 payload 之外的 bytes。KV cache 从 FP16 降到 INT8 或 INT4 时，主 payload 分别减半或减到四分之一，但实际占用仍要加 scale、padding 和 block metadata。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|模型加载成功但显存没有下降|启动时完整反量化、scale workspace、复制了 FP16 权重|记录每个生命周期的 bytes|
|量化后速度更慢|kernel fallback、解包开销、batch 太小|记录实际 kernel 和带宽|
|短样本正常、长文本退化|激活或 KV cache 长尾范围不足|按长度分桶比较 logits 和任务分数|
|不同设备结果不一致|round、zero point、累加 dtype 或打包顺序|运行固定码值和边界探针|
|per-channel 更好但吞吐下降|scale 广播和非融合路径|比较 scale 读取、寄存器和 kernel 时间|
|KV cache 只量化 K 或只量化 V|K/V scale 所有权或布局未同步|逐层、逐 head 比较 dequant 结果|
|checkpoint 无法复现|缺 group size、axis、scale dtype 或 calibration 版本|保存完整量化配置和 hash|

### 最小量化审计表

|检查项|应保存|验收问题|
|---|---|---|
|对象|weight、activation、K、V、logits|实际量化了哪些张量|
|码域|bit、signed/unsigned、codebook|码值如何解释|
|范围|min-max、percentile、MSE、clip|哪些样本决定 scale|
|粒度|tensor、channel、group、token、block|scale 沿哪个轴广播|
|元数据|scale、zero point、index、padding|bytes 和所有权是否完整|
|计算|compute、accumulator、output dtype|kernel 是否按同一合同运行|
|误差|逐层、logits、任务和生成协议|质量变化发生在哪一层|
|系统|显存、带宽、TTFT、ITL、吞吐|资源收益是否实现|
|复现|checkpoint、calibration、kernel、版本、seed|另一台设备能否复现|

量化的目标是改变表示和资源分配，而不是把所有计算都替换为整数。每个量化方案都应从浮点基线开始，固定对象、粒度、码域、元数据、kernel 和评估协议，再分别报告存储收益、计算收益与输出误差。只有这三组证据同时存在，低比特配置才具有可解释性。

## 相关词条

- [推理数学](../inference/inference-math/)：把权重、激活、KV cache、MAC 和延迟放进同一资源账。
- [混合精度训练](../training-nn/mixed-precision/)：区分训练期 dtype、loss scaling、累加器和主权重。
- [参数量总账](../transformer-components/parameter-count/)：计算参数存储、激活、optimizer state 和 KV cache 的边界。
- [KV cache](../inference/kv-cache/)：展开 K/V shape、分页布局、追加、淘汰和 cache 字节数。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：说明 $D_{kv}=h_{kv}d_h$ 如何改变 K/V cache 宽度。
- [推理](../inference/inference/)：固定 checkpoint、dtype、量化配置、batch 和端到端生成协议。
- [softmax 函数](../neurons-and-activations/softmax/)：说明 logit 误差如何进入归一化概率和采样。
- [FlashAttention](../attention/flash-attention/)：比较低精度存储、tile、workspace 和 attention kernel 的关系。
