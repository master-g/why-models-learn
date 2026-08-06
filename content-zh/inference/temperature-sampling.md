---
title: "温度采样：用温度改变下一 token 的随机分布"
tags: ["why-models-learn"]
---

温度采样先把 logits 除以正温度，再从得到的 categorical 分布中随机抽取下一个 token。温度较低时分布更集中，温度较高时分布更平坦；采样本身还取决于随机数生成器、候选 mask、logit 处理器和停止规则。它不是把 logits 乘一个展示参数，也不是“温度越高就一定生成更长文本”。本篇推导温度分布、稳定 softmax、逆 CDF 采样、随机种子和 batch 顺序影响，再处理 EOS、重排、top-k/top-p 组合与复现审计。

![温度采样示意图：同一组 logits 在低、中、高温度下形成不同 categorical 分布，随后由随机数和累计分布函数选择 token](/assets/inference/svg/temperature-sampling.1.svg)

## 先固定采样分布

### logits 经过正温度缩放

设词表大小为 $V$，当前 logits 为：

$$
z_t\in\mathbb R^V.
$$

给定温度 $T>0$，温度 logits 为：

$$
\widetilde z_t(v)
=
\frac{z_t(v)}{T}.
$$

对应的 categorical 概率为：

$$
p_T(v)
=
\frac{\exp\left(z_t(v)/T\right)}
{\sum_{j=0}^{V-1}\exp\left(z_t(j)/T\right)}.
$$

下一 token 不是取最大值，而是从 $p_T$ 中抽样：

$$
y_t\sim\operatorname{Categorical}(p_T).
$$

抽样得到的 $y_t$ 进入下一步的条件上下文。即使温度和 logits 固定，两次抽样也可能得到不同 token；同一个随机种子和完全相同的 RNG 消耗顺序可以复现同一序列。

### 温度只改变相对差异

对任意两个 token $a,b$，概率比为：

$$
\frac{p_T(a)}{p_T(b)}
=
\exp\left(
\frac{z_t(a)-z_t(b)}{T}
\right).
$$

$T$ 越小，logit 差异被放大，较高 logit 的相对概率增长；$T$ 越大，差异被压缩，分布更接近均匀。温度不会改变有限正温度下的 logit 排序：

$$
z_t(a)>z_t(b)
\Longrightarrow
p_T(a)>p_T(b).
$$

它改变的是抽样频率，不是候选的排序。若最终仍执行 argmax，正温度不会改变无 tie 的选择；只有从分布采样时，温度才会改变输出随机性。[贪心解码](../inference/greedy-decoding/)固定了 argmax 这条确定性路径。

### 三个极限说明方向

若 logits 有唯一最大值 $v^\star$：

$$
\lim_{T\to0^+}p_T(v^\star)=1.
$$

若词表大小为 $V$，则：

$$
\lim_{T\to\infty}p_T(v)=\frac1V.
$$

这两个极限不等于工程参数的直接取值。实现中的 $T=0$ 通常是 greedy 特判，不能执行除以零；很大的有限温度会接近均匀，但仍受浮点精度、候选 mask 和词表大小影响。

## softmax 要先数值稳定

### 直接取指数会溢出

直接计算 $\exp(z_t(v)/T)$，当 logits 或 $1/T$ 较大时可能溢出。令：

$$
m=\max_v\frac{z_t(v)}{T}.
$$

稳定 softmax 写成：

$$
p_T(v)
=
\frac{\exp\left(z_t(v)/T-m\right)}
{\sum_j\exp\left(z_t(j)/T-m\right)}.
$$

减去同一个最大值不改变概率，因为分子分母同时乘了 $\exp(-m)$。稳定 log-sum-exp 为：

$$
\operatorname{LSE}(a)
=
m+\log\sum_j\exp(a_j-m),
\qquad
m=\max_j a_j.
$$

温度分布的 log probability 可以写为：

$$
\log p_T(v)
=
\frac{z_t(v)}{T}
-
\operatorname{LSE}\left(\frac{z_t}{T}\right).
$$

### mask 之后再归一化

若当前状态只允许候选集合 $\mathcal V_t$，应先屏蔽不允许 token：

$$
z_t^{\mathrm{mask}}(v)
=
\begin{cases}
z_t(v),&v\in\mathcal V_t,\\
-\infty,&v\notin\mathcal V_t.
\end{cases}
$$

再对允许候选执行温度缩放和归一化：

$$
p_T(v)
=
\operatorname{softmax}
\left(
z_t^{\mathrm{mask}}/T
\right)_v.
$$

如果先在完整词表上归一化，再删除禁止 token 而不重新归一化，剩余概率总和小于 1，不能直接作为 categorical 分布。若所有候选都被屏蔽，服务必须显式报告约束失败或使用约定的回退 token。

### 温度和 logit processor 的顺序要写清楚

一种常见路径是：

$$
z^{\mathrm{raw}}
\rightarrow
\text{repetition penalty}
\rightarrow
\text{constraint mask}
\rightarrow
\text{temperature}
\rightarrow
\text{softmax}
\rightarrow
\text{sample}.
$$

也有实现把温度放在某些 processor 之前，或者对已经截断的候选重新归一化。非线性 processor 与温度不一定可交换。例如，先 top-k 再温度和先温度再 top-k 的候选集合可能不同。配置报告必须记录每个变换的顺序；只写 temperature 数值不够。

## categorical 采样如何使用随机数

### 累计分布函数把区间分给 token

给定概率向量 $p_0,\ldots,p_{V-1}$，累计分布函数为：

$$
c_k
=
\sum_{j=0}^{k}p_j.
$$

在理想实数下，$c_{V-1}=1$。取一个均匀随机数：

$$
u\sim\operatorname{Uniform}[0,1),
$$

选择最小的 $k$ 使：

$$
u<c_k.
$$

这就是逆 CDF 采样。实现需要固定：

- token 的排列顺序；
- 概率是否在采样前重新归一化；
- 区间端点使用 $<$ 还是 $\le$；
- $u=0$ 和接近 1 的边界；
- 概率累加的 dtype 和顺序；
- RNG 的算法、状态和消耗次数。

只要其中一项变化，接近累计边界的随机数就可能选择不同 token。

### CDF 采样的一个数字例子

设三个 token 的概率为：

$$
p=(0.50,0.30,0.20).
$$

累计概率为：

$$
c=(0.50,0.80,1.00).
$$

随机数区间可以写成：

|随机数区间|选择 token|
|---|---|
|$0\le u<0.50$|token 0|
|$0.50\le u<0.80$|token 1|
|$0.80\le u<1.00$|token 2|

如果 $u=0.79$，选择 token 1；如果 $u=0.81$，选择 token 2。采样器不需要生成一个连续的文本值，它只需要将随机数映射到词表候选的离散区间。

### multinomial、CDF 和 alias table 的结果条件

不同采样实现可以使用 sequential CDF、vectorized multinomial、alias table 或硬件 kernel。只要它们使用相同概率、相同 RNG 状态和相同边界规则，分布应当一致；给定同一个 seed 时，逐 token 的具体序列未必一致，因为 RNG 消耗方式可能不同。

因此要区分两个复现目标：

|复现目标|需要达到的条件|
|---|---|
|分布复现|多次采样的频率与目标 $p_T$ 一致|
|序列复现|RNG、算法、候选顺序、kernel 和消耗顺序都一致|
|跨设备位级复现|还要固定 dtype、归约、采样实现和硬件路径|

“相同 seed”只在明确 RNG 对象、调用顺序和采样实现后才有意义。

## 温度改变熵和尾部概率

### 熵衡量分布的不确定性

categorical 分布的 Shannon entropy 为：

$$
H(p_T)
=
-\sum_{v=0}^{V-1}
p_T(v)\log p_T(v).
$$

在固定 logits 且没有候选截断时，升高温度通常使分布更平坦、熵更高；降低温度通常使质量集中到高分 token、熵更低。若 logits 完全相同，任意正温度的分布都均匀，熵保持为 $\log V$。

熵是分布指标，不是输出长度指标。高温度可能让某一步选择了 EOS，也可能让模型避开 EOS；生成长度还受 EOS logit、stop sequence、上下文和 max new tokens 共同影响。

### 一个固定 logits 的温度比较

取：

$$
z=(2,1,0).
$$

标准库探针得到：

|温度|$p_T(0)$|$p_T(1)$|$p_T(2)$|最大概率|
|---:|---:|---:|---:|---:|
|0.5|0.866813|0.117310|0.015876|0.866813|
|1.0|0.665241|0.244728|0.090031|0.665241|
|2.0|0.506480|0.307196|0.186324|0.506480|

最大 token 始终是 token 0，但抽到 token 1 和 token 2 的频率随温度上升。温度只描述分布变化；实际样本还取决于随机数和后续历史。

### 低温度不是“更正确”

低温度提高高 logit token 的采样概率，但模型的高 logit 仍可能是错误、重复或格式不合法的 token。低温度也可能放大一个微小的 logit 差异：

$$
z(a)=1.00,\qquad z(b)=0.99.
$$

在很低温度下，$a$ 和 $b$ 的概率比为 $\exp(0.01/T)$，排序差异会被显著放大。若这个差异来自数值误差、训练噪声或不稳定 processor，低温度可能让输出更敏感，而不是更可靠。

### 高温度不是均匀随机

有限温度仍保留 logit 差异。只有 $T\to\infty$ 才趋向均匀。高温度会增加低分 token 的概率，但不会让所有 token 等概率。若同时使用 top-k 或 top-p，尾部还会被截断，最终分布可能仍然集中在一个小候选集。

## RNG、seed 和 batch 顺序

### 随机状态属于请求协议

一次可复核的采样请求至少要固定：

|字段|含义|
|---|---|
|seed|初始随机状态的来源|
|RNG algorithm|伪随机数生成算法和版本|
|generator scope|每个请求、每个 batch 还是全局 generator|
|consumption order|按请求、按 beam、按 token 的调用顺序|
|candidate order|token ID 到概率数组的排列|
|sample dtype|随机数和 CDF 的精度|
|resume state|checkpoint 或重试时是否恢复 RNG 状态|

如果两个请求共用一个 generator，新增请求或不同的 EOS 时刻都会改变后续 RNG 消耗，使其他请求的 token 序列改变。每个请求独立 generator 可以减少这种耦合，但服务仍需记录 generator 的初始和结束状态。

### batch 重新排序会改变随机数分配

假设 batch 中有 A、B 两条请求，服务按 A、B 的顺序各消耗一个随机数。若连续 batching 把顺序改为 B、A，而 generator 是全局共享，A 和 B 可能交换随机数。即使每条请求的 logits 和 seed 没变，输出也可能不同。

因此需要区分：

- **请求级随机性**：给每个 request 独立 RNG，重排 batch 不改变其随机流；
- **batch 级随机性**：按 batch tensor 顺序消耗 RNG，吞吐调度会影响序列；
- **设备级随机性**：kernel 内部并行采样，调用顺序和线程布局也可能影响结果。

如果用户要求同 seed 得到同一文本，应选择请求级或明确的 token 级 RNG 协议，并测试 batch 加入、取消和重排。

### 随机重复运行要比较分布

在固定 $p_T$ 下重复采样 $N$ 次，token $v$ 的计数 $n_v$ 近似满足：

$$
\frac{n_v}{N}\approx p_T(v).
$$

二项近似的标准误差为：

$$
\operatorname{SE}(\widehat p_v)
=
\sqrt{\frac{\widehat p_v(1-\widehat p_v)}{N}}.
$$

这不是证明某一次输出正确，而是检查采样实现是否从目标分布取样。低概率 token 需要较大的 $N$ 才能区分实现偏差和抽样噪声。

## 与 greedy、beam 和候选截断的边界

### temperature sampling 与 greedy

两者都从当前 logits 出发，但最终操作不同：

|规则|最终操作|随机数|是否保留历史分支|
|---|---|---|---|
|greedy|argmax|不需要|不保留|
|temperature sampling|categorical sample|需要|只保留已采样路径|
|beam|累计 score + top-k|通常不需要|保留固定数量|
|temperature + top-k/top-p|截断后 categorical sample|需要|只保留已采样路径|

正温度后执行 argmax 仍是 greedy；只有调用 categorical sampler 才是 sampling。配置中的 temperature 字段不能单独决定执行规则。

### 先温度再 top-k 与先 top-k 再温度

设 top-k 根据分数选择候选集合。因为正温度保持 logits 排序，在没有其他数值或 processor 差异时，先温度再 top-k 与先 top-k 再温度的候选集合相同；但归一化和实现精度仍需固定。

对于 top-p，候选集合依赖累计概率：

$$
\mathcal V_p
=
\left\{
v_1,\ldots,v_m
\right\},
\qquad
\sum_{i=1}^{m}p_T(v_i)\ge p.
$$

温度改变概率比，因此会改变达到 top-p 阈值所需的候选数量。先 top-p 再温度与先温度再 top-p 通常不是同一规则。[Top-k 与 Top-p](../inference/top-k-top-p/)应单独记录排序、阈值和归一化顺序。

### sampling 不等于 beam 的随机版

temperature sampling 每步只抽一个 token，未抽中的分支丢弃；beam 保留多个 parent，并对累计 score 做排序。若对每条 beam 先 sampling，再按累计分数选 beam，算法同时包含随机分支和束剪枝，不能称为普通 temperature sampling 或普通 beam。[束搜索](../inference/beam-search/)固定了确定性候选扩展和 cache reorder。

## KV cache 与采样状态同步

### 采样出的 token 进入下一次 decode

对 decoder-only 模型，采样器返回 token ID 后：

1. 将 token 追加到请求的逻辑序列；
2. 以新的 position ID 执行下一次 decode；
3. 将新位置的 K/V 写入 cache；
4. 更新生成计数和 stop 状态；
5. 用新 logits 继续采样，或在 EOS 时结束。

采样器本身不保存 K/V。随机状态和 KV cache 是两份不同的运行时状态：RNG 决定选择哪个 token，cache 保存该 token 进入模型后产生的历史 K/V。[KV cache](../inference/kv-cache/)说明了追加位置、batch slot 和 cache offset。

### 采样失败不能静默写入 cache

以下情况需要显式处理：

- 概率总和不是有限正数；
- 所有候选被 mask；
- CDF 没有覆盖随机数；
- token ID 超出词表；
- RNG 状态读取失败；
- tokenizer 无法解码返回 token；
- cache slot 或 position offset 不匹配。

如果采样器回退到 token 0，却继续把它写进 cache，后续输出会把一个服务层异常伪装成模型生成。日志应包含 raw logits、processed logits、候选集合、概率和、随机数、选择结果和回退原因。

## 运行方法

下面的标准库探针计算同一组 logits 在三种温度下的概率，并用固定 seed 的 Python RNG 重复采样。它只验证 softmax、CDF 和随机状态，不运行神经网络。

```python
from math import exp
import random


def temperature_probabilities(logits, temperature):
    scaled = [value / temperature for value in logits]
    offset = max(scaled)
    weights = [exp(value - offset) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def sample_from_cdf(probabilities, uniform_value):
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if uniform_value < cumulative:
            return index
    return len(probabilities) - 1


logits = [2.0, 1.0, 0.0]
for temperature in (0.5, 1.0, 2.0):
    probabilities = temperature_probabilities(logits, temperature)
    print(
        "temperature=",
        temperature,
        "probabilities=",
        [round(value, 6) for value in probabilities],
    )

probabilities = temperature_probabilities(logits, 1.0)
for seed in (7, 7, 11):
    rng = random.Random(seed)
    draws = [
        sample_from_cdf(probabilities, rng.random())
        for _ in range(5)
    ]
    print("seed=", seed, "draws=", draws)
```

运行输出为：

```text
temperature= 0.5 probabilities= [0.866813, 0.11731, 0.015876]
temperature= 1.0 probabilities= [0.665241, 0.244728, 0.090031]
temperature= 2.0 probabilities= [0.50648, 0.307196, 0.186324]
seed= 7 draws= [0, 0, 0, 0, 0]
seed= 7 draws= [0, 0, 0, 0, 0]
seed= 11 draws= [0, 0, 2, 0, 0]
```

同一个 seed 和相同的 CDF 消耗顺序给出相同 token 序列；seed=11 的序列不同。温度改变每个候选的概率区间，因而也改变相同随机数落入的 token。

## 失效模式和审计方法

### 直接对 raw logits 取样

logits 不是概率，可能为负，也不保证总和为 1。必须先按温度缩放、候选 mask 和稳定 softmax 得到有限的 categorical 分布。审计时检查概率总和、最小值、最大值和是否存在 NaN。

### softmax 溢出或下溢

不减最大值直接 exp 会得到 inf；温度很低时，尾部权重可能下溢为 0。稳定计算可以接受极小概率变为 0，但要记录 dtype 和目标精度。若最大候选也不是有限数，应先失败而不是继续采样。

### mask 后没有重新归一化

删除不允许 token 后，剩余概率和可能小于 1。CDF 最后一个边界会小于随机数上界，采样器被迫回退到最后一个 token。正确做法是对允许候选重新归一化，并记录候选顺序。

### CDF 边界不一致

$u=c_k$ 时使用 $<$ 或 $\le$ 会选择不同 token。浮点累加顺序也会让接近边界的 $u$ 改变结果。实现和参考脚本必须使用同一边界协议，并用手工设置的边界随机数测试。

### seed 相同但 RNG 消耗不同

batch 顺序、EOS 时刻、top-p 候选数量和并行 kernel 都可能改变随机数消耗。只记录 seed 无法复现。记录 RNG algorithm、generator scope、每步调用次数和请求重排事件。

### 把温度当作长度控制器

温度改变 token 分布，不直接规定生成长度。EOS、stop sequence、context limit、max new tokens 和格式约束共同决定停止。比较长度时必须固定这些条件。

### 用温度后 argmax 却称为 sampling

温度缩放后执行 argmax 没有随机性，也不会得到温度改变后的样本频率。检查最终调用的是 argmax 还是 categorical sampler，并记录 RNG 是否被消费。

### 先后顺序未记录

temperature、repetition penalty、mask、top-k、top-p 和 renormalization 的顺序会改变分布。离线复现要导出每个 processor 前后的 top-k、候选数、概率总和和最终 CDF。

### batch 共享 RNG 造成请求耦合

一个请求加入或取消会改变全局 RNG 的后续状态，使其他请求输出改变。需要独立 request generator，或接受并记录 batch 级随机协议。不要把吞吐调度变化误判为模型变化。

### 采样结果与 cache 不同步

选择出的 token ID、position、request slot 和 K/V cache 必须属于同一个请求。用两个请求的可识别 prompt 做交叉 batch 测试，检查采样 token 是否写入正确 cache。

### 一份最小温度采样审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|温度|$T>0$，$T=0$ 使用明确 greedy 特判|除零、配置分支|
|输入分数|raw 或 processed logits 的来源固定|processor 顺序、mask|
|数值稳定|softmax 先减最大值，概率有限|overflow、underflow、dtype|
|候选集合|mask 后候选非空|EOS、语法约束、bad words|
|归一化|概率和在容差内等于 1|删除 token 后重归一化|
|CDF|token 顺序和 $<$ 边界固定|累加顺序、u 边界|
|RNG|seed、算法、scope、消耗次数固定|batch 重排、EOS、kernel|
|温度效果|只改变采样分布，不改变正温度排序|误用 argmax|
|截断组合|top-k/top-p 顺序和阈值固定|temperature 前后、renormalization|
|KV cache|采样 token 写入正确 request 和 position|slot、offset、append|
|停止|EOS、stop、max tokens、timeout 分开记录|长度口径、返回文本|
|复现|保留每步概率摘要、随机数和 token ID|只保存最终字符串|

温度采样的输出是随机变量。可复核性不要求每个 seed 在所有硬件上都产生同一字符串，但必须先声明复现目标：是概率分布、请求级随机序列，还是跨设备位级序列。目标不同，所需的 RNG、dtype、kernel 和调度约束不同。

## 相关词条

- [推理](../inference/inference/)：固定生成请求、停止条件、batch 和性能协议。
- [贪心解码](../inference/greedy-decoding/)：对比从 logits 取 argmax 的确定性路径。
- [束搜索](../inference/beam-search/)：对比保留多个累计分数候选的有限搜索。
- [KV cache](../inference/kv-cache/)：说明采样 token 如何进入增量 decode 和历史 K/V。
- [Top-k 与 Top-p](../inference/top-k-top-p/)：处理候选截断、累计概率和重新归一化。
- [重复惩罚](../inference/repetition-penalty/)：说明历史 token 如何修改采样前 logits。
- [Softmax](../neurons-and-activations/softmax/)：推导 logits 到 categorical 概率的归一化。
- [采样](../probability/sampling/)：提供随机变量、CDF 和抽样的基础定义。
