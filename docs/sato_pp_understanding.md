# COMPASS 理解笔记

本文记录当前对本项目中 COMPASS 的理解，重点解释它和原始 T2M-GPT 的关系、图中各模块的含义、preserve/alter pair 的作用，以及训练过程到底改了什么。

## 一句话总结

COMPASS 不是替换 T2M-GPT 的新 backbone，而是在 T2M-GPT 的文本到动作生成器上加入一套语义扰动监督。

核心思想是：

- 保留 T2M-GPT 的 VQ-VAE + autoregressive Text2Motion Transformer 主干。
- 构造 preserve/alter 文本对。
- 通过 motion-conditioned attribution 找出哪些文本 token 真正影响动作生成。
- 对 preserve pair 约束动作相关语义表示保持一致。
- 对 alter pair 约束被编辑语义的 attribution 发生变化，未编辑部分保持稳定。
- 用 variance/covariance 正则防止表示坍缩。
- 用 teacher consistency 防止 fine-tuning 破坏原始生成能力。

## 和原始 T2M-GPT 的关系

原始 T2M-GPT 的主流程是：

```text
text
 -> CLIP text encoder
 -> Text2Motion Transformer
 -> motion tokens
 -> VQ-VAE decoder
 -> generated motion
```

训练目标主要是 motion token 的交叉熵：

```text
L_gen = CE(predicted motion tokens, target motion tokens)
```

COMPASS 基本沿用这条生成链路。它没有大改：

- VQ-VAE 动作 token 表示。
- autoregressive motion decoder。
- CLIP 文本条件输入。
- 基础 motion token generation loss。
- 常规 T2M 评估方式。

它主要改的是训练数据组织和训练目标：

```text
原始 T2M-GPT:
  caption x + motion tokens y
  -> L_gen

COMPASS:
  caption_orig x
  caption_pair x+ 或 x-
  motion tokens y
  relation = preserve / alter
  edit_mask m
  -> L_gen + L_teach + L_preserve 或 L_alter
```

所以从实验流程看，COMPASS 更像是在 pretrained T2M-GPT Transformer 上继续 fine-tune；从算法结构看，它是在同一个训练循环里多了 pair 输入、归因计算和额外 loss。

## Pair Construction

COMPASS 使用两类文本对。

### Preserve pair

Preserve pair 表示文本表达变化，但动作语义不变。

例子：

```text
x  = A person walks forward and raises both arms.
x+ = The person lifts up both arms while walking ahead.
```

虽然词汇和句式变了，但语义仍然是：

```text
walk forward
raise both arms
```

Preserve pair 的目标是：句式变了，模型内部理解到的动作相关语义应该尽量一致。

### Alter pair

Alter pair 表示只改变一个语义槽位，使动作语义发生变化。

例子：

```text
x  = A person turns left and walks slowly.
x- = A person turns right and walks slowly.
```

这里只有 `left -> right` 被改了。Alter pair 的目标是：模型要对这个关键语义变化敏感，同时不要让没变的部分一起乱变。

图里的 edit mask `m` 表示哪些 token 被编辑：

```text
0 = non-edited
1 = edited
```

## Motion-Conditioned Attribution

COMPASS 的核心不是直接拿普通句向量做对齐，而是先计算每个文本 token 对动作生成有多重要。

图中定义 early-step log-likelihood：

```text
s_AR(x, y) = 1/K * sum_{t=1..K} log p_theta(y_t | y_<t, x)
```

它表示：文本 `x` 对真实动作 `y` 的前 K 步 motion token 解释得有多好。

然后对每个文本 token feature `h_i` 做 Grad x Input attribution：

```text
a_i = | d s_AR(x, y) / d h_i * h_i |_1
```

直观理解：

如果某个词对生成当前动作很重要，那么这个词的表示轻微变化会明显影响 motion likelihood，因此它的 attribution 会比较高。

最后归一化：

```text
alpha_i = softmax(ReLU(a_i) / tau)
```

得到每个 token 的动作相关权重 `alpha_i`。

## Attribution-Induced Semantic Representation

图中第 4 块把 attribution 变成一个动作相关语义表示。

先投影 token feature：

```text
phi(h_i)
```

再用 attribution 权重加权求和：

```text
u(x) = sum_i alpha_i * phi(h_i)
```

这里的 `u(x)` 不是普通文本 embedding，而是 motion-conditioned semantic representation：

```text
文本中真正影响动作生成的语义表示。
```

然后经过 predictor：

```text
z(x) = q(u(x))
```

这个 `z(x)` 主要用于 preserve branch 的对齐。

## Preserve Branch: 约束语义不变

Preserve branch 处理 `x` 和 `x+`。

它的核心目标是：

```text
u(x) 和 u(x+) 应该表示同一个动作语义。
```

图中使用 alignment loss：

```text
L_align = -cos(z, sg(z+))
```

`sg` 是 stop-gradient，类似 SimSiam 的做法。

优化这个 loss 等价于让余弦相似度变大：

```text
z(x) ≈ z(x+)
```

这就是“文本句式变了但动作尽量一致”的主要约束。

### 为什么还需要防坍缩

如果只做 alignment，模型可能学到一个无意义解：

```text
所有文本都映射成同一个向量。
```

这样所有 preserve pair 都相同，但表示已经没有区分能力。这就是坍缩。

因此 preserve branch 还加入两个正则。

### Variance regularization

```text
L_var = mean_j max(0, gamma - sigma_j)
```

它要求 batch 内表示在每个维度上都有足够标准差。

直观理解：

```text
不同动作文本的表示不能全部挤成同一个点。
```

### Covariance regularization

`L_cov` 惩罚表示维度之间的相关性，鼓励不同维度表达不同信息。

直观理解：

```text
表示不能所有维度都在重复同一种信息。
```

Preserve branch 总目标是：

```text
L_preserve = lambda_align * L_align
           + lambda_var   * L_var
           + lambda_cov   * L_cov
```

所以防止表示坍缩的主要机制是 preserve branch 里的 variance/covariance regularization，而不是 alter pair 本身。

## 第 3/4 步和 Preserve Loss 的关系

第 3/4 步和 preserve loss 的关系可以理解为：

```text
第 3/4 步是在造出可以被 preserve loss 约束的对象；
第 5 步 preserve loss 是真正使用这个对象进行训练约束。
```

完整链路是：

```text
x / x+
  ↓
AR backbone 计算 motion likelihood
  ↓
第 3 步：计算每个 token 对动作生成的贡献 alpha
  ↓
第 4 步：用 alpha 加权得到 u(x), u(x+)
  ↓
第 5 步：preserve loss 约束 u(x) 和 u(x+) 接近
```

也就是说：

```text
第 3 步：找哪些词重要。
第 4 步：把重要词汇总成动作语义表示。
第 5 步：要求 preserve pair 的动作语义表示一致。
```

举例：

```text
x:
A person walks forward and raises both arms.

x+:
The person lifts up both arms while walking ahead.
```

第 3 步分别计算两句话中哪些 token 影响动作生成。可能得到：

```text
x  中重要的词：walks, forward, raises, arms
x+ 中重要的词：lifts, arms, walking, ahead
```

第 4 步把这些重要 token 加权汇总成：

```text
u(x)
u(x+)
```

第 5 步 preserve loss 则规定：

```text
因为 x+ 是 x 的同义/句式改写，动作语义没有变，
所以 u(x) 和 u(x+) 应该接近。
```

因此 preserve loss 不是凭空来的。它依赖第 3/4 步得到的 motion-conditioned representation。

如果没有第 3/4 步，preserve loss 只能对齐普通文本 embedding：

```text
CLIP(x) ≈ CLIP(x+)
```

这只是普通语义对齐，不一定和动作生成有关。

COMPASS 对齐的是：

```text
motion-conditioned u(x)
```

也就是：

```text
模型生成动作时真正依赖的文本语义。
```

所以更直白地说：

```text
preserve loss 是第 4 步 u(x) 的使用方式。
```

第 4 步得到 `u(x)`，第 5 步规定：

```text
如果 x+ 是 x 的同义/句式改写，
那么 u(x) 和 u(x+) 要接近。
```

这就是 preserve branch 的核心。

它和 AR backbone 的关系是间接的：

```text
AR backbone
  -> s_AR
  -> attribution alpha
  -> u(x)
  -> preserve loss
```

如果这条梯度链路不断，preserve loss 就能反向约束 AR backbone；如果链路断了，它就主要只训练 projection/predictor head。

一句话总结：

```text
preserve loss 用来训练第 4 步得到的动作语义表示，
让同义改写前后的 u(x) 保持一致；
它通过这个表示间接约束 AR backbone 的语义稳定性。
```

## Alter Branch: 约束语义变化要被感知

Alter branch 处理 `x` 和 `x-`。

它的目标不是让二者相同，而是：

- 被编辑的 slot，attribution 应该发生变化。
- 未编辑的 slot，attribution 应该保持稳定。

### Edited-slot difference loss

```text
L_diff = - sum_i m_i |alpha_i - alpha_i^-| / (sum_i m_i + eps)
```

因为前面有负号，优化时会鼓励 edited slot 上的 attribution 差异变大。

直观理解：

```text
left 改成 right 后，left/right 对应位置的 attribution 应该变化。
```

### Locality loss

```text
L_loc = sum_i (1 - m_i) |alpha_i - alpha_i^-| / (sum_i (1 - m_i) + eps)
```

它约束未编辑位置的 attribution 不要变化太大。

直观理解：

```text
只改了 left/right，walks slowly 这类没改的部分不应该一起乱变。
```

Alter branch 总目标是：

```text
L_alter = lambda_diff * L_diff + lambda_loc * L_loc
```

因此 alter pair 更准确地说是防止模型或 attribution 对语义变化不敏感。

如果把“全 0 坍缩”理解成 attribution 对任何编辑都没反应，那么 alter branch 确实是在防这种退化；但如果指 representation 全部映射成同一个向量，则主要靠 preserve branch 的 `L_var/L_cov` 防止。

## Teacher Consistency

图中的 frozen teacher 是原始或初始化时的 T2M-GPT Transformer。

COMPASS fine-tuning 时，如果只优化新约束，可能损害原始生成能力。因此加入 teacher consistency：

```text
L_teach
```

常见形式是 KL 或 logits L1，让 student 的 motion token 分布不要偏离 frozen teacher 太多。

直观理解：

```text
在增强语义鲁棒性的同时，尽量保留原本会生成动作的能力。
```

## Overall Objective

图中的总目标是：

```text
L = L_gen
  + lambda_teach * L_teach
  + 1[r = preserve] * L_preserve
  + 1[r = alter]    * L_alter
```

其中：

- `L_gen` 是原始 T2M-GPT 的 motion token cross-entropy。
- `L_teach` 是 teacher consistency。
- 当前样本是 preserve pair 时，加入 `L_preserve`。
- 当前样本是 alter pair 时，加入 `L_alter`。

## 训练过程如何理解

训练不是新加一个完全独立的模型阶段，而是：

```text
沿用已有 VQ-VAE 和 T2M-GPT Transformer
-> 在 Transformer 训练/fine-tuning 时加入 COMPASS 数据和 loss
```

更具体地说：

```text
原始 T2M-GPT 训练:
  caption + motion_tokens
  -> L_gen

COMPASS 训练:
  caption_orig + caption_pair + motion_tokens + relation + edit_mask
  -> L_gen
  -> L_teach
  -> preserve 样本加 L_preserve
  -> alter 样本加 L_alter
```

所以它不是“多训练一个楼层/模块后再接上去”，而是在同一个训练循环里多了 attribution 计算和额外约束。

## `u(x)` 如何约束 AR Backbone

一个容易混淆的问题是：

```text
u(x) 并没有作为输入喂给 AR backbone，那它怎么约束 backbone？
```

结论是：

```text
u(x) 不是 AR backbone 的输入；
u(x) 是从 AR backbone 的生成行为中“读出来”的辅助表示；
preserve/alter loss 理论上通过反向传播约束产生这个行为的网络。
```

### `u(x)` 依赖 AR backbone

`u(x)` 的计算链路是：

```text
text token features h_i
        |
AR backbone 计算 p_theta(y_t | y_<t, x)
        |
s_AR(x, y) = 1/K * sum_t log p_theta(y_t | y_<t, x)
        |
a_i = | d s_AR / d h_i * h_i |
        |
alpha_i = softmax(a_i)
        |
u(x) = sum_i alpha_i * phi(h_i)
```

关键点在于：

```text
alpha_i 依赖 s_AR
s_AR 依赖 AR backbone 参数 theta
```

所以虽然 `u(x)` 不作为 backbone 的 forward input，但它是 backbone 当前生成行为的函数：

```text
u(x) = f(theta, x, y)
```

Preserve loss 可以写成：

```text
L_preserve = L_preserve(u(x), u(x+))
           = L_preserve(f(theta, x, y), f(theta, x+, y))
```

因此理论上它最终仍然是 `theta` 的函数。

反向传播时，理想链路是：

```text
d L_preserve / d theta
  = d L_preserve / d u
  * d u / d alpha
  * d alpha / d a
  * d a / d s_AR
  * d s_AR / d theta
```

也就是说：

```text
preserve loss 不需要把 u(x) 输入 AR backbone；
只要 u(x) 由 AR backbone 的输出/梯度计算得到，
loss 就可以通过梯度反过来约束 backbone。
```

这和对 attention map 加辅助 loss 类似：attention map 不一定作为输入返回模型，但对 attention map 的约束可以改变模型参数，从而改变模型未来产生的 attention。

### 为什么这叫 motion-conditioned

如果 `u(x)` 只是普通 CLIP 句向量，那么它主要约束文本 encoder 的语义空间，和 motion generator 的行为关系较弱。

COMPASS 的区别是 `u(x)` 依赖：

```text
s_AR(x, y) = log p_theta(y | x)
```

因此 `u(x)` 表示的是：

```text
当前 AR backbone 为了生成动作 y 时，真正依赖的文本语义。
```

所以对齐 preserve pair 的 `u(x)` 和 `u(x+)`，本质上是在约束：

```text
AR backbone 在生成同一动作时，对原句和改写句依赖相似的动作语义。
```

### Text encoder 冻结时是否还能约束 backbone

理论上可以。

即使 text encoder 冻结，`h_i` 固定，`alpha_i` 仍然依赖 AR backbone：

```text
alpha_i = attribution_i(s_AR)
s_AR = function(theta, h, y)
```

因此只要保留 attribution 的梯度链路，COMPASS loss 仍然可以更新 AR backbone 参数 `theta`。

也就是说：

```text
CLIP/text encoder 冻结，不等于 preserve/alter loss 不能影响 AR backbone。
```

真正关键的是 attribution 计算是否保留高阶梯度。

### 高阶梯度是关键

因为 attribution 本身就是一个梯度：

```text
a_i = d s_AR / d h_i
```

如果还希望 preserve/alter loss 通过 `a_i` 继续回传到 AR backbone 参数 `theta`，就需要对这个梯度再求梯度。

这本质上涉及二阶梯度。因此理想实现需要类似：

```text
create_graph=True
```

如果使用：

```text
create_graph=False
```

则 attribution 会被当成普通数值，后续 loss 对 AR backbone 的梯度链路会被截断或严重削弱。

### 当前代码实现的限制

当前项目中 `compute_grad_input_attribution` 使用的是：

```python
torch.autograd.grad(score, h, retain_graph=False, create_graph=False)
```

这意味着：

```text
attribution alpha 对 AR backbone 参数 theta 的梯度链路大概率断开。
```

因此按当前代码实现，可能出现：

```text
L_gen 和 L_teach 正常更新 AR backbone；
preserve/alter loss 能训练 AttributionPooling / PredictorHead；
但 preserve/alter loss 对 AR backbone 的直接约束可能不完整。
```

所以要区分：

```text
架构思想:
  u(x) 是 AR backbone 行为的函数，COMPASS loss 通过 attribution 链路约束 backbone。

当前实现:
  create_graph=False 可能让这个链路没有完整实现。
```

这也是后续如果认真验证 COMPASS 效果，需要优先检查的点。

## 与本项目数据文件的对应关系

当前项目里常用的 COMPASS pair 文件是：

```text
dataset/HumanML3D/preserving_pairs_filtered.jsonl
```

它不是动作数据，而是文本 pair 标注文件，里面既有 preserve，也有 alter。

动作数据和监督 token 主要来自：

```text
dataset/HumanML3D/train.txt
dataset/HumanML3D/texts/*.txt
dataset/HumanML3D/VQVAE/*.npy
```

其中 `VQVAE/*.npy` 是离散 motion tokens，COMPASS 训练主要用它作为 motion token 监督。

另一个文件：

```text
dataset/HumanML3D/train_alter_v2.json
```

是额外的 alter-only 补充数据，主要包含：

- action
- direction
- bodypart

它可以增强 alter branch 的训练，但不是当前默认主训练直接读取的 pair 文件。要使用它，需要转换/合并到 JSONL pair 文件，或修改数据加载器。

## 当前理解中的关键点

1. COMPASS 的 backbone 基本还是 T2M-GPT。
2. 核心改进是 preserve/alter pair + motion-conditioned attribution + 新 loss。
3. Preserve pair 用于约束语义不变改写的一致性。
4. 防止 representation 坍缩主要靠 `L_var/L_cov`。
5. Alter pair 用于约束模型对被编辑语义敏感，并保持未编辑部分稳定。
6. Teacher consistency 用于保留原始 T2M-GPT 的生成能力。
7. 因此 COMPASS 更像 T2M-GPT 的 attribution-aware semantic fine-tuning，而不是一个全新的 text-to-motion 生成架构。
8. `u(x)` 不是 AR backbone 的输入，而是由 AR backbone 的 motion-conditioned attribution 构造出来的辅助表示。
9. 理论上 preserve/alter loss 通过 `u(x)` 的梯度链路约束 AR backbone；当前代码因 `create_graph=False` 可能没有完整实现这一点。
