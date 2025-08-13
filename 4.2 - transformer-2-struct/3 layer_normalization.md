
## ✅ 层归一化（Layer Normalization）

### 📌 形式：

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

LayerNorm 会对每一个 token 的表示向量进行归一化（沿着特征维度）：

$$
\text{LN}(h) = \frac{h - \mu}{\sigma} \cdot \gamma + \beta
$$

其中 $\mu, \sigma$ 是均值和标准差，$\gamma, \beta$ 是可学习参数。

### ✅ 作用：

| 功能               | 描述                                    |
| ---------------- | ------------------------------------- |
| ⚖️ **稳定训练过程**    | 归一化每个 token 的激活值 → 防止激活过大/过小          |
| 🔄 **加速收敛**      | 减少“内部协变量偏移”（internal covariate shift） |
| 🧠 **防止梯度爆炸/消失** | 与残差配合保证信息和梯度都不会“失控”                   |

---

## 🤔 四、为什么是“残差再 LayerNorm”，不是相反？

Transformer 原始论文中采用的是：

$$
\boxed{
\text{LayerNorm}(x + \text{Sublayer}(x))
}
$$

这种方式称为 **Post-LN**。优点是：

* **逻辑清晰**：先加残差，再归一化所有信息。
* **更稳定**：归一化之后数值不会爆炸。

---

## 为什么transformer是用LayerNorm而不是BatchNorm

主要原因是transformer处理的是变长序列，如果用BatchNorm，在归一化时PAD位置的无效数据和有效数据混在一起计算均值和方差，这些填充位的特征是无意义的，它们会 “污染” 批次的统计量，导致计算出的均值和方差有偏，从而影响模型训练的稳定性和效果。

---

## 🆚 扩展：Pre-LN 结构（后来的改进）

有些后来的 Transformer（如 GPT-2/3/4）使用的是：

$$
\boxed{
x + \text{Sublayer}(\text{LayerNorm}(x))
}
$$

这种称为 **Pre-LN**，好处是：

* 每层前都归一化，前向/反向梯度更稳定；
* 在非常深的网络中更利于收敛。



