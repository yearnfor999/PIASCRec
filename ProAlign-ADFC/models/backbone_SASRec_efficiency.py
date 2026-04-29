import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from SASmodules import SASRec
from models.modules import *

# ==================== [NEW] ProAlign 需要的额外导入 ====================
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ==================== [END NEW] ==========================================

# =============================================================================
# Item_Embedding 类：物品嵌入层
# 这是所有模型共用的嵌入层，支持多种嵌入策略：
#   - ID: 纯 ID 嵌入（随机初始化）
#   - SI: 语义初始化（用 LLM 嵌入初始化 ID 嵌入）
#   - SR: 语义重建（ID 嵌入 + 语言嵌入用于重建损失）
#   - Dual_view: 双视图（LLMESR 用）
#   - AP: 自适应投影（语言嵌入 + 适配器）
#   - WAP: 白化自适应投影（白化后的语言嵌入 + 适配器）
#   - AF: AlphaFuse（零空间融合）
# =============================================================================
class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, **key_words):
        """
        初始化物品嵌入层

        Args:
            emb_pipline: 嵌入策略类型 ("ID"/"SI"/"SR"/"Dual_view"/"AP"/"WAP"/"AF")
            key_words: 包含各种配置参数的字典
        """
        super(Item_Embedding, self).__init__()
        # 读取数据统计信息
        data_statis = pd.read_pickle(
            os.path.join(key_words["language_embs_path"], 'data_statis.df'))  # './data/ASO/data_statis.df'
        self.state_size = data_statis['seq_size'][0]  # 序列长度  10
        self.item_num = data_statis['item_num'][0]  # 物品数量  18357
        # 根据嵌入策略构建嵌入层（修改 self，不返回值）
        self.construct_item_embeddings(emb_pipline, **key_words)
        print("Item_Embedding 类初始化完成")

    def construct_item_embeddings(self, emb_pipline, **key_words):
        """
        根据嵌入策略构建物品嵌入层

        Args:
            emb_pipline: 嵌入策略类型
        """
        # -------------------- ID: 纯 ID 嵌入（SASRec 基线）--------------------
        if emb_pipline == "ID":
            # 随机初始化 ID 嵌入，不使用任何语义信息
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])

        # -------------------- SI: 语义初始化（LLMInit）--------------------
        elif emb_pipline == "SI":  # semantic initialization
            # 用 LLM 语言嵌入初始化 ID 嵌入，之后可微调
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)

        # -------------------- SR: 语义重建（RLMRec）--------------------
        elif emb_pipline == "SR":  # semantic reconstruction
            # ID 嵌入随机初始化，同时加载冻结的语言嵌入用于重建损失
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            # padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            # language_embs = np.vstack([language_embs, padding_emb])
            # 语言嵌入冻结，仅用于计算重建损失
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
            )

        # -------------------- Dual_view: 双视图（LLMESR）--------------------
        elif emb_pipline == "Dual_view":  # Dual view modeling of LLNESR
            # 同时使用 ID 嵌入和语言嵌入，通过交叉注意力融合
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        # -------------------- AP: 自适应投影（MoRec/UniSRec）--------------------
        elif emb_pipline == "AP":  # Adaptive Projection
            # 加载语言嵌入，通过适配器（MLP/MoE）投影到隐藏空间
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        # -------------------- WAP: 白化自适应投影（WhitenRec）--------------------
        elif emb_pipline == "WAP":  # Adaptive Projection for whitened language embeddings
            # 对语言嵌入进行 PCA 白化处理，消除各维度的相关性
            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion(None, **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        # -------------------- AF: AlphaFuse（本文方法）--------------------
        elif emb_pipline == "AF":  # AlphaFuse
            # 核心创新：在语言嵌入的零空间中注入 ID 信息
            # 1. 对语言嵌入进行 SVD 分解，识别零空间（方差小的维度）
            # 2. 语言嵌入投影到主成分空间（冻结）
            # 3. ID 嵌入只学习零空间维度，与语言嵌入相加融合

            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)  # (18357,128)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding 位置用随机向量  (128,)
            cliped_language_embs = np.vstack(
                [cliped_language_embs, padding_emb])  # (18358,128)  np.vstack：在原矩阵“下面”多叠了一行
            # 创建 LLM 侧的nn.Embedding
            #
            # 参数	                    含义
            # from_pretrained	    用预计算的权重初始化
            # freeze=True	        冻结权重，不更新
            # padding_idx=18357	    索引 18357 是 padding 位置

            # 为什么使用 from_pretrained？加载预计算好的 LLM 语言嵌入，而不是随机初始化
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs, dtype=torch.float32),  # (18358,128)
                freeze=True,  # ← 语言嵌入冻结，不参与训练
                padding_idx=self.item_num  # (18357)
            )  # (18358,128,padding_idx=18357)

            # 嵌入表结构：
            #
            # language_embeddings (18358, 128)     ID_embeddings (18358, 64)
            # ┌──────────────────────────┐         ┌─────────────┐
            # │ 物品 0    [128维向量]    │          │ 物品 0 [64] │
            # │ 物品 1    [128维向量]    │          │ 物品 1 [64] │
            # │    ...                   │    +    │    ...      │
            # │ 物品 18356 [128维向量]   │          │ 物品18356   │
            # │ padding   [随机128维]    │          │ padding     │
            # └──────────────────────────┘         └─────────────┘
            #         冻结                           可学习

            # self.nullity：表示 零空间的维度数，64
            #
            # key_words["ID_embs_init_type"]
            # 一个字符串，控制 ID 嵌入用什么方式初始化，比如：
            # "normal"：标准高斯初始化
            # "uniform"：均匀分布
            # "zero"：全 0
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])  # (18358, 64) ID 嵌入只学习零空间维度（nullity 维）
            print("初始化 language_embeddings和ID_embeddings完成！")
            # self.init_ID_embedding(self.nullity, "zeros")

    def load_language_embeddings(self, directory, language_model_type, scale):
        """
        加载预计算的 LLM 语言嵌入

        Args:
            directory: 数据目录路径
            language_model_type: 语言模型类型 ("3small" 或 "3large")
            scale: 缩放因子（放大嵌入值，避免数值过小）

        Returns:
            language_embs: [item_num, language_dim] 的语言嵌入矩阵
        """
        # 从 pickle 文件加载语言嵌入
        language_embs = pd.read_pickle(os.path.join(directory,
                                                    language_model_type + '_emb.pickle'))  # './data/ASO/3large_emb.pickle'   (18357,3072)
        self.item_num = len(language_embs)  # 物品数量   18357
        self.language_dim = len(language_embs[0])  # 语言嵌入维度（3small=1536, 3large=3072）  3072
        # np.stack 将list/pandas.Series 变成一个真正的 np.ndarray，shape = (N, D) 的矩阵
        # language_embs 已经是一个 np.ndarray，形状是 (18357, 3072)，不加 np.stack也可以
        #
        # *scale:对这个矩阵的所有元素乘以scale
        return np.stack(language_embs) * scale  # 堆叠并缩放

    def init_ID_embedding(self, ID_dim, init_type, **key_words):
        """
        初始化 ID 嵌入层

        Args:
            ID_dim: ID 嵌入维度
            init_type: 初始化方式
                - "language_embeddings": 用语言嵌入初始化（可微调）
                - "normal": 标准正态分布初始化
                - "zeros": 零初始化
                - "uniform": 均匀分布初始化
                - "ortho": 正交初始化
                - "xavier": Xavier 初始化
                - "sparse": 稀疏初始化
        """
        if init_type == "language_embeddings":
            # 用 LLM 语言嵌入初始化 ID 嵌入（可微调，freeze=False）
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            if self.language_dim == ID_dim:
                # 语言嵌入维度与 ID 维度相同，直接使用
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,  # 可微调
                    padding_idx=self.item_num
                )
            else:
                # 语言嵌入维度与 ID 维度不同，需要 PCA 降维
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                # language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,  # 可微调
                    padding_idx=self.item_num
                )
        else:
            # 随机初始化 ID 嵌入
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 1,  # +1 是 padding 位置   18358
                embedding_dim=ID_dim,  # 64 维（零空间维度）
                # padding_idx=self.item_num  # ← 建议加上
            )  # (18358,64)
            # 根据 init_type 选择初始化方式
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)  # U(0, 1)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)  # N(0, 1)   用均值 0、标准差 1 的高斯分布随机初始化 ID embedding 的参数
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)  # 全零
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)  # 正交矩阵
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)  # Xavier
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)  # 稀疏矩阵
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

    def semantic_space_decomposion(self, clipped_dim, **key_words):
        """
        语义空间分解（AlphaFuse 的核心算法）

        这是 AlphaFuse 的核心创新：对语言嵌入进行 SVD 分解，
        识别"零空间"（方差小的维度），用于注入 ID 信息。

        算法步骤：
        1. 计算语言嵌入的协方差矩阵
        2. SVD 分解得到特征向量 U 和特征值 S
        3. 根据阈值或维度确定零空间
        4. 可选：白化标准化（使各方向方差为 1）
        5. 投影到主成分空间

        Args:
            clipped_dim: 目标维度（投影后的维度）

        Returns:
            clipped_language_embs: [item_num, clipped_dim] 投影后的语言嵌入
        """
        # 加载语言嵌入
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"],
                                                      key_words["language_embs_scale"])  # (18357,3072)

        # 计算协方差矩阵
        if not key_words["item_frequency_flag"]:  # 不考虑物品频率时，按均匀权重算语言嵌入的均值和协方差（不按物品出现频率加权，也就是默认每个 item 的权重一样）
            # 默认：均匀分布（所有物品权重相同）
            self.language_mean = np.mean(language_embs, axis=0)  # 计算均值 (3072,)
            # language_embs - self.language_mean： 把每个 item 的向量都减去均值 𝜇，做中心化
            cov = np.cov(language_embs - self.language_mean, rowvar=False)  # 计算协方差矩阵 (3072, 3072)
        else:
            # 可选：按物品频率加权
            items_pop = np.load(os.path.join(key_words["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop * items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * items_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(items_freq), rowvar=False)
            # raise NotImplementedError("Custom item distribution is not implemented yet.")

        # SVD分解（Singular Value Decomposition，中文一般叫“奇异值分解”）
        # SVD = 把一个矩阵拆成 “方向 × 拉伸强度 × 方向” 的乘积
        # Cov = U @ diag(S) @ U^T
        # U: 特征向量矩阵（列为主成分方向）
        # S: 特征值（各方向的方差）

        # cov 是 (D, D) 的协方差矩阵，这里 D=3072
        # SVD 结果：
        # U.shape = (D, D)：列向量 u_i 就是主成分方向（方差大的方向 → 语义信息丰富（row space）方差很小的方向 → 几乎没语义（接近零空间））；
        # S.shape = (D,)：奇异值，对应每个语义方向上的方差大小 （越大代表语义越强）
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        # 确定零空间维度（nullity）
        if key_words["null_thres"] is not None:
            # 方式1：根据阈值确定（特征值 < 阈值的维度为零空间）
            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        elif key_words["null_dim"] is not None:
            # 方式2：直接指定零空间维度
            self.nullity = key_words["null_dim"]  # 64
        # print("The Nullity is", self.nullity)
        # self.squared_singular_values = S
        # self.language_bases = U

        # 确定投影维度
        if clipped_dim is None:  # 128
            clipped_dim = self.language_dim
        if key_words["cover"]:  # False
            # cover=True	覆盖	后 64 维完全是 ID 嵌入
            # cover=False	注入	ID 嵌入叠加到弱语义区
            #
            # 关键区别：
            # 覆盖（cover）：后 64 维完全是 ID 嵌入
            # 注入（inject）：后 64 维是 语义 + ID 的混合

            # 原始 LLM 嵌入 (3072 维)
            #          ↓
            #       SVD 分解
            #          ↓
            # ┌────────────────────────────────────────────────┐
            # │  特征值大 ──────────────────→ 特征值小           │
            # │  ↓                               ↓             │
            # │  主语义空间                      零空间          │
            # │  (semantic space)            (null space)      │
            # │  保留语言模型的                可以注入 ID        │
            # │  核心语义信息                  协同信息           │
            # └────────────────────────────────────────────────┘
            # 3072 维降维到128 维
            # ├── 前 64 维：强语义（保留）
            # └── 后 64 维：弱语义/零空间（被 ID 嵌入覆盖或注入）
            clipped_dim = clipped_dim - self.nullity

        # 构造投影矩阵
        #
        # U.shape = (D, D)
        # U 的每一列 U[:, i] 就是第 i 个主成分方向（语义方向），按 S 从大到小排好
        #
        # U[...,:clipped_dim] 是什么意思？
        # ... 在 NumPy 里就是“保持前面的维度都不动”的意思
        # 对二维矩阵来说： U[..., :clipped_dim]  ==  U[:, :clipped_dim]
        # 也就是：
        # 行维度：: → 取所有行（保持 3072 行）
        # 列维度：:clipped_dim → 取前 clipped_dim 列
        # 也就是把前 clipped_dim 个主成分方向拿出来，堆成一个矩阵
        Projection_matrix = U[..., :clipped_dim]  # 取前 clipped_dim 个主成分  (3072,128)

        # 可选：白化标准化（使各方向方差为 1）
        if key_words['standardization']:
            # 1. 白化系数计算
            # 符号	            含义	                        形状
            # S	                SVD 特征值（各方向的方差）	    (3072,)
            # 1/S	            方差的倒数	                (3072,)
            # np.sqrt(1/S)	    标准差的倒数	                (3072,)
            # [:clipped_dim]	只取前 128 维	            (128,)
            #
            # 作用：消除各方向的方差差异，使白化后每个方向方差 = 1
            Diagnals = np.sqrt(1 / S)[:clipped_dim]  # 1/sqrt(特征值)  (128,)
            # 2. 构造白化投影矩阵
            # 原始 Projection_matrix	        np.diag(Diagnals)	        结果
            # U[:, :128]	                对角矩阵	                白化投影矩阵
            # (3072, 128)	                (128, 128)	            (3072, 128)
            #
            # 数学含义：
            # 原始投影：X @ U → 投影到主成分空间（各方向方差不同）
            # 白化投影：X @ U @ diag(1/√S) → 投影 + 缩放（各方向方差=1）
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))  # V_{\lambda} -> V_1   (3072,128)

        # 3. 最终投影
        # 步骤	                操作	                形状变化
        # 1	                  中心化（减均值）	    (18357, 3072)
        # 2	                  投影（降维 + 白化）	    (18357, 3072) @ (3072, 128) = (18357, 128)
        #
        # 图示
        #
        # 原始 LLM 嵌入                              白化后的嵌入
        # (18357, 3072)                             (18357, 128)
        #
        # ┌─────────────────┐                      ┌─────────┐
        # │                 │                      │         │
        # │  每个物品       │   减均值               │  每个   │
        # │  3072 维向量    │  ──────→  投影矩阵     │  物品   │
        # │                 │          (3072,128)  │  128维  │
        # │  18357 个物品   │                       │         │
        # └─────────────────┘                      └─────────┘
        #
        # 投影后特性：
        #  各维度独立（协方差矩阵是对角阵）
        #  各维度方差 = 1（白化）
        #  前面的维度 = 强语义
        #  后面的维度 = 弱语义（零空间）
        #
        # 白化的作用
        #   不白化	                    白化后
        # 前几维方差很大（主成分）	    所有维度方差 = 1
        # 后几维方差很小（零空间）	    所有维度方差 = 1
        # ID 嵌入难以与语义嵌入匹配	ID 嵌入更容易融合
        # 本质：白化让各维度"平等"，ID 信息注入时不会被强语义维度压制
        clipped_language_embs = (language_embs - self.language_mean).dot(
            Projection_matrix)  # (18357,128)  投影：(X - mean) @ Projection_matrix
        return clipped_language_embs


# =============================================================================
# SASRec_backbone：SASRec 骨干网络基类
# 这是所有序列推荐模型的基类，实现了：
#   - Transformer 编码器结构
#   - 三种损失函数：CE、BCE、InfoNCE
#   - 预测接口
# 子类只需实现 embed_ID() 和 return_item_emb() 方法
# =============================================================================
class SASRec_backbone(nn.Module):
    def __init__(self, device, **key_words):
        """
        初始化 SASRec 骨干网络

        架构：
        Input -> Embedding + Position -> Dropout -> LayerNorm ->
        MultiHeadAttention -> FeedForward -> LayerNorm -> Output

        Args:
            device: 计算设备 cuda
            key_words: 配置参数字典
        """
        super(SASRec_backbone, self).__init__()

        # 读取数据统计信息
        data_statis = pd.read_pickle(
            os.path.join(key_words["language_embs_path"], 'data_statis.df'))  # './data/ASO/data_statis.df'   './data/Beauty/data_statis.df'
        self.seq_len = data_statis['seq_size'][0]  # 序列长度   10   50
        self.item_num = data_statis['item_num'][0]  # 物品数量（padding_idx = item_num）18357   12101
        # self.item_embeddings = Item_Embedding("ID", **key_words)
        # self.item_num = item_num
        # self.seq_len = seq_len

        # 基本配置
        self.dropout = key_words["dropout_rate"]  # Dropout 概率  0.1
        self.device = device  # 计算设备
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失
        self.bce_loss = nn.BCEWithLogitsLoss()  # 二元交叉熵损失

        # self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]  # 隐藏层维度    128

        # 位置嵌入：学习序列中每个位置的表示
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,  # 序列长度           10           50
            embedding_dim=self.hidden_dim  # 与隐藏层维度相同   128          128
        )  # (10,128)   (50,128)

        # Transformer 组件
        #
        # 与标准 Transformer 对比
        # 标准 Transformer（2 层 LN，Post-LN）
        #
        # x → Attention → + → LN ─┐
        #                    ↑    │
        #                    └────┘ 残差
        #
        #   → FFN → + → LN ─┐
        #              ↑    │
        #              └────┘ 残差
        #
        # AlphaFuse（3 层 LN，Pre-LN + 输出 LN）
        # x → LN → Attention → (+) → LN → FFN → (+) → mask → LN → 输出
        #     ↑                       ↑                       ↑
        #    ln_1                    ln_2                    ln_3
        #
        # AlphaFuse 多了一个 ln_3 在最后输出时使用
        self.emb_dropout = nn.Dropout(self.dropout)  # 嵌入层 Dropout      0.1
        self.ln_1 = nn.LayerNorm(self.hidden_dim)  # 注意力前的 LayerNorm
        self.ln_2 = nn.LayerNorm(self.hidden_dim)  # FFN 前的 LayerNorm
        self.ln_3 = nn.LayerNorm(self.hidden_dim)  # 输出前的 LayerNorm
        # 多头自注意力层（带因果掩码，防止看到未来信息）
        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        # 前馈网络层
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        # self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()

    def embed_ID(self, x):
        """
        获取物品 ID 嵌入（抽象方法，子类必须实现）

        Args:
            x: [B, S] 或 [B] 物品 ID 序列

        Returns:
            embeddings: [B, S, D] 或 [B, D] 物品嵌入
        """
        # return self.item_embeddings.ID_embeddings(x)
        pass

    def return_item_emb(self, ):
        """
        返回全量物品嵌入矩阵（抽象方法，子类必须实现）

        Returns:
            item_embs: [item_num+1, D] 所有物品的嵌入（包含 padding）
        """
        # return self.item_embeddings.ID_embeddings.weight
        pass

    # 调用 class SASRec_backbone 中的 forward()
    # 训练：
    # train_loader → calculate_infonce_loss() → forward()
    #
    # 推理/评估：
    # val_loader → evaluate() → model.predict() → forward()
    def forward(self, sequences):
        """
        前向传播：序列编码

        流程：
        1. 物品嵌入 + 位置嵌入
        2. Dropout
        3. Padding 掩码
        4. LayerNorm -> MultiHeadAttention（带因果掩码）
        5. LayerNorm -> FeedForward
        6. 取最后一个时间步的输出作为用户表示

        Args:
            sequences: [B, S] 输入序列（物品 ID）

        Returns:
            logits: [B, D] 用户表示（最后一个时间步的隐状态）
        """
        # 物品嵌入  注意：这边的后64维是融合后的
        inputs_emb = self.embed_ID(sequences)  # sequences：(256,10) ——> inputs_emb：(256,10,128)
        # 位置嵌入
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))  # (256,10) ——>(256,10,128)
        # BSARec 和 AlphaFuse 两者的区别是：
        #
        # BSARec：先 LayerNorm，再 Dropout
        # AlphaFuse：直接 Dropout（没有 LayerNorm）
        seq = self.emb_dropout(inputs_emb)  # 0.1   (256,10,128)

        # Padding 掩码：根据 padding ID（self.item_num）为每个序列位置生成一个 0/1 掩码

        # 举个小例子（用小 batch 更好理解）：
        # 1.sequences =
        # tensor([[1, 2, 3, 5],
        #         [4, 6, 7, 5]])
        #
        # self.item_num = 5  # 约定 5 是 padding ID
        #
        # 则：
        #
        # torch.ne(sequences, 5)
        # = tensor([[ True,  True,  True, False],
        #           [ True,  True,  True, False]])
        #
        #
        # True 代表“真实 item（非 padding）”，False 代表“padding 位置”
        # 此时 mask 布尔张量的形状：(256, 10)，dtype 是 torch.bool
        #
        # 2.float()
        # ... .float()
        # 把布尔值转成浮点数：
        # True → 1.0
        # False → 0.0
        #
        # 3.unsqueeze(-1)
        # ... .unsqueeze(-1)
        #
        # 在最后一维加一个维度，相当于：
        # 原来：(batch_size, seq_len) → (batch_size, seq_len, 1)
        # 即：(256, 10) → (256, 10, 1)
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)  # (256,10,1)
        # 非 padding 位置：embedding * 1 = 保留
        # padding 位置：embedding * 0 = 置零
        #
        #
        # 为什么注意力掩码不够？
        # 你可能会问：注意力层里已经有 Key Masking 了，为什么还要在这里置零？
        #
        # 原因 1：LayerNorm 的影响
        # seq_normalized = self.ln_1(seq)  # LayerNorm 会用到所有位置！
        # LayerNorm 计算均值和方差时会包含 padding 位置的值：
        #
        # μ = mean(seq)  # 如果 padding 非零，会污染均值
        # σ = std(seq)   # 同样会受影响
        # 如果 padding 不置零，LayerNorm 的结果会受到影响！
        #
        # 原因 2：前馈网络的影响
        # ff_out = self.feed_forward(...)  # FFN 也会处理所有位置
        # FFN 是逐位置操作，如果 padding 位置有非零值，会产生非零输出。
        #
        # 原因 3：残差连接
        # output = output + seq  # 残差连接会把 padding 的值带到输出
        # BSARec 也有类似操作吗？
        # 没有显式的 seq *= mask，但 BSARec 通过其他方式处理：
        #
        # BSARec 在 nn.Embedding 中设置了 padding_idx=0
        # self.item_embeddings = nn.Embedding(..., padding_idx=0)
        # 这保证 id=0 的嵌入向量全是 0 且不更新
        # 但 AlphaFuse 的 padding_idx = item_num（最后一个位置），所以需要显式置零
        seq *= mask  # (256,10,128)

        # Transformer 编码
        #
        # ========== Pre-LN 结构 ==========
        seq_normalized = self.ln_1(seq)  # (256,10,128)                                         1. 先做 LayerNorm
        mh_attn_out = self.mh_attn(seq_normalized, seq)  # 多头自注意力（带因果掩码） (256,10,128)  2. 再做 Attention
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))  # 前馈网络    (256,10,128)           3. FFN 也是 Pre-LN
        # 为什么 FFN 后要再次 ff_out *= mask？
        # 因为经过多头注意力和 FFN 后，padding 位置可能再次变成非零！
        #
        # 对比 BSARec
        # BSARec 没有显式的 ff_out *= mask，但它也能正常工作，原因：
        #
        # padding_idx=0：嵌入本身就是 0 向量
        # 不同的残差结构：BSARec 的 Transformer 实现方式略有不同
        # 掩码设计：BSARec 的加性掩码在 softmax 前已经处理好了
        #
        #
        # 因为BSARec padding_idx=0：嵌入本身就是 0 向量，AlphaFuse 是Item_num 吗
        # 是的，你理解正确！
        #
        # 对比
        # 项目	        padding_idx	        padding         值	                    特点
        # BSARec	    0	                序列中用         0 填充	            最前面的 ID 是 padding
        # AlphaFuse	    item_num	        序列中用        item_num 填充	        最后面的 ID 是 padding
        # AlphaFuse 的代码证据
        # backbone_SASRec.py
        # class Item_Embedding(nn.Module):
        #     def __init__(self, ...):
        #         data_statis = pd.read_pickle(...)
        #         self.item_num = data_statis['item_num'][0]  # 例如 18357
        #
        #     # AF 模式
        #     self.language_embeddings = nn.Embedding.from_pretrained(
        #         ...,
        #         freeze=True,
        #         padding_idx=self.item_num  # ← padding_idx = 18357
        #     )
        #
        #     self.ID_embeddings = nn.Embedding(
        #         num_embeddings=self.item_num + 1,  # 18358 个嵌入
        #         embedding_dim=...,
        #         # 注意：这里没有设置 padding_idx！
        #     )
        # AlphaFuse 的 ID 范围：
        #
        # 物品 ID: 0, 1, 2, ..., 18356  (共 18357 个物品)
        # padding: 18357                (最后一个位置)
        # 这导致了什么问题？
        # BSARec（padding_idx=0）
        # self.item_embeddings = nn.Embedding(..., padding_idx=0)
        # # ID=0 的嵌入向量：
        # # - 初始化为 0
        # # - 训练时梯度不更新
        # # - 始终保持为 0
        #  天然就是 0 向量，不需要额外处理
        #
        # AlphaFuse（padding_idx=item_num）
        # self.language_embeddings: padding_idx=item_num  # ← 有设置
        # self.ID_embeddings: 没有设置 padding_idx！      # ← 问题在这里！
        #  ID_embeddings 的 padding 位置可能不是 0
        #
        # 而且位置编码也会加上去：
        #
        # python
        # inputs_emb = self.embed_ID(sequences)               # padding 可能非零
        # inputs_emb += self.positional_embeddings(...)       # 加上位置编码
        # # 现在 padding 位置肯定非零了！
        # 所以必须显式 seq *= mask 置零！
        #
        # 总结
        # 问题	                BSARec	                AlphaFuse
        # padding 位置	        ID=0（最前面）	        ID=item_num（最后面）
        # 嵌入是否为 0	         是（padding_idx=0）	 不一定（ID_embeddings 没设 padding_idx）
        # 需要手动置零吗？	     不需要	             需要 seq *= mask
        # 这就是 AlphaFuse 代码更复杂的原因之一！
        ff_out *= mask  # 再次应用掩码  (256,10,128)
        ff_out = self.ln_3(ff_out)  # (256,10,128)

        # 取最后一个时间步作为用户表示
        logits = ff_out[:, -1].squeeze()  # [B, D]  (256,128)
        return logits

    def predict(self, sequences):
        """
        预测：计算用户对所有物品的得分

        Args:
            sequences: [B, S] 输入序列

        Returns:
            scores: [B, item_num] 用户对每个物品的预测得分
        """
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)  # [B, D] 用户表示           (256,128)  调用 Class SASRec_backbone 的forward()
        item_embs = self.return_item_emb()  # 调用 class AlphaFuse(SASRec_backbone) 的 def return_item_emb(self,)    [item_num+1, D] 物品嵌入  (18358,128)
        # 为什么去掉 padding
        #
        # 一句话总结：因为 LLM 侧的 padding 嵌入是随机初始化的非零向量，可能被 Top-K 选中
        #
        # 详细原因
        # 1. LLM 侧的 padding 嵌入不是零向量
        # 代码：backbone_SASRec.py
        # padding_emb = np.random.rand(128)  # ← 随机初始化！不是 0！
        # cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
        #
        # 2. 预测时会计算 padding 的得分
        # scores = torch.matmul(user_state, item_embs.transpose(0, 1))
        # user_state · padding_emb ≠ 0  ← 可能得分较高！
        #
        # 3. Top-K 可能选中 padding
        # _, topK = scores.topk(100, largest=True)
        # topK 可能包含 18357（padding）
        #
        # 4. 导致评估指标错误
        # 预测：[物品5, padding, 物品3, ...]  ← 错误的推荐！
        #
        #
        # 与 BSARec 对比
        # 项目	        padding 嵌入	            需要手动去掉吗？
        # BSARec	    0 向量	                 不需要（得分=0，不会被选中）
        # AlphaFuse	    随机非零向量	             必须去掉
        scores = torch.matmul(state_hidden, item_embs[:-1].transpose(0, 1))  # [B, item_num]（去掉 padding） (256,18357)
        return scores

    def calculate_ce_loss(self, sequences, target):
        """
        计算 Cross-Entropy 损失（全物品 softmax）

        Args:
            sequences: [B, S] 输入序列
            target: [B] 目标物品 ID

        Returns:
            loss: 标量损失值
        """
        seq_output = self.forward(sequences)  # [B, D]
        item_embs = self.return_item_emb()  # [item_num+1, D]
        # item_embs = self.item_emb.return_embs()
        logits = torch.matmul(seq_output, item_embs[:-1].transpose(0, 1))  # [B, item_num]
        loss = self.ce_loss(logits, target)
        return loss

    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        """
        计算 Binary Cross-Entropy 损失（负采样二分类）

        思路：
        - 正样本：目标物品，标签为 1
        - 负样本：随机采样 neg_ratio 个物品，标签为 0
        - 二分类：sigmoid(user · item) → 0/1

        Args:
            sequences: [B, S] 输入序列
            target: [B] 正样本物品 ID
            neg_ratio: 负采样数量

        Returns:
            loss: 标量损失值
        """
        # ==================== 负采样 ====================
        # 随机采样负样本，确保不与正样本重复
        # sequences_set = set(sequences.view(-1).tolist())
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        # expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        # mask_target = neg_samples == expanded_target
        # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
        # mask = mask_target | mask_sequences
        mask = neg_samples == expanded_target
        # 重采样与正样本重复的负样本
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
            # mask_target = neg_samples == expanded_target
            # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            # mask = mask_target | mask_sequences
        target_neg = neg_samples.to(target.device)

        # ==================== 计算得分 ====================
        # pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)  # [B, D] 正样本嵌入
        neg_embs = self.embed_ID(target_neg)  # [B, neg_ratio, D] 负样本嵌入

        log_feats = self.forward(sequences)  # [B, D] 用户表示

        # 点积得分
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [B]
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)  # [B, neg_ratio]

        # ==================== BCE 损失 ====================
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape,
                                                                                               device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)  # 正样本损失
        loss += self.bce_loss(neg_logits, neg_labels)  # 负样本损失

        return loss

    # def calculate_infonce_loss  最后是每一个样本，对应一个正样本，64个负样本
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        """
        计算 InfoNCE 损失（对比学习损失）

        这是推荐系统中最常用的损失函数，核心思想：
        - 正样本对 (user, positive_item) 应该相似
        - 负样本对 (user, negative_items) 应该不相似
        - 使用 softmax 归一化，正样本概率应该最大

        公式：
        L = -log( exp(sim(u, i+)/τ) / Σ exp(sim(u, i)/τ) )

        其中 τ 是温度参数，控制分布的锐度

        Args:
            sequences: [B, S] 输入序列
            target: [B] 正样本物品 ID
            neg_ratio: 负采样数量
            temperature: 温度参数（通常 0.07）

        Returns:
            loss: 标量损失值
        """
        # ==================== 第一部分：负采样 ====================
        # 负采样的目的：从全体物品中随机选择一些物品作为"负样本"
        # 负样本应该是用户没有交互过的物品，用于对比学习

        # sequences_set = set(sequences.view(-1).tolist())  # （已注释）将序列中的物品 ID 转为集合，用于去重

        batch_size = target.shape[0]  # 获取 batch 大小，例如 256

        # 从 [0, item_num) 范围内随机采样 neg_ratio 个负样本
        # 形状：[B, neg_ratio]，例如 [256, 64]，每个样本有 64 个负样本
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))  # (256, 64)

        # 将正样本 target 扩展为 [B, neg_ratio]，用于后续与负样本比较
        # target: [256,] → [256, 1] → [256, 64]（每行都是相同的正样本 ID）
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()  # (256, 64)

        # 将序列扩展为 [B, S, neg_ratio]，用于检查负样本是否在用户历史中（已注释掉）
        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()

        # mask_target = neg_samples == expanded_target           # （已注释）检查负样本是否与正样本相同
        # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)  # （已注释）检查负样本是否在序列中
        # mask = mask_target | mask_sequences                    # （已注释）合并两个掩码

        # 生成掩码：标记哪些负样本与正样本相同（需要重新采样）
        # mask[i, j] = True 表示第 i 个样本的第 j 个负样本与正样本相同
        #
        # neg_samples 形状是 [B, neg_ratio]，expanded_target 形状也是 [B, neg_ratio]
        # 做比较 == 之后，mask 也是 [B, neg_ratio] 的 布尔张量：
        #
        # mask[b, j] == True 表示：
        # 第 b 个样本的第 j 个负样本 刚好等于 正样本 ID → 这其实是“假负样本”，需要重采样
        #
        # mask[b, j] == False 表示：
        # 这个负样本和正样本不一样，是一个“合格的负样本”
        mask = neg_samples == expanded_target  # [256, 64] 布尔张量

        # 重采样与正样本重复的负样本（循环直到没有重复）
        #
        # 对一个布尔张量调用 .any() 时，会在所有维度上做逻辑或：
        #
        # 也就是：
        # 只要 mask 里 有任意一个位置是 True，mask.any() 就会返回 True；
        # 如果 mask 里 全部都是 False，mask.any() 才会返回 False。
        #
        # 所以：
        # mask.any() == True → 说明当前还有至少一个负样本和正样本重复；
        # mask.any() == False → 所有负样本都已经和正样本不重复了
        while mask.any():  # 只要还有重复的负样本
            # 重新采样 [B, neg_ratio] 个新的随机物品
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            # 只替换重复的位置：mask=True 的位置用 new_samples，否则保留 neg_samples
            neg_samples = torch.where(mask, new_samples, neg_samples)
            # 重新检查是否还有重复
            mask = neg_samples == expanded_target
            # mask_target = neg_samples == expanded_target
            # mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            # mask = mask_target | mask_sequences

        # 将负样本移动到与 target 相同的设备（GPU/CPU）
        target_neg = neg_samples.to(
            target.device)  # 负样本 [256, 64]，batch 里有 256 个样本（256 个用户 / 序列），给每个样本采了 64 个 负样本 item 的 ID

        # ==================== 第二部分：获取嵌入向量 ====================

        # pos_embs = self.item_embeddings(target)  # （已注释）直接从嵌入表查找，不经过模型处理

        # 获取正样本的嵌入向量
        # embed_ID 会根据模型类型（如 AlphaFuse）进行相应的嵌入融合
        pos_embs = self.embed_ID(target)  # [B, D] = [256, 128] 正样本嵌入

        # 获取负样本的嵌入向量
        # target_neg: [B, neg_ratio] → neg_embs: [B, neg_ratio, D]
        neg_embs = self.embed_ID(target_neg)  # [B, neg_ratio, D] = [256, 64, 128] 负样本嵌入

        # 前向传播，获取用户表示（序列编码后的最后一个时间步）
        log_feats = self.forward(sequences)  # [B, D] = [256, 128] 用户表示   调用 Class SASRec_backbone 的forward()

        # ==================== 第三部分：L2 归一化（计算余弦相似度）====================
        # 归一化的目的：将向量投影到单位超球面上，使点积等价于余弦相似度
        # 余弦相似度范围 [-1, 1]，不受向量模长影响

        # 对用户表示进行 L2 归一化：||log_feats|| = 1
        log_feats = F.normalize(log_feats, p=2, dim=-1)  # [B, D]  (256,128)

        # 对正样本嵌入进行 L2 归一化：||pos_embs|| = 1
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)  # [B, D]  (256,128)

        # 对负样本嵌入进行 L2 归一化：||neg_embs|| = 1
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)  # [B, neg_ratio, D]  (256,64,128)

        # （已注释）手动实现 L2 归一化的等价写法
        # normed_log_feats = log_feats / torch.sqrt(1e-8 + log_feats.square().sum(-1, keepdim=True))
        # normed_pos_embs = pos_embs / torch.sqrt(1e-8 + pos_embs.square().sum(-1, keepdim=True))
        # normed_neg_embs = neg_embs / torch.sqrt(1e-8 + neg_embs.square().sum(-1, keepdim=True))

        # ==================== 第四部分：计算相似度得分 ====================

        # 计算正样本得分：用户表示 · 正样本嵌入（逐元素乘积后求和 = 点积）
        # log_feats: [B, D], pos_embs: [B, D]
        # 逐元素乘积: [B, D]，然后沿 dim=-1 求和得到 [B]，keepdim=True 保持形状为 [B, 1]
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)  # [B, 1] (256,1) 正样本得分

        # 计算负样本得分：用户表示 · 每个负样本嵌入
        # neg_embs: [B, neg_ratio, D]
        # log_feats.unsqueeze(-1): [B, D, 1]
        # bmm (batch matrix multiply): [B, neg_ratio, D] @ [B, D, 1] = [B, neg_ratio, 1]
        # squeeze(-1): [B, neg_ratio]
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)  # [B, neg_ratio] (256,64) 负样本得分

        # 拼接正样本和负样本得分：正样本在第 0 位
        # pos_logits: [B, 1], neg_logits: [B, neg_ratio]
        # logits: [B, 1 + neg_ratio] = [256, 65]
        logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, 1+neg_ratio]  (256,65)

        # 温度缩放：logits / τ
        # 温度越小，softmax 分布越尖锐（更自信），温度越大，分布越平滑
        # 常用值：τ = 0.07（CLIP）、τ = 0.1（对比学习）
        logits /= temperature  # 温度缩放  (256,65)

        # ==================== 第五部分：交叉熵损失 ====================
        # InfoNCE 本质上是一个 (1 + neg_ratio) 分类问题
        # 正样本在第 0 位，所以标签全为 0

        # 创建标签：全为 0，表示正确答案是第 0 个位置（正样本）
        #
        # 为什么全是 0？
        # 前面构造 logits 是这样的：
        # pos_logits = ...                     # [B, 1]   正样本得分
        # neg_logits = ...                     # [B, neg_ratio] 负样本得分
        # logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [B, 1 + neg_ratio]
        #
        # 对于第 b 个样本，logits[b] 是一个长度为 1 + neg_ratio 的向量：
        # logits[b, 0]：正样本的 logit
        # logits[b, 1:]：neg_ratio 个负样本的 logit
        #
        # F.cross_entropy(logits, labels) 的语义是：
        # 假设 logits 形状是 [B, C]（C 个类别）
        # labels 形状是 [B]，里面存的是 每个样本的“正确类别索引”
        # 比如 labels[b] = 3，意思是：
        # 第 b 个样本的“正确类别”是第 3 类（索引 3）
        #
        # 在你的设计里，我们约定：
        # 每个样本的第 0 维（logits[..., 0]）是正样本，其余都是负样本
        #
        # 所以标签自然就是：
        # labels = [0, 0, 0, 0, ..., 0]  # 长度 B，每个位置都是 0
        #
        # 含义就是：
        # 对于 batch 里的每一个样本，
        # 正确类的 index = 0（也就是正样本所在的位置
        labels = torch.zeros(batch_size, dtype=torch.long,
                             device=logits.device)  # [B] = [256]  (256,)   一个 batch 中，每个样本的正样本在这一行 logits 里的位置（索引）

        # 计算交叉熵损失
        # F.cross_entropy = softmax + negative log likelihood
        # L = -log( exp(logits[0]) / Σ exp(logits[i]) )
        #   = -logits[0] + log(Σ exp(logits[i]))
        loss = F.cross_entropy(logits, labels)

        return loss  # 返回标量损失值


# =============================================================================
# SASRec：纯 ID 嵌入的序列推荐模型（基线）
# 这是最基本的 SASRec 实现，只使用随机初始化的 ID 嵌入
# =============================================================================
class SASRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用纯 ID 嵌入策略
        self.item_embeddings = Item_Embedding("ID", **key_words)

    def embed_ID(self, x):
        """获取物品 ID 嵌入"""
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        return self.item_embeddings.ID_embeddings.weight

    # =============================================================================


# MoRec：语言嵌入 + MLP 适配器
# 直接使用 LLM 语言嵌入，通过 MLP 投影到隐藏空间
# =============================================================================
class MoRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用自适应投影策略（只有语言嵌入）
        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        # MLP 适配器：语言维度 -> 隐藏维度
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()  # GELU 激活函数
        )

    def embed_ID(self, x):
        """获取物品嵌入：语言嵌入 -> MLP 适配器"""
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)


# =============================================================================
# WhitenRec：白化语言嵌入 + MLP 适配器
# 与 MoRec 类似，但先对语言嵌入进行 PCA 白化处理
# 白化消除了各维度之间的相关性
# =============================================================================
class WhitenRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用白化自适应投影策略
        self.item_embeddings = Item_Embedding("WAP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        # MLP 适配器：白化后维度 -> 隐藏维度
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )

    def embed_ID(self, x):
        """获取物品嵌入：白化语言嵌入 -> MLP 适配器"""
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)

    # =============================================================================


# LLMInit：语义初始化
# 用 LLM 语言嵌入初始化 ID 嵌入，然后微调
# 相当于用语义信息进行预训练
# =============================================================================
class LLMInit(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用语义初始化策略（ID 嵌入用语言嵌入初始化，可微调）
        self.item_embeddings = Item_Embedding("SI", **key_words)
        # self.language_dim = self.item_embeddings.language_dim

    def embed_ID(self, x):
        """获取物品 ID 嵌入（已用语言嵌入初始化）"""
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        return self.item_embeddings.ID_embeddings.weight

    # =============================================================================


# =============================================================================
# RLMRec：语义重建
# =============================================================================
# 使用重建损失对齐 ID 嵌入和语言嵌入
#
# 【物品侧对齐】（原有）
#   两种对齐方式：
#   - con (contrastive): 语言嵌入 -> ID 嵌入（对比式）
#   - gen (generative): ID 嵌入 -> 语言嵌入（生成式）
#
# 【用户侧对齐】（NEW 2024-12-15）
#   使用 usr_intent_emb.pkl（用户 LLM 语义嵌入，形状 [N_users, 3072]）
#   
#   数据流：
#     usr_intent_emb[user_id]  →  MLP 映射  →  用户序列表示
#          (3072)                              (hidden_dim)
#                                     ↓
#                            InfoNCE / Cosine 对齐损失
#
#   两种对齐模式（通过 --user_align_mode 选择）：
#   - infonce: InfoNCE 对比学习（与原版 RLMRec 一致，推荐）
#   - cosine:  余弦相似度（与 AlphaFuse 物品侧一致）
#
#   与 LLMESR 的 RASD 区别：
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │  RLMRec 用户侧对齐              vs        LLMESR RASD               │
#   ├──────────────────────────────────────────────────────────────────────┤
#   │  usr_intent_emb.pkl                    sim_user_100.pkl             │
#   │  (用户 LLM 语义嵌入)                    (相似用户列表)                │
#   │  直接语义对齐                           相似用户蒸馏                  │
#   └──────────────────────────────────────────────────────────────────────┘
#
#   命令行参数：
#     --use_user_llm:     是否启用用户侧 LLM 信息 (默认 False)
#     --alpha_user:       用户侧对齐损失权重 (默认 1.0)
#     --user_align_mode:  对齐模式 infonce/cosine (默认 infonce)
#     --user_align_temp:  InfoNCE 温度参数 (默认 1.0)
#
#   损失函数：
#     L_total = L_main + β * L_item + α_user * β * L_user
#     其中：
#       - L_main: 主损失（InfoNCE）
#       - L_item: 物品侧重建损失（原有）
#       - L_user: 用户侧重建损失（新增）
#
# =============================================================================
# ==================== [NEW 2024-12-15] 添加用户侧 LLM 信息支持 ====================
class RLMRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用语义重建策略
        self.item_embeddings = Item_Embedding("SR", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        
        # ==================== [NEW] 保存 key_words 供后续使用 ====================
        self.key_words = key_words
        # ==================== [END NEW] ====================

        # 根据对齐类型构建重建器
        if key_words['SR_aligement_type'] == 'con':
            # 对比式：语言嵌入 -> ID 嵌入
            self.reconstructor = nn.Sequential(
                nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
            )
        elif key_words['SR_aligement_type'] == 'gen':
            # 生成式：ID 嵌入 -> 语言嵌入
            self.reconstructor = nn.Sequential(
                nn.Linear(key_words['hidden_dim'], (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, self.language_dim)
            )
        
        # ==================== [NEW 2024-12-15] 用户侧 LLM 信息相关组件 ====================
        # 用户 LLM 语义嵌入（延迟加载）
        self.usr_intent_emb = None
        
        # 用户侧重建器（用户 LLM 语义 → 序列表示维度）
        # 注意：只有对比式对齐需要此组件，生成式对齐方向相反
        if key_words['SR_aligement_type'] == 'con':
            self.usr_reconstructor = nn.Sequential(
                nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
            )
        elif key_words['SR_aligement_type'] == 'gen':
            # 生成式：序列表示 → 用户 LLM 语义
            self.usr_reconstructor = nn.Sequential(
                nn.Linear(key_words['hidden_dim'], (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, self.language_dim)
            )
        # ==================== [END NEW] ====================

    def embed_ID(self, x):
        """获取物品 ID 嵌入"""
        return self.item_embeddings.ID_embeddings(x)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        return self.item_embeddings.ID_embeddings.weight

    def reconstruct_gen_loss(self, ):
        """
        生成式重建损失：ID 嵌入 -> 语言嵌入
        L = 1 - cosine_similarity(reconstructor(ID_emb), language_emb)
        """
        rec_language_embs = self.reconstructor(self.return_item_emb()[:-1])  # 去掉 padding 嵌入
        language_embs = self.item_embeddings.language_embeddings.weight
        # L2 归一化计算余弦相似度
        rec_language_embs = F.normalize(rec_language_embs, p=2, dim=-1)
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        return 1 - (rec_language_embs * language_embs).sum() / self.item_num

    def reconstruct_con_loss(self, ):
        """
        对比式重建损失：语言嵌入 -> ID 嵌入
        L = 1 - cosine_similarity(reconstructor(language_emb), ID_emb)
        """
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_ID_embs = self.reconstructor(language_embs)  # 去掉 padding 嵌入
        ID_embs = self.return_item_emb()[:-1]
        # L2 归一化计算余弦相似度
        rec_ID_embs = F.normalize(rec_ID_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (rec_ID_embs * ID_embs).sum() / self.item_num

    # ==================== [NEW 2024-12-15] 用户侧 LLM 信息相关方法 ====================
    
    def load_user_intent_embedding(self, user_intent_path):
        """
        加载用户 LLM 语义嵌入
        
        Args:
            user_intent_path: 用户 LLM 语义嵌入文件路径 (usr_intent_emb.pkl)
        """
        import os
        import pickle  # [FIX] 添加 pickle 导入
        if os.path.exists(user_intent_path):
            with open(user_intent_path, 'rb') as f:
                user_intent = pickle.load(f)
            self.usr_intent_emb = torch.tensor(user_intent, dtype=torch.float32)
            print(f"[RLMRec] Loaded user intent embedding: {self.usr_intent_emb.shape}")
        else:
            print(f"[RLMRec] Warning: User intent file not found: {user_intent_path}")
            self.usr_intent_emb = None
    
    def cal_infonce_loss(self, embeds1, embeds2, all_embeds2, temp=1.0):
        """
        InfoNCE 对比损失（与原版 RLMRec 一致）
        
        Args:
            embeds1: [B, D] 锚点嵌入（用户序列表示）
            embeds2: [B, D] 正样本嵌入（当前 batch 用户的 MLP 映射后的 LLM 语义）
            all_embeds2: [N, D] 全部样本嵌入（全部用户的 MLP 映射后的 LLM 语义，作为负样本池）
            temp: 温度参数
        
        Returns:
            InfoNCE loss
        """
        # L2 归一化
        normed_embeds1 = F.normalize(embeds1, p=2, dim=-1)
        normed_embeds2 = F.normalize(embeds2, p=2, dim=-1)
        normed_all_embeds2 = F.normalize(all_embeds2, p=2, dim=-1)
        
        # 正样本相似度（分子）
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        
        # 与全部样本的相似度（分母，包含负样本）
        # ==================== [OLD] log(sum(exp(...)))（数值不稳定）====================
        # deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        # ==================== [END OLD] ====================
        
        # ==================== [NEW 2024-12-17] logsumexp（更稳定更快）====================
        deno_term = torch.logsumexp(normed_embeds1 @ normed_all_embeds2.T / temp, dim=-1)
        # ==================== [END NEW] ====================
        
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss
    
    def user_alignment_loss_infonce(self, user_embeds, user_ids, temperature=1.0):
        """
        用户侧 InfoNCE 对齐损失（方案 A：完全复刻原版 RLMRec）
        
        Args:
            user_embeds: [B, hidden_dim] 用户序列表示 (forward 输出)
            user_ids: [B] 用户 ID
            temperature: 温度参数
        
        Returns:
            InfoNCE 对齐损失
        """
        if self.usr_intent_emb is None:
            return torch.tensor(0.0, device=user_embeds.device)
        
        # 1. 获取当前 batch 用户的 LLM 语义嵌入
        usr_llm_batch = self.usr_intent_emb[user_ids.cpu()].to(user_embeds.device)  # [B, language_dim]
        
        # 2. MLP 映射（对比式：LLM 语义 → 序列表示维度）
        usr_llm_mapped = self.usr_reconstructor(usr_llm_batch)  # [B, hidden_dim]
        
        # 3. 全部用户的 LLM 语义嵌入（作为负样本池）
        # 注意：这里需要将全部用户的 LLM 语义嵌入映射到序列表示空间
        all_usr_llm = self.usr_intent_emb.to(user_embeds.device)  # [N, language_dim]
        all_usr_llm_mapped = self.usr_reconstructor(all_usr_llm)  # [N, hidden_dim]
        
        # 4. InfoNCE 对比损失
        loss = self.cal_infonce_loss(user_embeds, usr_llm_mapped, all_usr_llm_mapped, temperature)
        return loss / user_embeds.shape[0]
    
    def user_alignment_loss_cosine(self, user_embeds, user_ids):
        """
        用户侧余弦相似度对齐损失（方案 B：与 AlphaFuse 物品侧保持一致）
        
        Args:
            user_embeds: [B, hidden_dim] 用户序列表示 (forward 输出)
            user_ids: [B] 用户 ID
        
        Returns:
            余弦相似度对齐损失：1 - cos_sim(MLP(usr_llm_emb), user_embeds)
        """
        if self.usr_intent_emb is None:
            return torch.tensor(0.0, device=user_embeds.device)
        
        # 1. 获取当前 batch 用户的 LLM 语义嵌入
        usr_llm_batch = self.usr_intent_emb[user_ids.cpu()].to(user_embeds.device)  # [B, language_dim]
        
        # 2. MLP 映射
        usr_llm_mapped = self.usr_reconstructor(usr_llm_batch)  # [B, hidden_dim]
        
        # 3. L2 归一化
        usr_llm_mapped = F.normalize(usr_llm_mapped, p=2, dim=-1)
        user_embeds_norm = F.normalize(user_embeds, p=2, dim=-1)
        
        # 4. 余弦相似度损失
        loss = 1 - (usr_llm_mapped * user_embeds_norm).sum(dim=-1).mean()
        return loss
    
    def user_alignment_loss(self, user_embeds, user_ids, mode='infonce', temperature=1.0):
        """
        用户侧对齐损失（统一接口）
        
        Args:
            user_embeds: [B, hidden_dim] 用户序列表示 (forward 输出)
            user_ids: [B] 用户 ID
            mode: 对齐模式，'infonce' 或 'cosine'
            temperature: InfoNCE 温度参数（仅 mode='infonce' 时使用）
        
        Returns:
            对齐损失
        """
        if mode == 'infonce':
            return self.user_alignment_loss_infonce(user_embeds, user_ids, temperature)
        elif mode == 'cosine':
            return self.user_alignment_loss_cosine(user_embeds, user_ids)
        else:
            raise ValueError(f"Unknown user alignment mode: {mode}, expected 'infonce' or 'cosine'")
    
    def user_alignment_loss_gen(self, user_embeds, user_ids):
        """
        用户侧生成式对齐损失（生成式：序列表示 → LLM 语义）
        
        Args:
            user_embeds: [B, hidden_dim] 用户序列表示 (forward 输出)
            user_ids: [B] 用户 ID
        
        Returns:
            生成式对齐损失：1 - cos_sim(MLP(user_embeds), usr_llm_emb)
        """
        if self.usr_intent_emb is None:
            return torch.tensor(0.0, device=user_embeds.device)
        
        # 1. 获取当前 batch 用户的 LLM 语义嵌入
        usr_llm_batch = self.usr_intent_emb[user_ids.cpu()].to(user_embeds.device)  # [B, language_dim]
        
        # 2. MLP 映射（生成式：序列表示 → LLM 语义维度）
        rec_usr_llm = self.usr_reconstructor(user_embeds)  # [B, language_dim]
        
        # 3. L2 归一化
        rec_usr_llm = F.normalize(rec_usr_llm, p=2, dim=-1)
        usr_llm_batch = F.normalize(usr_llm_batch, p=2, dim=-1)
        
        # 4. 余弦相似度损失
        loss = 1 - (rec_usr_llm * usr_llm_batch).sum(dim=-1).mean()
        return loss
    # ==================== [END NEW 2024-12-15] ====================


# =============================================================================
# UniSRec：语言嵌入 + MoE 适配器
# 使用 Mixture of Experts (MoE) 替代简单的 MLP
# MoE 可以根据输入动态选择不同的专家网络
# =============================================================================
class UniSRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用自适应投影策略
        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        # MoE 适配器：8 个专家，Dropout 0.2
        self.adapter = MoEAdaptorLayer(
            8,  # 专家数量
            [self.language_dim, key_words['hidden_dim']],  # 输入输出维度
            0.2  # Dropout 概率
        )

    def embed_ID(self, x):
        """获取物品嵌入：语言嵌入 -> MoE 适配器"""
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)

    def return_item_emb(self, ):
        """返回全量物品嵌入"""
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs)


# =============================================================================
# LLMESR：双视图模型
# =============================================================================
# 同时使用 ID 嵌入和语言嵌入，通过交叉注意力进行交互
# 最终输出是 ID 视图和语言视图的拼接
#
# 【物品侧 LLM 信息】（原有）
#   - 使用 3large_emb.pickle / itm_intent_emb.pkl（物品 LLM 语义嵌入）
#   - 通过 Adapter (MLP) 降维后作为语言视图输入
#   - 与 ID 视图通过交叉注意力交互
#   - reg_loss: 双视图对比正则化损失
#
# 【用户侧 LLM 信息】（RASD - Retrieval Augmented Self-Distillation）
#   使用 sim_user_100.pkl（相似用户列表，形状 [N_users, 100]）
#   
#   数据流：
#     sim_user_100[user_id]  →  [相似用户 ID 列表]  →  相似用户序列
#                                                          ↓
#                                                    forward(sim_seqs)
#                                                          ↓
#                                            Contrastive / KD 蒸馏损失
#
#   与 RLMRec 用户侧对齐的区别：
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │  RLMRec 用户侧对齐              vs        LLMESR RASD               │
#   ├──────────────────────────────────────────────────────────────────────┤
#   │  usr_intent_emb.pkl                    sim_user_100.pkl             │
#   │  (用户 LLM 语义嵌入, 3072维)            (相似用户列表, 100个)         │
#   │  直接语义对齐                           相似用户蒸馏                  │
#   │  MLP 映射后 InfoNCE/Cosine             序列表示对比学习              │
#   └──────────────────────────────────────────────────────────────────────┘
#
#   RASD 命令行参数：
#     --use_rasd:       是否启用 RASD (默认 False)
#     --alpha_rasd:     RASD 损失权重 (默认 0.1)
#     --sim_user_num:   使用的相似用户数量 K (默认 10)
#     --user_sim_func:  蒸馏函数 cl/kd (默认 cl)
#
#   损失函数：
#     L_total = L_main + β * L_reg + α_rasd * L_rasd
#     其中：
#       - L_main: 主损失（InfoNCE）
#       - L_reg:  双视图对比正则化损失（原有）
#       - L_rasd: RASD 蒸馏损失（用户侧，新增）
#
# =============================================================================
class LLMESR(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        # 使用双视图策略
        self.item_embeddings = Item_Embedding("Dual_view", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        # 语言嵌入适配器：降维到隐藏维度
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, int(self.language_dim / 2)),
            nn.Linear(int(self.language_dim / 2), key_words['hidden_dim'])
        )

        # 交叉注意力层：ID 和语言视图相互增强
        self.language2ID = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)  # 语言 -> ID
        self.ID2language = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)  # ID -> 语言

        # 对比损失：用于正则化两个视图
        self.reg = Contrastive_Loss2()

    def embed_ID_text(self, x):
        """获取 ID 嵌入和语言嵌入"""
        language_embs = self.item_embeddings.language_embeddings(x)
        ID_embs = self.item_embeddings.ID_embeddings(x)
        return ID_embs, self.adapter(language_embs)

    def embed_ID(self, x):
        """获取拼接后的嵌入：[ID_emb, language_emb]"""
        ID_embs, language_embs = self.embed_ID_text(x)
        return torch.cat([ID_embs, language_embs], dim=-1)

    def return_item_emb(self, ):
        """返回全量物品嵌入（拼接）"""
        ID_embs = self.item_embeddings.ID_embeddings.weight
        language_embs = self.item_embeddings.language_embeddings.weight
        language_embs = self.adapter(language_embs)
        return torch.cat([ID_embs, language_embs], dim=-1)  # [item_num+1, 2*D]

    def forward(self, sequences):
        """
        前向传播：双视图编码

        流程：
        1. 分别获取 ID 嵌入和语言嵌入
        2. 交叉注意力：ID <-> 语言
        3. 分别经过 Transformer 编码
        4. 拼接两个视图的输出
        """
        # 获取两种嵌入
        inputs_id_emb, inputs_text_emb = self.embed_ID_text(sequences)
        inputs_text_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        inputs_id_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))

        text_seq = self.emb_dropout(inputs_text_emb)
        # id_seq = self.emb_dropout(inputs_text_emb)  # ❌ [BUG] 原代码错误：应该用 inputs_id_emb
        # ==================== [FIX] 修复 id_seq 使用错误的嵌入 ====================
        id_seq = self.emb_dropout(inputs_id_emb)  # ✅ 正确：使用 ID 嵌入
        # ==================== [END FIX] ====================

        # 交叉注意力：两个视图相互增强
        cross_id_seqs = self.language2ID(text_seq, id_seq, sequences, self.item_num)  # 语言 -> ID
        cross_text_seqs = self.ID2language(id_seq, text_seq, sequences, self.item_num)  # ID -> 语言
        cross_id_seqs = 1 * cross_id_seqs + 0 * id_seq  # 残差连接（权重 1:0）
        cross_text_seqs = 1 * cross_text_seqs + 0 * text_seq

        # ID 视图的 Transformer 编码
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_id_seqs *= mask
        seq_normalized = self.ln_1(cross_id_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_id_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        id_logits = ff_out[:, -1].squeeze()  # [B, D]

        # 语言视图的 Transformer 编码
        # mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_text_seqs *= mask
        seq_normalized = self.ln_1(cross_text_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_text_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        text_logits = ff_out[:, -1].squeeze()  # [B, D]

        # 拼接两个视图的输出
        log_feats = torch.cat([id_logits, text_logits], dim=-1)  # [B, 2*D]

        return log_feats

    def reg_loss(self, sequences):
        """
        正则化损失：对齐 ID 视图和语言视图
        使用对比学习损失鼓励两个视图的一致性
        """
        unfold_item_id = torch.masked_select(sequences, sequences != self.item_num)
        language_emb, id_emb = self.embed_ID_text(unfold_item_id)
        reg_loss = self.reg(language_emb, id_emb)
        return reg_loss

    # ==================== [NEW] RASD (Retrieval Augmented Self-Distillation) 损失 ====================
    def calculate_rasd_loss(self, sequences, sim_seqs, user_sim_func='cl'):
        """
        计算 RASD 对齐损失（LLM-ESR 原始方法）
        
        思路：用相似用户的表示作为"教师"，让当前用户的表示向教师靠拢
        
        Args:
            sequences: [B, S] 当前用户的物品序列
            sim_seqs: [B, K, S] 相似用户的物品序列（K 个相似用户）
            user_sim_func: 'cl' (对比学习) 或 'kd' (知识蒸馏/MSE)
        
        Returns:
            rasd_loss: 标量损失值
        """
        B, K, S = sim_seqs.shape
        
        # 1. 获取当前用户的表示
        h_u = self.forward(sequences)  # [B, 2*D]
        
        # 2. 获取相似用户的表示
        sim_seqs_flat = sim_seqs.view(B * K, S)  # [B*K, S]
        h_sim = self.forward(sim_seqs_flat)  # [B*K, 2*D]
        
        # 3. 关键：stop gradient，相似用户作为"教师"不更新梯度
        h_sim = h_sim.detach()
        
        # 4. 重塑并取平均
        h_sim = h_sim.view(B, K, -1)  # [B, K, 2*D]
        h_sim_avg = h_sim.mean(dim=1)  # [B, 2*D] 多个相似用户的平均表示
        
        # 5. 计算对齐损失
        if user_sim_func == 'cl':
            # 对比学习损失
            rasd_loss = self.reg(h_u, h_sim_avg)
        elif user_sim_func == 'kd':
            # 知识蒸馏损失 (MSE)
            rasd_loss = F.mse_loss(h_u, h_sim_avg)
        else:
            raise ValueError(f"Unknown user_sim_func: {user_sim_func}")
        
        return rasd_loss
    # ==================== [END NEW] ====================

    # ==================== [FIX 2024-12-20] 添加缺失的 calculate_infonce_loss（与 GRU/BERT4Rec 版本一致）====================
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature):
        """
        计算 InfoNCE 对比学习损失（双视图版本）
        
        注意：LLMESR 的 forward 返回 [B, 2D]，embed_ID 也返回 [B, 2D]
        """
        batch_size = target.shape[0]
        
        # 负采样
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        # L2 归一化
        logits = self.forward(sequences)  # [B, 2D]
        logits = F.normalize(logits, p=2, dim=-1)

        # 正负样本嵌入（使用 embed_ID 返回 [B, 2D]）
        target_emb = self.embed_ID(target)  # [B, 2D]
        target_emb = F.normalize(target_emb, p=2, dim=-1)
        
        neg_emb = self.embed_ID(target_neg)  # [B, neg_ratio, 2D]
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)

        # 计算相似度
        pos_sim = torch.sum(logits * target_emb, dim=-1, keepdim=True) / temperature  # [B, 1]
        neg_sim = torch.bmm(neg_emb, logits.unsqueeze(-1)).squeeze(-1) / temperature  # [B, neg_ratio]

        # InfoNCE 损失
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1 + neg_ratio]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(all_sim, labels)

        return loss
    # ==================== [END FIX] ====================


# =============================================================================
# AlphaFuse：本文提出的方法
# 核心创新：在语言嵌入的"零空间"中注入 ID 信息
#
# 数学原理：
# 1. 对语言嵌入矩阵进行 SVD 分解，得到主成分空间和零空间
# 2. 零空间是语言嵌入方差较小的维度，包含较少的语义信息
# 3. 在零空间中注入 ID 信息，避免破坏语义信息
#
# 融合方式：
# - cover=False（默认）：fuse_emb = language_emb + ID_emb（零空间部分相加）
# - cover=True：fuse_emb = [language_emb, ID_emb]（拼接）
# =============================================================================
class AlphaFuse(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)  # 调用 SASRec_backbone 的 init()
        # 使用 AlphaFuse 嵌入策略
        #
        # Item_Embedding(
        #   (language_embeddings): Embedding(18358, 128, padding_idx=18357)
        #   (ID_embeddings): Embedding(18358, 64)
        # )
        self.item_embeddings = Item_Embedding("AF", **key_words)  # 调用 Item_Embedding 的 init()
        # self.language_dim = self.item_embeddings.language_dim
        # 调用链
        # 1. 创建 Item_Embedding 实例
        # self.item_embeddings = Item_Embedding("AF", **key_words)
        #
        # 2. 在 __init__ 中调用
        # self.construct_item_embeddings("AF", **key_words)
        #
        # 3. AF 模式调用 semantic_space_decomposion
        # cliped_language_embs = self.semantic_space_decomposion(hidden_dim, **key_words)
        #                             ↑ 这里设置了 self.nullity = 64
        #
        # 4. 现在 item_embeddings 就有 nullity 属性了
        # self.item_embeddings.nullity  # 64
        self.nullity = self.item_embeddings.nullity  # 零空间维度  64
        self.cover = key_words["cover"]  # 融合模式   False

    def embed_ID(self, x):
        """
        获取融合后的物品嵌入

        融合策略：
        - cover=False：语言嵌入的零空间部分 + ID 嵌入
        - cover=True：语言嵌入和 ID 嵌入拼接
        """
        language_embs = self.item_embeddings.language_embeddings(x)  # x:(B,S) ——> language_embs:[B, S, D] (256,10,128)
        # fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings(x)  # [B, S, nullity] (256,10,64)
        if self.cover:
            # 拼接模式：[语言嵌入, ID嵌入]

            # 注意：
            # 根据论文 4.1.5 节：
            #
            # "drop the original values in the null space, and replace them with E_ID"
            # 论文描述的是替换（replace），而这段代码是相加（add）
            # 如果要严格按论文实现，应该是：
            # fuse_embs[..., -self.nullity:] = ID_embs  # 替换，而非相加
            return torch.cat((language_embs, ID_embs), dim=-1)  # (256, 10, 192)
        else:
            # 相加模式：在零空间维度（最后 nullity 维）相加
            fuse_embs = language_embs.clone()  # (256,10,128)
            # language_embs: [███████████████|░░░░░░░░░░░░░░░░] 128维
            #                 ↑语义64维      ↑零空间64维
            # ID_embs:                       [████████████████] 64维
            #                               ↓ 相加
            # 输出:          [███████████████|████████████████] 128维
            #                ↑语义不变       ↑零空间+ID

            # ...（ellipsis）在 numpy / PyTorch 里表示：“前面的所有维度都不要动，我只关心最后这几维
            # x[..., -k:] 等价于：x[:, -k:]
            fuse_embs[..., -self.nullity:] = language_embs[..., -self.nullity:] + ID_embs  # (256,10,128)
        return fuse_embs

    def return_item_emb(self, ):
        """返回全量融合物品嵌入"""
        # 这里为什么取权重 .weight
        # 因为  return_item_emb 需要返回所有物品的嵌入，而不是查找特定物品
        language_embs = self.item_embeddings.language_embeddings.weight  # (18358,128)
        # fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings.weight  # (18358,64)
        if self.cover:
            # 拼接模式
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            # 相加模式
            fuse_embs = language_embs.clone()
            fuse_embs[..., -self.nullity:] = language_embs[..., -self.nullity:] + ID_embs  # (18358,128)
        return fuse_embs


# =============================================================================
# [NEW] ProAlign-FA: 原型对齐的序列推荐模型
# 核心创新：
# 1. 利用 LLM 的"反事实未来推理"生成用户未来意图 z_next
# 2. 通过共享原型空间 P 进行知识蒸馏
# 3. 门控自适应融合微观状态 h_u 和宏观意图 r_u
# 4. 推理时无需 LLM，延迟与原始 SASRec 相当
#
# ==================== [NEW 2024-12-15] 消融实验控制 ====================
# 通过 --use_user_intent 参数控制是否使用用户侧 LLM 信息：
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  ProAlign (Item-only)           vs        ProAlign (Full)           │
# ├──────────────────────────────────────────────────────────────────────┤
# │  --use_user_intent False              --use_user_intent True        │
# │  Item 侧: ✅ itm_intent_emb.pkl       Item 侧: ✅ itm_intent_emb.pkl │
# │  User 侧: ❌ usr_intent_emb.pkl       User 侧: ✅ usr_intent_emb.pkl │
# │  L_align = 0                          L_align = 正常计算              │
# ├──────────────────────────────────────────────────────────────────────┤
# │  用途：消融实验，证明 User 侧 LLM 信息的贡献                            │
# │  对比基线：LLMInit（只用 Item 侧 LLM 信息）                            │
# └──────────────────────────────────────────────────────────────────────┘
# ==================== [END NEW] 消融实验控制 ====================
# =============================================================================
class ProAlign(SASRec_backbone):
    """
    ProAlign-FA + SASRec 模型（适配 AlphaFuse 框架）

    架构：
    - Base Model: SASRec (继承自 SASRec_backbone)
    - 原型矩阵 P: 从物品意图 embedding 聚类初始化
    - Adapter: 将 LLM 意图 (3072维) 降维到 hidden_size
    - 门控融合: 自适应融合微观状态 h_u 和宏观意图 r_u
    
    消融实验:
    - --use_user_intent True  : ProAlign (Full) - 使用 Item + User 侧 LLM 信息
    - --use_user_intent False : ProAlign (Item-only) - 仅使用 Item 侧 LLM 信息
    """

    def __init__(self, device, **key_words):
        super().__init__(device, **key_words) # 调用 class SASRec_backbone 的 init()

        # ==================== 保存参数 ====================
        self.key_words = key_words

        # ==================== ID Embedding（与 SASRec 一致）====================
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num + 1,  # +1 for padding
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        ) # Embedding(12102, 128, padding_idx=12101)

        # ==================== ProAlign-FA 超参数 ====================
        # 原型数量 K
        self.num_prototypes = key_words.get('num_prototypes', 64) # 64
        # 温度参数 τ（控制 softmax 锐度）
        self.temperature = key_words.get('proto_temperature', 0.04) # 0.04
        
        # ==================== [NEW 2025-01-17] 原型机制消融开关 ====================
        # no_prototype=True: 禁用原型机制（消融实验：w/o Prototype）
        # no_prototype=False (默认): 正常使用原型机制
        self.no_prototype = key_words.get('no_prototype', False)
        if self.no_prototype:
            print("[ProAlign] ⚠️ ABLATION MODE: Prototype mechanism DISABLED (w/o Prototype)")
        # ==================== [END NEW 2025-01-17] ====================
        # 对齐损失权重 α
        self.alpha = key_words.get('alpha', 0.1) # 0.1
        # 聚类损失权重 β
        self.beta_proto = key_words.get('beta_proto', 0.01) # 0.01
        # LLM 意图维度
        self.llm_dim = key_words.get('llm_dim', 3072) # 3072
        # 融合模式: 'add' 或 'concat'
        self.fusion_mode = key_words.get('fusion_mode', 'concat') # 'concat'
        # 加法融合时的语义权重
        self.semantic_weight = key_words.get('semantic_weight', 0.5) # 0.5
        
        # ==================== [NEW-MultiHead] 多头解耦原型参数 ====================
        # 原型注意力头数 H（解决"语义中和"问题）
        # 核心思想：将 D 维切分为 H 个子空间，每个 head 独立寻址
        # Head 1: 关注"功能"维度, Head 2: 关注"品牌/价格"维度, etc.
        self.num_heads_proto = key_words.get('num_heads_proto', 1)  # 默认1=单头（向后兼容）
        self.head_dim = self.hidden_dim // self.num_heads_proto # 128//1=128
        assert self.hidden_dim % self.num_heads_proto == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads_proto ({self.num_heads_proto})"
        # ==================== [END NEW-MultiHead] ====================

        # ==================== 原型矩阵 P: [K, hidden_size] ====================

        # 原型矩阵 P: [K, hidden_size]
        # 初始化方式：从物品意图 embedding 聚类得到
        # nn.Parameter 把一个 Tensor 变成“这个模型的可学习参数”
        self.prototypes = nn.Parameter(torch.zeros(self.num_prototypes, self.hidden_dim)) # (64,128)

        # ==================== Adapter: LLM 意图降维 ====================

        # Sequential(
        #   (0): Linear(in_features=3072, out_features=768, bias=True)
        #   (1): ReLU()
        #   (2): Linear(in_features=768, out_features=128, bias=True)
        # )
        self.adapter = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim // 4),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 4, self.hidden_dim)
        )

        # ==================== 门控网络（拼接模式使用）====================

        # 门控网络: 自适应融合权重
        # 输入: [h_u; r_u] (2 * hidden_size)
        # 输出:
        #   - 标量门控 (默认): g ∈ (0, 1)^1，全局统一加权

        # Sequential(
        #   (0): Linear(in_features=256, out_features=128, bias=True)
        #   (1): ReLU()
        #   (2): Linear(in_features=128, out_features=1, bias=True)
        #   (3): Sigmoid()
        # )
        #
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # ==================== 可学习缩放因子 ====================
        self.macro_scale = nn.Parameter(torch.tensor(1.0))

        # ==================== [NEW 2025-01-17] 推理效率优化状态 ====================
        self._inference_mode = False  # 推理模式标志
        self._item_proto_cache = None  # 物品原型表示缓存
        self._fused_item_cache = None  # 融合物品嵌入缓存
        # ==================== [END NEW] ====================

        # ==================== [NEW-SLSI] 序列级语义注入参数 ====================
        # 是否启用 SLSI（在每个序列位置注入语义信息）
        self.use_slsi = key_words.get('use_slsi', True)
        # SLSI 语义注入权重
        self.slsi_weight = key_words.get('slsi_weight', 1.0)
        # ==================== [NEW-SLSI-ContextAware] 上下文感知 SLSI ====================
        # 是否启用上下文感知 SLSI（结合历史位置的信息）
        # False: 每个位置独立做原型寻址（默认，简单高效）
        # True: 结合历史位置的累积表示做原型寻址（上下文感知，更复杂）
        self.slsi_context_aware = key_words.get('slsi_context_aware', False)
        # ==================== [END NEW-SLSI-ContextAware] ====================
        # ==================== [END NEW-SLSI] ====================

        # # ==================== [NEW] Forward Predictor (前向预测器) (已注释，恢复原始状态) ====================
        # # h_u → Predictor → h_pred, 让模型"学会预测未来意图"
        # self.forward_predictor = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # )
        # # 是否使用前向预测器（可通过参数控制，便于消融实验）
        # self.use_forward_predictor = key_words.get('use_forward_predictor', True)

        # ==================== [NEW 2024-12-31] 动态注意力融合 ====================
        # 替代简单的加法/门控，实现动态的原型寻址
        # 与 ProAlign_BERT4Rec 保持一致
        self.use_attn_fusion = key_words.get('use_attn_fusion', True)
        if self.use_attn_fusion:
            self.proto_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,  # 4头捕捉不同维度的意图
                batch_first=True
            )
        # ==================== [END NEW 2024-12-31] ====================

        # ==================== [NEW 2024-12-31] 课程学习参数 ====================
        # 与 ProAlign_BERT4Rec 保持一致
        self.warmup_epochs = key_words.get('warmup_epochs', 5)
        self.current_epoch = 0  # 由 train.py 更新
        # ==================== [END NEW 2024-12-31] ====================

        # ==================== [NEW 2024-12-31] 语义困难负样本 ====================
        # 与 ProAlign_BERT4Rec 保持一致
        self.hard_neg_indices = None  # [V, K] 每个物品的 Top-K 相似物品
        self.hard_neg_top_k = key_words.get('hard_neg_top_k', 10)
        self.item_intent_emb_for_align = None  # 保存原始物品意图嵌入供 L_align 使用
        # ==================== [END NEW 2024-12-31] ====================

        # ==================== 意图 Embedding（延迟加载）====================
        self.user_intent_emb = None  # 用户未来意图
        self.item_intent_emb = None  # 物品意图（用于初始化原型）
        self.item_emb_reduced = None  # PCA 降维后的物品 embedding
        self.prototype_initialized = False

        # ==================== 初始化权重 ====================
        # ==================== [OLD] 原始初始化方式（已注释）====================
        # self._init_proalign_weights()
        # ==================== [END OLD] ====================
        
        # ==================== [NEW 2024-12-17] BSARec 风格统一初始化 ====================
        # 使用 apply 统一初始化所有子模块，然后处理特殊参数
        self.apply(self._init_bsarec_weights)
        self._init_special_params()
        # ==================== [END NEW] ====================

    def _init_proalign_weights(self):
        """初始化 ProAlign 特有的权重"""
        # ID Embedding 初始化
        nn.init.normal_(self.item_embeddings.weight, 0, 0.02)
        # Adapter 初始化
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Gate 初始化
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # # [NEW] Forward Predictor 初始化 (已注释，恢复原始状态)
        # for module in self.forward_predictor:
        #     if isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, 0, 0.02)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)

    # ==================== [NEW 2024-12-17] BSARec 风格统一初始化 ====================
    # 参考 BSARec 的 _abstract_model.py 中的 init_weights 方法
    # 优点：
    #   1. 统一初始化所有子模块（Embedding/Linear/LayerNorm/GRU）
    #   2. 自动处理 padding_idx 行清零
    #   3. 预留 GRU 支持
    
    def _init_bsarec_weights(self, module):
        """
        BSARec-style: 统一初始化所有子模块参数
        - Embedding: normal_(0, 0.02) + padding 行清零
        - Linear: normal_(0, 0.02) + bias 清零
        - LayerNorm: weight=1, bias=0
        - GRU: xavier_uniform (input-hidden), orthogonal (hidden-hidden)
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 关键：将 padding_idx 对应的行清零
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
                    
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.GRU):
            # 预留：后续添加 GRU 时自动初始化
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def _init_special_params(self):
        """
        初始化不会被 module.apply 覆盖的参数（如 nn.Parameter）
        """
        # prototypes 是 nn.Parameter，不属于某个子 module 的 weight
        with torch.no_grad():
            if torch.all(self.prototypes == 0):
                nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        
        # macro_scale 保持默认值 1.0（已在定义时初始化）
        
        # 保险：再次确保 padding 行为 0
        with torch.no_grad():
            self.item_embeddings.weight[self.item_num].fill_(0.0)
    # ==================== [END NEW] BSARec 风格统一初始化 ====================

    def load_intent_embeddings(self, user_intent_path, item_intent_path):
        """
        加载预计算的 LLM 意图 embedding

        Args:
            user_intent_path: 用户未来意图 embedding 路径
            item_intent_path: 物品意图 embedding 路径
        """
        # 加载用户意图 embedding
        if os.path.exists(user_intent_path):
            with open(user_intent_path, 'rb') as f:
                user_intent = pickle.load(f)
            self.user_intent_emb = torch.tensor(user_intent, dtype=torch.float32) # (22363,3072)
            print(f"[ProAlign] Loaded user intent embedding: {self.user_intent_emb.shape}") # [ProAlign] Loaded user intent embedding: torch.Size([22363, 3072])
        else:
            print(f"[ProAlign] Warning: User intent file not found: {user_intent_path}")

        # 加载物品意图 embedding
        if os.path.exists(item_intent_path):
            with open(item_intent_path, 'rb') as f:
                item_intent = pickle.load(f)
            self.item_intent_emb = item_intent  # numpy array   # (12101,3072)
            print(f"[ProAlign] Loaded item intent embedding: {self.item_intent_emb.shape}") # [ProAlign] Loaded item intent embedding: (12101, 3072)
        else:
            print(f"[ProAlign] Warning: Item intent file not found: {item_intent_path}")

    def initialize_item_embeddings(self):
        """
        使用 LLM 语义向量初始化 ID Embedding (Semantic Warm-up)
        """

        #                 item_intent_emb [12101, 3072]
        #                     (原始 LLM 向量)
        #                           │
        #                           ▼
        #                    PCA 降维 (只执行一次)
        #                           │
        #                           ▼
        #               item_emb_reduced [12101, 128]
        #                       (缓存)
        #                     ┌────┴────┐
        #                     ▼         ▼
        #             ID Embedding   K-Means 聚类
        #                初始化          │
        #                   │           ▼
        #                   ▼       Prototypes [K, 128]
        #          item_embeddings     (原型矩阵)
        #            [12102, 128]

        if self.item_intent_emb is None:
            print("[ProAlign] Warning: No item intent found, using random init.")
            return

        print("[ProAlign] Initializing ID Embeddings from LLM semantics...")

        # 1. PCA 降维
        if self.item_emb_reduced is None:
            pca = PCA(n_components=self.hidden_dim) # PCA(n_components=128)
            self.item_emb_reduced = pca.fit_transform(self.item_intent_emb)  # self.item_emb_reduced （PCA 结果缓存在 self.item_emb_reduced）
            print(f"  PCA: {self.item_intent_emb.shape} -> {self.item_emb_reduced.shape}") #  PCA: (12101, 3072) -> (12101, 128)

        # 2. 归一化
        with torch.no_grad():
            pretrained_weight = torch.tensor(self.item_emb_reduced, dtype=torch.float32) # numpy → torch.Tensor   [V, D]  (12101,128)
            pretrained_weight = F.normalize(pretrained_weight, p=2, dim=-1) # (12101,128) [V, D], 模长=1

            # 3. 拼接 Padding (Index = item_num)
            padding = torch.zeros(1, self.hidden_dim) # [1, 128]
            # 使用零向量作为 Padding 是正确的！
            # 原因解释
            # 1. 数学意义
            # Padding 的作用：表示"无交互"或"无效位置"
            #
            # 零向量的特性：
            # - 点积: h_u · 0 = 0  → 不影响得分计算
            # - 注意力: 零向量不贡献任何信息
            #
            # 2. 与 nn.Embedding 的 padding_idx 一致
            # self.item_embeddings = nn.Embedding(item_num + 1, hidden_dim, padding_idx=item_num)
            # 当设置 padding_idx=12101 时，PyTorch 会自动：
            # 将 index=12101 的向量初始化为零向量
            # 训练时不更新这个位置的梯度
            #
            # 3. 我们手动初始化也用零向量
            # padding = torch.zeros(1, self.hidden_dim)  # 零向量，与 padding_idx 的默认行为一致

            # 结果：
            # new_weight[0:12101] = 物品 0-12100 的语义向量
            # new_weight[12101] = 零向量 (Padding)

            # padding_idx=12101 表示 index 12101 是 Padding
            # torch.cat([pretrained_weight, padding], dim=0) 把零向量放在 index 12101
            new_weight = torch.cat([pretrained_weight, padding], dim=0)  # [item_num+1, D]   [12102, 128]

            # 4. 覆盖权重
            if new_weight.shape[0] == self.item_embeddings.weight.shape[0]:
                self.item_embeddings.weight.data.copy_(new_weight)
                # 验证更新的是 ID Embedding，不是 LLM的Embedding
                print('更新的是 ID Embedding，不是 LLM的Embedding', self.item_embeddings.weight.requires_grad)  # True ✅
                print(f"   ID Embeddings initialized! Shape: {new_weight.shape}")
            else:
                print(f"   Shape mismatch: {new_weight.shape} vs {self.item_embeddings.weight.shape}")

    def initialize_prototypes(self):
        """
        使用物品意图 embedding 初始化原型矩阵（K-Means 聚类）
        """
        if self.prototype_initialized:
            return

        if self.item_emb_reduced is None: # ← 检查缓存
            if self.item_intent_emb is None:
                print("[ProAlign] Warning: No item intent, using random prototypes.")
                nn.init.normal_(self.prototypes.data, 0, 0.02)
                self.prototype_initialized = True
                return
            # PCA 降维
            pca = PCA(n_components=self.hidden_dim)
            self.item_emb_reduced = pca.fit_transform(self.item_intent_emb)

        print(f"[ProAlign] Initializing prototypes with K-Means (K={self.num_prototypes})...")

        # K-Means 聚类
        kmeans = KMeans(n_clusters=self.num_prototypes, # 原型数量 K（比如 64）
                        random_state=42,                # 随机种子，保证可复现
                        n_init=10                       # 运行 10 次不同的随机初始化，选最好的那次
                        ) # KMeans(n_clusters=64, n_init=10, random_state=42)
        # item_emb_reduced：PCA 之后的物品语义向量
        # 形状是 [V, D]，在你这里是 [12101, 128]
        # 每一行：一个物品在 128 维语义空间里的向量
        #
        # fit 的作用：
        # 用 KMeans 算法，把这 12101 个点分成 K 个簇，并求出每个簇的中心
        kmeans.fit(self.item_emb_reduced) # ← 使用已缓存的 PCA 结果做 K-Means  (12101, 128)
        # 把 KMeans 学到的 K 个簇中心 取出来，当作“语义质心”
        centroids = kmeans.cluster_centers_    # (64,128)  64 个簇中心，每个簇中心是一个 128 维向量

        # 赋值原型矩阵
        with torch.no_grad():
            centroids_tensor = torch.tensor(centroids, dtype=torch.float32)   # (64,128)
            self.prototypes.data = F.normalize(centroids_tensor, p=2, dim=-1) # (64,128)

        # ==================== [NEW] 根据参数决定是否冻结原型 ====================
        # 原始代码: self.prototypes.requires_grad = False (始终冻结)

        # get('freeze_prototypes', True) 的意思：
        #
        # 去字典里查 key 'freeze_prototypes'
        # 如果 查到了，就返回它对应的值，比如 True 或 False
        # 如果 没查到这个 key，就返回后面的默认值 True
        freeze_proto = self.key_words.get('freeze_prototypes', True)
        if freeze_proto:
            self.prototypes.requires_grad = False
            print(f"  ✅ Prototypes initialized and FROZEN. Shape: {self.prototypes.shape}") # (64,128)
        else:
            self.prototypes.requires_grad = True
            print(f"  ✅ Prototypes initialized and TRAINABLE. Shape: {self.prototypes.shape}")
        # ==================== [END NEW] ====================

        # 释放内存
        self.item_intent_emb = None
        # 作用：防止重复初始化原型矩阵
        #
        # 原因：
        # K-Means 聚类是耗时操作
        # 如果多次调用 initialize_prototypes() ，不应该重复执行
        # 用布尔标志防止重复初始化
        self.prototype_initialized = True

    # ==================== [NEW 2024-12-31] 预计算困难负样本（与 BERT4Rec 版本一致）====================
    def precompute_hard_negatives(self, top_k=10):
        """
        预计算每个物品的 Top-K 语义相似物品作为困难负样本

        原理：
        - 随机负样本太弱（"鼠标" vs "卫生纸"），模型走捷径
        - 困难负样本（"游戏鼠标" vs "办公鼠标"）强迫模型学习细粒度区分

        Args:
            top_k: 每个物品保留的困难负样本数量
        """
        if self.item_intent_emb is None:
            print("[ProAlign] Warning: item_intent_emb not loaded, skip hard negative precomputation")
            return

        # 使用原始物品意图嵌入计算相似度
        item_emb = self.item_intent_emb  # [V, llm_dim]
        # [FIX 2024-12-31] numpy array → torch.Tensor
        if isinstance(item_emb, np.ndarray):
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
        item_emb_norm = F.normalize(item_emb, p=2, dim=-1)

        # 计算物品间余弦相似度矩阵
        V = item_emb.size(0)
        batch_size = 1000  # 分批计算，避免 OOM

        hard_neg_indices = []
        for i in range(0, V, batch_size):
            end_i = min(i + batch_size, V)
            batch_emb = item_emb_norm[i:end_i]  # [batch, D]
            sim_matrix = torch.matmul(batch_emb, item_emb_norm.t())  # [batch, V]

            # 排除自己（设为极小值）
            for j in range(end_i - i):
                sim_matrix[j, i + j] = -1e9

            # 取 Top-K 最相似的物品
            _, topk_indices = torch.topk(sim_matrix, top_k, dim=-1)  # [batch, K]
            hard_neg_indices.append(topk_indices)

        self.hard_neg_indices = torch.cat(hard_neg_indices, dim=0)  # [V, K]
        self.hard_neg_top_k = top_k
        self.item_intent_emb_for_align = item_emb  # 保存供 L_align 使用
        print(f"[ProAlign] Hard negatives precomputed: {V} items × Top-{top_k}")
    # ==================== [END NEW 2024-12-31] ====================

    # ==================== [NEW] 原型寻址工具函数 ====================
    def _proto_address(self, x, normalize_proto=False):
        """
        对任意形状的输入 x[..., D] 做原型寻址，输出 r[..., D]
        支持单头 / 多头
        
        Args:
            x: [..., D] 输入嵌入
            normalize_proto: 是否对原型进行 L2 归一化
        
        Returns:
            r: [..., D] 原型加权表示
        """
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.view(-1, D)  # [N, D]
        N = x_flat.size(0)

        P = self.prototypes
        if normalize_proto:
            P = F.normalize(P, p=2, dim=-1)

        if self.num_heads_proto == 1:
            # 单头模式
            score = torch.matmul(x_flat, P.t()) / self.temperature  # [N, K]
            pi = F.softmax(score, dim=-1)
            r = torch.matmul(pi, P)  # [N, D]
        else:
            # 多头模式：x -> [N, H, d]，P -> [K, H, d]
            x_h = x_flat.view(N, self.num_heads_proto, self.head_dim)                 # [N, H, d]
            P_h = P.view(self.num_prototypes, self.num_heads_proto, self.head_dim)   # [K, H, d]
            scores = torch.einsum('nhd,khd->nhk', x_h, P_h) / self.temperature        # [N, H, K]
            pi = F.softmax(scores, dim=-1)                                           # [N, H, K]
            r_h = torch.einsum('nhk,khd->nhd', pi, P_h)                               # [N, H, d]
            r = r_h.reshape(N, D)                                                    # [N, D]

        r = r * self.macro_scale
        return r.view(*orig_shape)
    # ==================== [END NEW] ====================

    # ==================== [NEW 2025-01-17] 推理效率优化 ====================
    def precompute_for_inference(self):
        """
        推理前预计算物品原型表示，实现与 SASRec 同量级的推理效率
        
        优化原理：
        1. SLSI 的原型寻址从 O(B×S×D×K) 降为 O(1) 查表
        2. 物品侧融合嵌入预计算，推理时直接使用
        3. 总推理 GFLOPs: 0.0266 → ~0.015（与 SASRec 同量级）
        
        调用时机：在 model.eval() 后、推理前调用一次
        
        Usage:
            model.eval()
            model.precompute_for_inference()
            with torch.no_grad():
                scores = model.predict(sequences)
        """
        if self.no_prototype:
            print("[ProAlign-Efficient] Prototype disabled, skip precompute")
            self._inference_mode = True
            return
            
        with torch.no_grad():
            # ========== 1. 预计算所有物品的原型表示 r_i ==========
            item_emb = self.item_embeddings.weight  # [V+1, D]
            V = item_emb.size(0)
            P = self.prototypes
            
            if self.num_heads_proto == 1:
                score_all = torch.matmul(item_emb, P.t()) / self.temperature
                pi_all = F.softmax(score_all, dim=-1)
                r_all = torch.matmul(pi_all, P)
            else:
                item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)
                proto_heads = P.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature
                pi = F.softmax(scores, dim=-1)
                r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)
                r_all = r_heads.reshape(V, self.hidden_dim)
            
            r_all = r_all * self.macro_scale
            self._item_proto_cache = r_all  # [V+1, D] 物品原型表示缓存
            
            # ========== 2. 预计算融合后的物品嵌入（用于打分）==========
            if self.fusion_mode == 'add':
                self._fused_item_cache = item_emb + self.semantic_weight * r_all
            else:
                self._fused_item_cache = torch.cat([item_emb, r_all], dim=-1)
            
            self._inference_mode = True
            print(f"[ProAlign-Efficient] Inference cache ready: {V} items, mode={self.fusion_mode}")
    
    def clear_inference_cache(self):
        """清除推理缓存，恢复训练模式"""
        self._item_proto_cache = None
        self._fused_item_cache = None
        self._inference_mode = False
    # ==================== [END NEW 2025-01-17] ====================

    def embed_ID(self, x):
        """获取物品 ID embedding（兼容基类接口）"""
        return self.item_embeddings(x)  # 基类 SASRec_backbone 的 forward() 会调用

    def return_item_emb(self):
        """
        返回全量物品 embedding（兼容基类接口）
        
        用于推理时计算用户表示与所有物品的相似度
        返回的是融合后的物品表示：ID embedding + 原型语义信息
        """
        # ==================== [NEW 2025-01-17] 推理效率优化 ====================
        # 推理模式：直接返回预计算的融合嵌入，避免在线计算
        if self._inference_mode and self._fused_item_cache is not None:
            return self._fused_item_cache
        # ==================== [END NEW 2025-01-17] ====================
        
        # ==================== [NEW 2025-01-17] 原型机制消融：w/o Prototype ====================
        if self.no_prototype:
            # 消融模式：不使用原型机制，直接返回 ID embedding
            item_emb = self.item_embeddings.weight  # [V+1, D]
            if self.fusion_mode == 'add':
                return item_emb
            else:
                # concat 模式需要保持维度一致
                r_dummy = torch.zeros_like(item_emb)  # [V+1, D]
                return torch.cat([item_emb, r_dummy], dim=-1)  # [V+1, 2D]
        # ==================== [END NEW 2025-01-17] ====================
        
        if self.fusion_mode == 'add':
            return self._get_fused_item_emb_add()  # 加法融合：e_i + α * r_i
        else:
            return self._get_fused_item_emb_concat()  # 拼接融合：[e_i, r_i]

    def _get_fused_item_emb_add(self):
        """
        加法融合：e_i + α * r_i
        
        为每个物品计算：原始 ID embedding + 语义权重 × 原型加权表示
        用于推理时与用户表示计算相似度
        """
        item_emb = self.item_embeddings.weight  # [V+1, D] 获取所有物品的 ID embedding（包括 padding）

        # ==================== [OLD] 原型寻址（单头，已注释）====================
        # score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature
        # pi_all = F.softmax(score_all, dim=-1)
        # r_all = torch.matmul(pi_all, self.prototypes)
        # r_all = r_all * self.macro_scale
        # ==================== [END OLD] ====================
        
        # ==================== [NEW-MultiHead] 多头原型寻址 ====================
        V = item_emb.size(0)  # 物品数量 + 1（包括 padding）
        if self.num_heads_proto == 1:
            # 单头模式：标准原型寻址
            score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature  # [V, K] 每个物品与每个原型的相似度
            pi_all = F.softmax(score_all, dim=-1)  # [V, K] 每个物品的原型分布
            r_all = torch.matmul(pi_all, self.prototypes)  # [V, D] 每个物品的加权原型表示
        else:
            # 多头模式：将 D 维切分为 H 个子空间，每个 head 独立寻址
            item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)  # [V, H, d] 切分物品嵌入
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d] 切分原型
            scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature  # [V, H, K] 每个 head 的相似度
            pi = F.softmax(scores, dim=-1)  # [V, H, K] 每个 head 独立 softmax
            r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)  # [V, H, d] 每个 head 的加权原型
            r_all = r_heads.reshape(V, self.hidden_dim)  # [V, D] 拼接回原始维度
        r_all = r_all * self.macro_scale  # 应用可学习缩放因子
        # ==================== [END NEW-MultiHead] ====================

        return item_emb + self.semantic_weight * r_all  # [V, D] 加法融合：ID + α×语义

    def _get_fused_item_emb_concat(self):
        """
        拼接融合：[e_i, r_i]
        
        为每个物品计算：[原始 ID embedding, 原型加权表示] 拼接
        输出维度是 2D，用于推理时与用户表示计算相似度
        """
        # self.item_embeddings 是一个 nn.Embedding 层，里面有一个参数矩阵，形状是：[num_embeddings, embedding_dim]，[12102, 128]，每一行就是一个 item 的向量表示
        # .weight：拿到的就是 这整个 embedding 矩阵本身
        #
        # 把 Embedding 层里那张 完整的 item embedding 矩阵 取出来
        item_emb = self.item_embeddings.weight  # [V+1, D] (12102,128) 获取所有物品的 ID embedding（包括 padding）

        # ==================== [OLD] 原型寻址（单头，已注释）====================
        # score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature
        # pi_all = F.softmax(score_all, dim=-1)
        # r_all = torch.matmul(pi_all, self.prototypes)
        # r_all = r_all * self.macro_scale
        # ==================== [END OLD] ====================
        
        # ==================== [NEW-MultiHead] 多头原型寻址 ====================
        V = item_emb.size(0)  # 物品数量 + 1（包括 padding）12102
        if self.num_heads_proto == 1:
            # 单头模式：标准原型寻址
            score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature  # [V, K] 相似度分数  (12102,128)@(128,64)——>(12102,64)
            pi_all = F.softmax(score_all, dim=-1)  # [V, K] 原型分布 (12102,64)
            r_all = torch.matmul(pi_all, self.prototypes)  # [V, D] 加权原型表示  (12102,128)
        else:
            # 多头模式：将 D 维切分为 H 个子空间，每个 head 独立寻址
            item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)  # [V, H, d] 切分物品嵌入
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d] 切分原型
            scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature  # [V, H, K] 每个 head 的相似度
            pi = F.softmax(scores, dim=-1)  # [V, H, K] 每个 head 独立 softmax
            r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)  # [V, H, d] 每个 head 的加权原型
            r_all = r_heads.reshape(V, self.hidden_dim)  # [V, D] 拼接回原始维度
        r_all = r_all * self.macro_scale  # 应用可学习缩放因子
        # ==================== [END NEW-MultiHead] ====================

        # 拼接物品侧的微观与宏观
        #
        # 表示	    变量	            维度	        含义
        # 微观	    item_emb        [V, D]	物品的 ID embedding（可学习，编码个体特征）
        # 宏观	    r_all	        [V, D]	物品的 语义表示（在原型空间的软分配结果）
        #
        # 举例
        # 假设物品 ID=1234 是 "iPhone 15 手机壳"：
        #
        # 表示	                                    内容
        # item_emb[1234]（微观）	    该物品的唯一 ID embedding，编码"这个具体的手机壳"
        # 原型分布 π[1234]	        [手机配件: 0.6, 苹果生态: 0.3, 保护套: 0.1]
        # r_all[1234]（宏观）	    加权原型 = "手机配件类 + 苹果生态类" 的语义表示
        return torch.cat([item_emb, r_all], dim=-1)  # [V, 2D] (12102,256) 拼接融合：[ID embedding, 语义表示]

    def forward(self, sequences):
        """
        前向传播（重写基类方法）    重写后，ProAlign 完全使用自己的 forward()，不会用 SASRec_backbone 的

        Args:
            sequences: [B, S] 用户历史交互序列

        Returns:
            H_final: [B, D] (加法模式) 或 [B, 2D] (拼接模式)
        """
        # 获取 ID 嵌入
        inputs_emb = self.embed_ID(sequences) # (256,50,128)
        
        # ==================== [NEW-SLSI] 序列级语义注入 ====================
        # 核心思想：在每个序列位置注入语义信息，而非只在最后一步
        # 让语义信息参与整个注意力计算过程
        # [NEW 2025-01-17] 当 no_prototype=True 时跳过 SLSI（因为 SLSI 依赖原型机制）
        if self.use_slsi and not self.no_prototype:
            B, S, D = inputs_emb.shape
            
            # ==================== [NEW-SLSI-FIX 2024-12-16] Padding Mask ====================
            # 核心修复：防止 Padding 位置参与原型寻址
            # 问题：如果 inputs_emb 在 padding 位置是 0，softmax 会算出均匀分布 (1/K)
            #       导致 padding 位置获得"平均语义向量"，污染后续 Transformer 计算
            # 解决：显式创建 Mask，确保 Padding 位置的语义注入为 0
            slsi_mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)  # [B, S, 1]
            # ==================== [END NEW-SLSI-FIX] ====================
            
            # ==================== [OLD] 单头 SLSI（已注释）====================
            # slsi_score = torch.matmul(inputs_emb, self.prototypes.t()) / self.temperature
            # slsi_pi = F.softmax(slsi_score, dim=-1)  # [B, S, K]
            # r_seq = torch.matmul(slsi_pi, self.prototypes)  # [B, S, D]
            # ==================== [END OLD] ====================
            
            # ==================== [NEW-SLSI-ContextAware] 上下文感知 SLSI ====================
            # 选择 SLSI 模式：独立位置 vs 上下文感知
            if self.slsi_context_aware:
                # ==================== 上下文感知模式 ====================
                # 核心思想：每个位置的语义注入考虑历史上下文
                # 使用因果（causal）累积均值：position i 只看 [0, 1, ..., i]
                # 这样不会泄露未来信息
                
                # [NEW-SLSI-FIX] 计算累积和之前，先确保 padding 位置是 0
                inputs_emb_masked = inputs_emb * slsi_mask  # 双重保险
                
                # Step 1: 计算因果累积均值
                # cumsum[i] = sum(inputs_emb[0:i+1])
                # 然后除以位置数得到均值
                cumsum = torch.cumsum(inputs_emb_masked, dim=1)  # [B, S, D] 累积和
                # ==================== [OLD] 固定位置数除法（会被 padding 稀释）====================
                # positions = torch.arange(1, S + 1, device=inputs_emb.device).float().view(1, S, 1)  # [1, S, 1]
                # context_repr = cumsum / positions  # [B, S, D] 因果累积均值
                # ==================== [END OLD] ====================
                
                # ==================== [NEW 2024-12-17] 按累计有效 token 数除 ====================
                # 修复：如果序列后半段是 padding，应该按累计有效 token 数除，而不是固定位置数
                # 这样 padding 位置不会"稀释"表示
                counts = torch.cumsum(slsi_mask, dim=1).clamp_min(1.0)  # [B, S, 1] 累计有效 token 数
                context_repr = cumsum / counts  # [B, S, D] 因果累积均值（按有效长度）
                # ==================== [END NEW] ====================
                # context_repr[i] = mean(inputs_emb[0:i+1])，只包含历史和当前，不包含未来
                
                # Step 2: 用上下文表示做原型寻址
                if self.num_heads_proto == 1:
                    # 单头模式
                    slsi_score = torch.matmul(context_repr, self.prototypes.t()) / self.temperature  # [B, S, K]
                    slsi_pi = F.softmax(slsi_score, dim=-1)  # [B, S, K]
                    r_seq = torch.matmul(slsi_pi, self.prototypes)  # [B, S, D]
                else:
                    # 多头模式
                    context_heads = context_repr.view(B, S, self.num_heads_proto, self.head_dim)  # [B, S, H, d]
                    proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d]
                    slsi_scores = torch.einsum('bshd,khd->bshk', context_heads, proto_heads) / self.temperature  # [B, S, H, K]
                    slsi_pi = F.softmax(slsi_scores, dim=-1)  # [B, S, H, K]
                    r_seq_heads = torch.einsum('bshk,khd->bshd', slsi_pi, proto_heads)  # [B, S, H, d]
                    r_seq = r_seq_heads.reshape(B, S, self.hidden_dim)  # [B, S, D]
            else:
                # ==================== 独立位置模式（原始 SLSI）====================
                # 每个位置独立做原型寻址，不考虑上下文
                
                # ==================== [NEW 2025-01-17] 推理效率优化 ====================
                # 推理模式：直接从预计算缓存中查表，O(1) 时间复杂度
                if self._inference_mode and self._item_proto_cache is not None:
                    # 查表获取序列中每个物品的原型表示
                    r_seq = self._item_proto_cache[sequences]  # [B, S, D]
                # ==================== [END NEW 2025-01-17] ====================
                elif self.num_heads_proto == 1:
                    # 单头模式（向后兼容）
                    # 计算每个位置的原型相似度
                    #
                    # inputs_emb: [B, S, D]，每个位置的 ID embedding
                    # self.prototypes: [K, D]，K 个原型，每个 D 维
                    # 给序列中每个位置，算一遍它对所有 K 个原型的相似度
                    slsi_score = torch.matmul(inputs_emb, self.prototypes.t()) / self.temperature  # [B,S,D]  @ [D,K] ——> [B, S, K]  (256,50,64)
                    # Softmax 归一化  给序列中每个位置，算一遍它对所有 K 个原型的相似度
                    slsi_pi = F.softmax(slsi_score, dim=-1)  # [B, S, K]  (256,50,64)
                    # 加权原型，得到每个位置的语义表示
                    r_seq = torch.matmul(slsi_pi, self.prototypes)  # [B, S, D]   (256,50,128)
                else:
                    # 多头 SLSI
                    inputs_heads = inputs_emb.view(B, S, self.num_heads_proto, self.head_dim)  # [B, S, H, d]
                    proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d]
                    # [B, S, H, d] @ [K, H, d] -> [B, S, H, K]
                    slsi_scores = torch.einsum('bshd,khd->bshk', inputs_heads, proto_heads) / self.temperature
                    slsi_pi = F.softmax(slsi_scores, dim=-1)  # [B, S, H, K]
                    # [B, S, H, K] @ [K, H, d] -> [B, S, H, d]
                    r_seq_heads = torch.einsum('bshk,khd->bshd', slsi_pi, proto_heads)
                    r_seq = r_seq_heads.reshape(B, S, self.hidden_dim)  # [B, S, D]
            # ==================== [END NEW-SLSI-ContextAware] ====================
            
            # ==================== [NEW-SLSI-FIX 2024-12-16] 关键修复 ====================
            # 注入语义后，立即 Mask，确保 Padding 位置的语义注入为 0
            # 这样后面的 Transformer 即使有 LayerNorm 也不会受噪声影响
            r_seq = r_seq * slsi_mask
            # ==================== [END NEW-SLSI-FIX] ====================
            
            # 语义增强：原始嵌入 + 加权语义表示
            inputs_emb = inputs_emb + self.slsi_weight * r_seq # (256,50,128)
        # ==================== [END NEW-SLSI] ====================
        
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device)) # 添加位置编码：[B,S,D] + [S,D] → [B,S,D]，让模型知道每个物品在序列中的位置
        seq = self.emb_dropout(inputs_emb)  # Dropout 正则化：随机丢弃部分神经元，防止过拟合
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)  # 创建掩码：标记非 padding 位置为 1，padding 位置为 0，shape: [B,S,1]   (256,50,1)
        seq *= mask  # 应用掩码：将 padding 位置的嵌入置零
        seq_normalized = self.ln_1(seq)  # Layer Normalization：标准化输入，加速训练收敛
        mh_attn_out = self.mh_attn(seq_normalized, seq)  # 多头自注意力：捕获序列中物品之间的依赖关系，输出 [B,S,D]
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))  # 前馈网络：两层全连接 + ReLU，增加模型表达能力
        ff_out *= mask  # 再次应用掩码：确保 padding 位置不产生输出
        ff_out = self.ln_3(ff_out)  # 最后一层 Layer Normalization   (256,50,128)
        h_u = ff_out[:, -1, :]  # [B, D] (256,128)  微观状态：取序列最后一个时间步的输出作为用户表示

        # ==================== [OLD] 原型寻址（单头，已注释）====================
        # score_stu = torch.matmul(h_u, self.prototypes.t()) / self.temperature
        # pi_stu = F.softmax(score_stu, dim=-1)  # [B, K]
        # r_u = torch.matmul(pi_stu, self.prototypes)  # [B, D] 宏观意图
        # r_u = r_u * self.macro_scale
        # ==================== [END OLD] ====================
        
        # ==================== [NEW 2025-01-17] 原型机制消融：w/o Prototype ====================
        # 当 no_prototype=True 时，跳过原型寻址，直接返回 h_u
        if self.no_prototype:
            # 消融模式：不使用原型机制，直接返回 h_u
            if self.fusion_mode == 'add':
                H_final = h_u  # [B, D]
            else:
                # concat 模式需要保持维度一致，用零向量填充 r_u 的位置
                r_u_dummy = torch.zeros_like(h_u)  # [B, D]
                H_final = torch.cat([h_u, r_u_dummy], dim=-1)  # [B, 2D]
            return H_final
        # ==================== [END NEW 2025-01-17] ====================
        
        # ==================== [NEW-MultiHead] 多头解耦原型寻址 ====================
        # 核心思想：将 D 维切分为 H 个子空间，每个 head 独立寻址原型
        # 这样不同意图方向不会被"平均化"抵消
        B = h_u.size(0)
        
        # ==================== [NEW 2024-12-31] 动态注意力融合 ====================
        # 与 ProAlign_BERT4Rec 保持一致
        if self.use_attn_fusion and hasattr(self, 'proto_attn'):
            # 用户状态作为 Query，在原型库中动态检索最匹配的意图
            query = h_u.unsqueeze(1)  # [B, 1, D]
            keys = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

            # Attention: r_dynamic = softmax(Q·K^T / sqrt(d)) · V
            r_dynamic, _ = self.proto_attn(query, keys, keys)
            r_u = r_dynamic.squeeze(1)  # [B, D]
        # ==================== [END NEW 2024-12-31] ====================
        elif self.num_heads_proto == 1:
            # 单头模式（向后兼容）
            score_stu = torch.matmul(h_u, self.prototypes.t()) / self.temperature # (256,64)@(64,128)——>(256,64)
            pi_stu = F.softmax(score_stu, dim=-1)  # [B, K]  (256,64)
            r_u = torch.matmul(pi_stu, self.prototypes)  # [B, D]  (256,128)
        else:
            # 多头模式
            # 1. 将 h_u 和 prototypes 切分为 H 个子空间
            h_u_heads = h_u.view(B, self.num_heads_proto, self.head_dim)  # [B, H, d]
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d]
            
            # 2. 每个 head 独立计算注意力分布
            # einsum: 'bhd,khd->bhk' 表示每个 head 内做点积
            scores = torch.einsum('bhd,khd->bhk', h_u_heads, proto_heads) / self.temperature  # [B, H, K]
            pi = F.softmax(scores, dim=-1)  # [B, H, K] 每个 head 独立 softmax
            
            # 3. 每个 head 独立加权原型
            # einsum: 'bhk,khd->bhd' 表示每个 head 用自己的 pi 加权原型
            r_u_heads = torch.einsum('bhk,khd->bhd', pi, proto_heads)  # [B, H, d]
            
            # 4. 拼接回 [B, D]
            r_u = r_u_heads.reshape(B, self.hidden_dim)  # [B, D]
        
        r_u = r_u * self.macro_scale # (256,128)
        # ==================== [END NEW-MultiHead] ====================

        # ==================== 融合：将微观状态 h_u 和宏观意图 r_u 组合 ====================
        if self.fusion_mode == 'add':  # 加法融合模式
            H_final = h_u + self.semantic_weight * r_u  # [B, D]，直接加权相加：用户表示 = 微观 + α×宏观
        else:  # concat 拼接融合模式
            concat_feat = torch.cat([h_u, r_u], dim=-1)  # [B, 2D] (256,256)，先拼接两个表示
            g = self.gate(concat_feat)  # [B, 1] (256,1)，门控网络计算自适应权重 g ∈ (0,1)
            # 举例说明
            # 假设用户历史：[iPhone手机壳, AirPods, MacBook保护套, iPad触控笔]
            #
            # 表示	        内容
            # h_u（微观）	编码了"用户买了这 4 个具体物品"
            # 原型分布 π	    [电子配件: 0.7, 数码产品: 0.2, 其他: 0.1]
            # r_u（宏观）	0.7×电子配件原型 + 0.2×数码原型 + ... = "苹果生态配件爱好者"
            H_final = torch.cat([h_u, g * r_u], dim=-1)  # [B, 2D] (256,256)，用门控值加权宏观意图后拼接

        return H_final  # 返回最终用户表示：加法模式 [B,D]，拼接模式 [B,2D]

    def predict(self, sequences):
        """推理预测（兼容基类接口）：给定用户序列，返回所有物品的预测分数"""
        H_final = self.forward(sequences)  # 前向传播：获取用户最终表示 [B, D] 或 [B, 2D]  (256,256)
        item_embs = self.return_item_emb()  # 获取所有物品的融合嵌入 [V+1, D] 或 [V+1, 2D] (12102,256)

        # 去掉 padding embedding（最后一行是 padding 的嵌入向量）
        if self.fusion_mode == 'add':
            item_embs = item_embs[:-1]  # [V, D]，去掉第 V+1 行（padding）
        else:
            item_embs = item_embs[:-1]  # [V, 2D]  (12101,256)，去掉第 V+1 行（padding）

        scores = torch.matmul(H_final, item_embs.t())  # [B, V]，(256,12101)  计算用户与所有物品的相似度分数
        return scores  # 返回预测分数矩阵，用于排序推荐

    def calculate_loss_with_align(self, sequences, target, user_ids, neg_ratio, temperature):
        """
        计算带对齐损失的总损失（ProAlign 专用）
        
        Args:
            sequences: [B, S] 用户历史交互序列
            target: [B] 目标物品 ID（正样本）
            user_ids: [B] 用户 ID（用于获取用户意图 embedding）
            neg_ratio: int 负采样比例（每个正样本采多少负样本）
            temperature: float InfoNCE 温度参数

        Returns:
            loss: L = L_rec + α * L_align + β * L_cluster
        """
        H_final = self.forward(sequences)  # 前向传播：获取用户最终表示 [B, D] 或 [B, 2D] (256,256)

        # ==================== L_rec: 主推荐损失（InfoNCE）====================
        item_embs = self.return_item_emb()  # 获取所有物品的融合嵌入 (12102,256)

        # 正样本：获取目标物品的融合嵌入
        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)  # [B, D]
        else:
            pos_embs = self._get_target_fused_emb_concat(target)  # [B, 2D]   (256,)——>(256,256)  正样本嵌入

        # 负采样：为每个样本随机采样 neg_ratio 个负样本
        batch_size = target.shape[0]  # 批大小 B
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))  # [B, neg_ratio] (256,64) 随机采样
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()  # [B, neg_ratio] (256,64) 扩展正样本
        mask = neg_samples == expanded_target  # 检查负样本是否与正样本重复
        while mask.any():  # 如果有重复，重新采样
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))  # 重新采样
            neg_samples = torch.where(mask, new_samples, neg_samples)  # 只替换重复的位置
            mask = neg_samples == expanded_target  # 重新检查
        neg_samples = neg_samples.to(target.device)  # 移到正确的设备

        # 获取负样本的融合嵌入
        if self.fusion_mode == 'add':
            neg_embs = self._get_target_fused_emb_add(neg_samples)  # [B, neg_ratio, D]
        else:
            neg_embs = self._get_target_fused_emb_concat(neg_samples)  # [B, neg_ratio, 2D] (256,64,256)

        # L2 归一化：将向量投影到单位超球面，使点积等价于余弦相似度
        H_final_norm = F.normalize(H_final, p=2, dim=-1)  # [B, D] 或 [B, 2D]   (256,256)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)  # [B, D] 或 [B, 2D]  (256,256)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)  # [B, neg_ratio, D] 或 [B, neg_ratio, 2D]  (256,64,256)

        # InfoNCE 损失计算
        pos_logits = (H_final_norm * pos_embs_norm).sum(dim=-1, keepdim=True)  # [B, 1] 正样本相似度
        neg_logits = torch.bmm(neg_embs_norm, H_final_norm.unsqueeze(-1)).squeeze(-1)  # [B, neg_ratio] 负样本相似度
        logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature  # [B, 1+neg_ratio] (256,65) 拼接并除以温度
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # [B] (256,)标签全为 0（正样本在第 0 位）
        L_rec = F.cross_entropy(logits, labels)  # 交叉熵损失

        # ==================== L_align: 意图对齐损失 ====================
        # 通过 --align_mode 参数选择对齐方式：
        #   'kl'     : 原型分布对齐（KL 散度）- 让 h_u 和 z_next 在原型空间的分布一致
        #   'infonce': 跨视图对比学习（InfoNCE）- 直接拉近 h_u 和 z_next 的向量表示
        L_align = torch.tensor(0.0, device=sequences.device)  # 初始化对齐损失为 0
        align_mode = self.key_words.get('align_mode', 'infonce')  # 获取对齐模式，默认 InfoNCE
        cl_temperature = self.key_words.get('cl_temperature', 0.04)  # 对比学习温度参数

        # 只有当用户意图 embedding 存在且有 user_ids 时才计算对齐损失
        if self.user_intent_emb is not None and user_ids is not None:
            z_next = self.user_intent_emb[user_ids.cpu()].to(sequences.device)  # [256, 3072] 获取用户未来意图 embedding
            z_next_proj = self.adapter(z_next)  # [256, 128] 通过 Adapter 降维：3072 → 128

            # 提取 h_u（微观状态）：从 H_final 中取前 hidden_dim 维
            h_u = H_final[:, :self.hidden_dim]  # [256, 128]

            # L2 归一化：用于计算余弦相似度
            h_u_norm = F.normalize(h_u, p=2, dim=-1)  # [256, 128] 归一化后模长=1
            z_next_norm = F.normalize(z_next_proj, p=2, dim=-1)  # [256, 128] 归一化后模长=1

            if align_mode == 'kl':
                # ==================== 方案 A：原型分布对齐（KL 散度） ====================
                # 思路：让 h_u 和 z_next 在原型空间的分布一致
                proto_norm = F.normalize(self.prototypes, p=2, dim=-1)  # [K, D] 归一化原型

                # 学生分布（h_u 在原型空间的软分配）
                score_stu = torch.matmul(h_u_norm, proto_norm.t()) / self.temperature  # [B, K] 学生分数

                # 教师分布（z_next 在原型空间的软分配）
                score_tea = torch.matmul(z_next_norm, proto_norm.t()) / self.temperature  # [B, K] 教师分数
                pi_tea = F.softmax(score_tea, dim=-1)  # [B, K] 教师软分布（目标分布）

                # KL 散度：度量两个分布的差异，让学生分布逼近教师分布
                log_pi_stu = F.log_softmax(score_stu, dim=-1)  # [B, K] 学生 log 概率
                L_align = F.kl_div(log_pi_stu, pi_tea, reduction='batchmean')  # KL(学生 || 教师)

            elif align_mode == 'infonce':
                # # ==================== [NEW] Forward Prediction Contrastive Learning ====================
                # # 核心：让模型"学会预测未来意图"，而不是简单融合
                # #
                # # h_u:     ID-based user representation (what user has done)
                # # h_pred:  Predicted future intent (what model thinks user will want)
                # # z_next:  LLM-based future intent (ground truth from LLM)
                # #
                # # h_u → Predictor → h_pred, then InfoNCE(h_pred, z_next)
                #
                # if self.use_forward_predictor:
                #     # [NEW] Forward Prediction: h_u → Predictor → h_pred
                #     h_pred = self.forward_predictor(h_u)  # [B, D]
                #     h_pred_norm = F.normalize(h_pred, p=2, dim=-1)
                #     query_norm = h_pred_norm
                # else:
                #     # [OLD] Direct alignment (用于消融实验对比)
                #     query_norm = h_u_norm
                #
                # batch_size_align = query_norm.size(0)
                #
                # # 跨视图相似度矩阵 [B, B]
                # #         z₀    z₁    z₂
                # # query₀ [  ★     ·     ·  ]   label=0
                # # query₁ [  ·     ★     ·  ]   label=1
                # # query₂ [  ·     ·     ★  ]   label=2
                # # ★ = 正样本（对角线），· = 负样本
                # sim_matrix = torch.matmul(query_norm, z_next_norm.t()) / cl_temperature
                #
                # # 正样本在对角线上
                # labels_align = torch.arange(batch_size_align, device=sequences.device)
                #
                # # 双向对比
                # loss_h2z = F.cross_entropy(sim_matrix, labels_align)      # query → z_next
                # loss_z2h = F.cross_entropy(sim_matrix.t(), labels_align)  # z_next → query
                #
                # L_align = (loss_h2z + loss_z2h) / 2

                # ==================== [OLD] 以下为旧版直接对齐代码（已注释保留）====================
                # 方案 B：跨视图 InfoNCE
                # 思路：直接拉近 h_u 和 z_next 的向量表示
                # 正样本：(h_u[i], z_next[i]) - 同一用户的两种表示
                # 负样本：batch 内其他用户
                #
                # sim_matrix [B, B]:
                #         z₀    z₁    z₂
                # h₀  [  ★     ·     ·  ]   label=0
                # h₁  [  ·     ★     ·  ]   label=1
                # h₂  [  ·     ·     ★  ]   label=2
                # ★ = 正样本（对角线），· = 负样本

                batch_size_align = h_u_norm.size(0)  # 批大小  256

                # 跨视图相似度矩阵 [B, B]：计算所有 h_u 和 z_next 之间的相似度
                sim_matrix = torch.matmul(h_u_norm, z_next_norm.t()) / cl_temperature  # [256, 256] 相似度矩阵

                # 相似度矩阵 sim_matrix 的示意：
                # 假设 batch_size = 4，h_u 和 z_next 的顺序一一对应：
                #
                #   sim_matrix =
                #   [
                #     [(h_0, z_0), (h_0, z_1), (h_0, z_2), (h_0, z_3)],
                #     [(h_1, z_0), (h_1, z_1), (h_1, z_2), (h_1, z_3)],
                #     [(h_2, z_0), (h_2, z_1), (h_2, z_2), (h_2, z_3)],
                #     [(h_3, z_0), (h_3, z_1), (h_3, z_2), (h_3, z_3)],
                #   ]
                #
                # - 第 0 行：h_0 和所有 z 的相似度 -> 正样本是 (h_0, z_0) -> 在第 0 列
                # - 第 1 行：h_1 和所有 z 的相似度 -> 正样本是 (h_1, z_1) -> 在第 1 列
                # - 第 2 行：h_2 和所有 z 的相似度 -> 正样本是 (h_2, z_2) -> 在第 2 列
                # - 第 3 行：h_3 和所有 z 的相似度 -> 正样本是 (h_3, z_3) -> 在第 3 列
                #
                # 所以：正样本对 (h_i, z_i) 正好就是 sim_matrix 的对角线元素


                # 正样本在对角线上：第 i 个用户的 h_u[i] 应该与 z_next[i] 最相似
                labels_align = torch.arange(batch_size_align, device=sequences.device)  # [0, 1, 2, ..., B-1]

                # 双向对比损失：
                loss_h2z = F.cross_entropy(sim_matrix, labels_align)  # h_u → z_next：给定 h_u[i]，找对应的 z_next[i]
                loss_z2h = F.cross_entropy(sim_matrix.t(), labels_align)  # z_next → h_u：给定 z_next[i]，找对应的 h_u[i]

                L_align = (loss_h2z + loss_z2h) / 2  # 取平均作为最终对齐损失

        # ==================== L_cluster: 聚类正则化损失 ====================
        # 目的：让目标物品的 ID embedding 靠近其对应的原型表示
        # 这促使 ID embedding 学习到与原型一致的语义结构

        # 为什么加上这个损失函数
        # 问题：训练过程中 ID Embedding 会"失去语义"
        # 初始化时（semantic_init=True）：
        #   - 护肤品A 的 embedding 靠近 护肤品B
        #   - 手机壳 的 embedding 靠近 手机膜
        #   ✅ 同类物品的 embedding 相似（因为来自 LLM）
        #
        # 训练 500 epochs 后：
        #   - 只有 L_rec 损失（推荐损失）
        #   - 优化目标：让用户喜欢的物品得分高
        #   - 结果：embedding 为了"拟合训练数据"而移动
        #   ❌ 可能导致护肤品A 和 护肤品B 的 embedding 不再相似！
        #
        #
        # L_cluster 的作用：防止语义结构被破坏
        # L_cluster = MSE(e_target, r_target)
        #
        # e_target = 物品当前的 embedding（可能已经偏离）
        # r_target = 物品"应该在"的位置（根据 LLM 语义）
        # L_cluster = 惩罚两者的距离
        # 就像弹簧一样，把 embedding 拉回到"语义正确"的位置
        #
        #
        # 类比
        # 想象你在训练一个推荐系统：
        #
        # 没有 L_cluster	有 L_cluster
        # 只关心"推荐准不准"	        同时关心"推荐准"和"语义对"
        # 苹果和橘子可能被训练到很远	苹果和橘子保持在"水果"附近
        # 失去泛化能力	            保持泛化能力
        
        # ==================== [NEW 2025-01-17] 原型机制消融：w/o Prototype ====================
        # 当 no_prototype=True 时禁用 L_cluster（因为 L_cluster 依赖原型机制）
        if self.no_prototype:
            L_cluster = torch.tensor(0.0, device=sequences.device)
        else:
            e_target = self.item_embeddings(target)  # [B, D] (256,128) 获取目标物品的 ID embedding
            e_target_norm = F.normalize(e_target, p=2, dim=-1)  # [B, D]  (256,128)  L2 归一化
            proto_norm = F.normalize(self.prototypes, p=2, dim=-1)  # [K, D] 归一化原型矩阵
            
            # ==================== [OLD] 单头原型寻址（已注释）====================
            # score_target = torch.matmul(e_target_norm, proto_norm.t()) / self.temperature
            # pi_target = F.softmax(score_target, dim=-1)
            # r_target = torch.matmul(pi_target, self.prototypes)
            # r_target = r_target * self.macro_scale
            # ==================== [END OLD] ====================
            
            # ==================== [NEW-MultiHead] 多头原型寻址 ====================
            B_proto = e_target.size(0)  # 批大小
            if self.num_heads_proto == 1:
                # 单头模式：标准原型寻址
                score_target = torch.matmul(e_target_norm, proto_norm.t()) / self.temperature  # [B, K] (256,64)相似度分数
                pi_target = F.softmax(score_target, dim=-1)  # [B, K] (256,64)原型分布（软分配）
                r_target = torch.matmul(pi_target, self.prototypes)  # [B, D] (256,128)加权原型表示
            else:
                # 多头模式：将 D 维切分为 H 个子空间，每个 head 独立寻址
                e_target_heads = e_target_norm.view(B_proto, self.num_heads_proto, self.head_dim)  # [B, H, d]
                proto_heads = proto_norm.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d]
                scores = torch.einsum('bhd,khd->bhk', e_target_heads, proto_heads) / self.temperature  # [B, H, K]
                pi_target = F.softmax(scores, dim=-1)  # [B, H, K] 每个 head 独立 softmax
                proto_unnorm = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)  # [K, H, d]
                r_heads = torch.einsum('bhk,khd->bhd', pi_target, proto_unnorm)  # [B, H, d] 每个 head 的加权原型
                r_target = r_heads.reshape(B_proto, self.hidden_dim)  # [B, D] 拼接回原始维度
            r_target = r_target * self.macro_scale  # 应用可学习缩放因子 (256,128)
            # ==================== [END NEW-MultiHead] ====================
            
            # MSE 损失：让 ID embedding 逼近对应的原型表示
            # detach() 阻止梯度回传到原型，只更新 ID embedding
            L_cluster = F.mse_loss(e_target, r_target.detach())
        # ==================== [END NEW 2025-01-17] ====================

        # ==================== 总损失 ====================
        # L = L_rec + α * L_align + β * L_cluster
        # L_rec: 主推荐任务损失（InfoNCE）
        # L_align: 用户意图对齐损失（让 ID 表示与 LLM 意图对齐）
        # L_cluster: 聚类正则化损失（让 ID embedding 保持语义结构）
        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster

        return loss  # 返回总损失用于反向传播

    def _get_target_fused_emb_add(self, target):
        """获取目标物品的加法融合 embedding"""
        e_i = self.item_embeddings(target)
        
        # ==================== [NEW 2025-01-17] 原型机制消融：w/o Prototype ====================
        if self.no_prototype:
            return e_i  # 消融模式：直接返回纯 ID embedding
        # ==================== [END NEW 2025-01-17] ====================
        
        # ==================== [OLD] 单头原型寻址（已注释）====================
        # score = torch.matmul(e_i, self.prototypes.t()) / self.temperature
        # pi = F.softmax(score, dim=-1)
        # r_i = torch.matmul(pi, self.prototypes)
        # r_i = r_i * self.macro_scale
        # ==================== [END OLD] ====================
        
        # ==================== [NEW-MultiHead] 多头原型寻址 ====================
        if self.num_heads_proto == 1:
            score = torch.matmul(e_i, self.prototypes.t()) / self.temperature
            pi = F.softmax(score, dim=-1)
            r_i = torch.matmul(pi, self.prototypes)
        else:
            # target 可能是 [B] 或 [B, N]，需要处理不同维度
            orig_shape = e_i.shape[:-1]  # 保存原始形状（除了最后一维）
            e_i_flat = e_i.view(-1, self.hidden_dim)  # 展平为 [*, D]
            N = e_i_flat.size(0)
            
            e_i_heads = e_i_flat.view(N, self.num_heads_proto, self.head_dim)  # [*, H, d]
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('nhd,khd->nhk', e_i_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('nhk,khd->nhd', pi, proto_heads)
            r_i_flat = r_heads.reshape(N, self.hidden_dim)
            r_i = r_i_flat.view(*orig_shape, self.hidden_dim)  # 恢复原始形状
        r_i = r_i * self.macro_scale
        # ==================== [END NEW-MultiHead] ====================
        
        return e_i + self.semantic_weight * r_i

    def _get_target_fused_emb_concat(self, target):
        """获取目标物品的拼接融合 embedding"""
        e_i = self.item_embeddings(target) # (256,128)
        
        # ==================== [NEW 2025-01-17] 原型机制消融：w/o Prototype ====================
        if self.no_prototype:
            r_i_dummy = torch.zeros_like(e_i)  # 消融模式：用零向量填充
            return torch.cat([e_i, r_i_dummy], dim=-1)
        # ==================== [END NEW 2025-01-17] ====================
        
        # ==================== [OLD] 单头原型寻址（已注释）====================
        # score = torch.matmul(e_i, self.prototypes.t()) / self.temperature
        # pi = F.softmax(score, dim=-1)
        # r_i = torch.matmul(pi, self.prototypes)
        # r_i = r_i * self.macro_scale
        # ==================== [END OLD] ====================
        
        # ==================== [NEW-MultiHead] 多头原型寻址 ====================
        if self.num_heads_proto == 1:
            score = torch.matmul(e_i, self.prototypes.t()) / self.temperature # (256,64)
            pi = F.softmax(score, dim=-1) # (256,64)
            r_i = torch.matmul(pi, self.prototypes) # (256,128)
        else:
            # target 可能是 [B] 或 [B, N]，需要处理不同维度
            orig_shape = e_i.shape[:-1]
            e_i_flat = e_i.view(-1, self.hidden_dim)
            N = e_i_flat.size(0)
            
            e_i_heads = e_i_flat.view(N, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('nhd,khd->nhk', e_i_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('nhk,khd->nhd', pi, proto_heads)
            r_i_flat = r_heads.reshape(N, self.hidden_dim)
            r_i = r_i_flat.view(*orig_shape, self.hidden_dim)
        r_i = r_i * self.macro_scale
        # ==================== [END NEW-MultiHead] ====================
        
        return torch.cat([e_i, r_i], dim=-1) # (256,256)

    # ==================== 辅助方法：计算对齐损失和聚类损失 ====================
    def _compute_align_cluster_loss(self, sequences, target, h_u):
        """
        计算 L_align 和 L_cluster（供三种损失函数共用）

        Args:
            sequences: [B, S] 输入序列
            target: [B] 目标物品 ID
            h_u: [B, D] SASRec 编码的用户表示

        Returns:
            L_align: 对齐损失
            L_cluster: 聚类损失
        """
        # ==================== L_align: 意图对齐损失 ====================
        L_align = torch.tensor(0.0, device=sequences.device)
        if self.user_intent_emb is not None:
            # 注意：当前数据集没有 user_id，暂时跳过
            # 如果需要启用，需要在数据集中添加 user_id
            pass

        # ==================== L_cluster: 聚类损失 ====================
        e_target = self.item_embeddings(target)
        e_target_norm = F.normalize(e_target, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # [FIX 2024-12-16] 添加多头原型支持（与 GRU/BERT4Rec 保持一致）
        B = e_target.size(0)
        if self.num_heads_proto == 1:
            # 单头模式（原逻辑）
            score_target = torch.matmul(e_target_norm, proto_norm.t()) / self.temperature
            pi_target = F.softmax(score_target, dim=-1)
            r_target = torch.matmul(pi_target, self.prototypes)
        else:
            # 多头模式
            e_target_heads = e_target_norm.view(B, self.num_heads_proto, self.head_dim)
            proto_heads = proto_norm.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('bhd,khd->bhk', e_target_heads, proto_heads) / self.temperature
            pi_target = F.softmax(scores, dim=-1)
            proto_unnorm = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            r_heads = torch.einsum('bhk,khd->bhd', pi_target, proto_unnorm)
            r_target = r_heads.reshape(B, self.hidden_dim)
        r_target = r_target * self.macro_scale
        L_cluster = F.mse_loss(e_target, r_target.detach())

        return L_align, L_cluster

    # ==================== 兼容基类的损失函数接口 ====================
    def calculate_ce_loss(self, sequences, target):
        """
        Cross-Entropy 损失（全量 softmax）

        L = L_rec + α * L_align + β * L_cluster
        """
        H_final = self.forward(sequences)

        # L_rec: 全量物品 softmax
        item_embs = self.return_item_emb()[:-1]  # 去掉 padding
        logits = torch.matmul(H_final, item_embs.t())
        L_rec = self.ce_loss(logits, target)

        # 提取 h_u（用于对齐损失）
        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        # L_align + L_cluster
        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        # 总损失
        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    def calculate_bce_loss(self, sequences, target, neg_ratio):
        """
        Binary Cross-Entropy 损失（负采样二分类）

        L = L_rec + α * L_align + β * L_cluster
        """
        H_final = self.forward(sequences)

        # ==================== 负采样 ====================
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        neg_samples = neg_samples.to(target.device)

        # ==================== 获取融合后的嵌入 ====================
        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)
            neg_embs = self._get_target_fused_emb_add(neg_samples)
        else:
            pos_embs = self._get_target_fused_emb_concat(target)
            neg_embs = self._get_target_fused_emb_concat(neg_samples)

        # ==================== BCE 损失 ====================
        pos_logits = (H_final * pos_embs).sum(dim=-1)
        neg_logits = (H_final.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.device)
        neg_labels = torch.zeros(neg_logits.shape, device=self.device)

        L_rec = self.bce_loss(pos_logits, pos_labels) + self.bce_loss(neg_logits, neg_labels)

        # 提取 h_u
        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        # L_align + L_cluster
        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        # 总损失
        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature):
        """
        InfoNCE 对比学习损失

        L = L_rec + α * L_align + β * L_cluster
        """
        H_final = self.forward(sequences)

        # ==================== 负采样 ====================
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        neg_samples = neg_samples.to(target.device)

        # ==================== 获取融合后的嵌入 ====================
        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)
            neg_embs = self._get_target_fused_emb_add(neg_samples)
        else:
            pos_embs = self._get_target_fused_emb_concat(target)
            neg_embs = self._get_target_fused_emb_concat(neg_samples)

        # ==================== InfoNCE 损失 ====================
        # L2 归一化
        H_final_norm = F.normalize(H_final, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)

        # 计算相似度
        pos_logits = (H_final_norm * pos_embs_norm).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs_norm, H_final_norm.unsqueeze(-1)).squeeze(-1)

        # 拼接并除以温度
        logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        L_rec = F.cross_entropy(logits, labels)

        # 提取 h_u
        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        # L_align + L_cluster
        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        # 总损失
        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    # ==================== [NEW] RASD (Retrieval Augmented Self-Distillation) 损失 ====================
    def calculate_rasd_loss(self, sequences, sim_seqs, user_sim_func='cl'):
        """
        计算 RASD 对齐损失（适配 ProAlign 的版本）
        
        思路：用相似用户的表示作为"教师"，让当前用户的表示向教师靠拢
        
        Args:
            sequences: [B, S] 当前用户的物品序列
            sim_seqs: [B, K, S] 相似用户的物品序列（K 个相似用户）
            user_sim_func: 'cl' (对比学习) 或 'kd' (知识蒸馏/MSE)
        
        Returns:
            rasd_loss: 标量损失值
        """
        B, K, S = sim_seqs.shape
        
        # 1. 获取当前用户的表示
        h_u = self.forward(sequences)  # [B, D] 或 [B, 2D]
        
        # 2. 获取相似用户的表示
        sim_seqs_flat = sim_seqs.view(B * K, S)  # [B*K, S]
        h_sim = self.forward(sim_seqs_flat)  # [B*K, D] 或 [B*K, 2D]
        
        # 3. 关键：stop gradient，相似用户作为"教师"不更新梯度
        h_sim = h_sim.detach()
        
        # 4. 重塑并取平均
        h_sim = h_sim.view(B, K, -1)  # [B, K, D] 或 [B, K, 2D]
        h_sim_avg = h_sim.mean(dim=1)  # [B, D] 或 [B, 2D] 多个相似用户的平均表示
        
        # 5. 计算对齐损失
        if user_sim_func == 'cl':
            # 对比学习损失：1 - cosine_similarity
            h_u_norm = F.normalize(h_u, p=2, dim=-1)
            h_sim_norm = F.normalize(h_sim_avg, p=2, dim=-1)
            rasd_loss = 1.0 - (h_u_norm * h_sim_norm).sum(dim=-1).mean()
        elif user_sim_func == 'kd':
            # 知识蒸馏损失 (MSE)
            rasd_loss = F.mse_loss(h_u, h_sim_avg)
        else:
            raise ValueError(f"Unknown user_sim_func: {user_sim_func}")
        
        return rasd_loss
    # ==================== [END NEW] ====================


# ==================== [END NEW] ProAlign 类 ====================


# ==================== [NEW] PPD 调度器 ====================
# 移植自 BSARec proalign_sasrec.py
# 渐进式原型蒸馏（Progressive Prototype Distillation）
# ==================================================================================

class PPDScheduler:
    """
    Progressive Prototype Distillation 调度器

    核心思想：
    - 训练早期：保持 LLM 语义先验（冻结原型）
    - 训练中期：吸收协同过滤信号（解冻原型）
    - 训练后期：EMA 稳定收敛（防止波动）

    Args:
        model: ProAlign 模型实例
        total_epochs: 总训练 epoch 数
        warmup_ratio: Phase 1 占比，默认 0.3 (30%)
        transition_ratio: Phase 2 结束点占比，默认 0.7 (70%)
        ema_decay: EMA 衰减系数，默认 0.99
        verbose: 是否打印状态变化
    """

    def __init__(self, model, total_epochs, warmup_ratio=0.3, transition_ratio=0.7,
                 ema_decay=0.99, verbose=True):
        self.model = model
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.transition_epochs = int(total_epochs * transition_ratio)
        self.ema_decay = ema_decay
        self.verbose = verbose

        # EMA 影子副本（用于 Phase 3）
        self.prototype_shadow = None

        # 记录当前 phase（避免重复打印）
        self.current_phase = None

    def step(self, epoch):
        """
        每个 epoch 开始时调用，更新原型状态

        Args:
            epoch: 当前 epoch 编号 (0-indexed)
        """
        if epoch < self.warmup_epochs:
            # ====== Phase 1: 冻结 (Warmup) ======
            # 保持 LLM 语义结构不变，让 ID Embedding 先学习
            self._set_phase(1, epoch)
            self.model.prototypes.requires_grad = False

        elif epoch < self.transition_epochs:
            # ====== Phase 2: 解冻 (Transition) ======
            # 开始吸收协同过滤信号，原型可训练
            self._set_phase(2, epoch)
            self.model.prototypes.requires_grad = True

        else:
            # ====== Phase 3: EMA 稳定 (Refinement) ======
            # 可训练，但用 EMA 防止剧烈波动
            self._set_phase(3, epoch)
            self.model.prototypes.requires_grad = True

            # EMA 更新原型
            with torch.no_grad():
                if self.prototype_shadow is None:
                    # 首次进入 Phase 3，初始化影子副本
                    self.prototype_shadow = self.model.prototypes.data.clone()
                else:
                    # EMA 更新：shadow = decay * shadow + (1 - decay) * current
                    self.prototype_shadow = (
                            self.ema_decay * self.prototype_shadow +
                            (1 - self.ema_decay) * self.model.prototypes.data
                    )
                    # 用 EMA 平滑后的值覆盖当前原型
                    self.model.prototypes.data = self.prototype_shadow.clone()

    def _set_phase(self, phase, epoch):
        """打印 phase 变化信息"""
        if self.current_phase != phase:
            self.current_phase = phase
            if self.verbose:
                phase_names = {
                    1: "Phase 1: FROZEN (Warmup)",
                    2: "Phase 2: TRAINABLE (Transition)",
                    3: "Phase 3: TRAINABLE + EMA (Refinement)"
                }
                print(f"[PPD] Epoch {epoch}: Entering {phase_names[phase]}")

    def get_current_phase(self):
        """返回当前 phase 编号"""
        return self.current_phase

    def get_phase_info(self):
        """返回 phase 配置信息（用于 logging）"""
        return {
            "warmup_epochs": self.warmup_epochs,
            "transition_epochs": self.transition_epochs,
            "total_epochs": self.total_epochs,
            "ema_decay": self.ema_decay
        }

# ==================== [END NEW] PPD 调度器 ====================


# ==================== [NEW 2024-12-15] IRLLRec 模型 ====================
# =============================================================================
# IRLLRec: Intent Representation Learning with LLM for Recommendation
# 论文: SIGIR 2025
# 
# 核心创新（简化版，适配序列推荐）：
# 1. 意图原型学习 (user_intent, item_intent)
# 2. 意图分解 (r = softmax(e @ C) @ C.T)
# 3. 多层次蒸馏 (L_kd + L_kd_int + L_kd_int_2 + L_ITM)
# 4. 动量编码器 (int_mlp_m, EMA 更新)
# 
# 注意：简化版不包含 GSL（图结构学习），因为序列推荐没有显式的用户-物品图
# =============================================================================
class IRLLRec(SASRec_backbone):
    """
    IRLLRec: Intent Representation Learning (SIGIR 2025)
    基于 SASRec backbone 的序列推荐适配版本
    
    核心组件：
    - 意图原型矩阵: user_intent [emb_size, K], item_intent [emb_size, K]
    - 意图分解: r = softmax(e @ C) @ C.T
    - Profile MLP: usr/itm_emb_np.pkl → hidden_dim (粗粒度)
    - Intent MLP: user/item_intent_emb_3.pkl → hidden_dim (细粒度)
    - 动量 Intent MLP: EMA 更新的教师模型
    
    损失函数：
    - L_rec: 推荐损失 (CE/BPR)
    - L_kd: Profile 级别对齐 (InfoNCE)
    - L_kd_int: Intent 级别对齐 (InfoNCE) ⭐核心
    - L_kd_int_2: 加噪对比损失 (Translation Alignment)
    - L_ITM: 动量蒸馏损失 (Interaction-Text Matching)
    """
    
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        # ==================== 保存参数 ====================
        self.key_words = key_words
        self.device = device
        
        # ==================== ID Embedding ====================
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )
        
        # ==================== IRLLRec 超参数 ====================
        # 意图原型数量 K
        self.intent_num = key_words.get('intent_num', 128)
        # Profile 蒸馏权重
        self.kd_weight = key_words.get('kd_weight', 0.01)
        # Profile 蒸馏温度
        self.kd_temperature = key_words.get('kd_temperature', 0.2)
        # Intent 蒸馏权重
        self.kd_int_weight = key_words.get('kd_int_weight', 0.02)
        # Intent 蒸馏温度
        self.kd_int_temperature = key_words.get('kd_int_temperature', 0.2)
        # 加噪对比损失权重
        self.kd_int_weight_2 = key_words.get('kd_int_weight_2', 1e-7)
        # 动量蒸馏损失权重
        self.kd_int_weight_3 = key_words.get('kd_int_weight_3', 1e-7)
        # 动量系数
        self.momentum = key_words.get('momentum', 0.999)
        # LLM 维度
        self.llm_dim = key_words.get('llm_dim', 3072)
        self.profile_dim = key_words.get('profile_dim', 1536)
        
        # ==================== 意图原型矩阵 ====================
        # 用户意图原型 [hidden_dim, intent_num]
        self.user_intent = nn.Parameter(
            torch.empty(self.hidden_dim, self.intent_num)
        )
        nn.init.xavier_uniform_(self.user_intent)
        # 物品意图原型 [hidden_dim, intent_num]
        self.item_intent = nn.Parameter(
            torch.empty(self.hidden_dim, self.intent_num)
        )
        nn.init.xavier_uniform_(self.item_intent)
        
        # ==================== LLM 嵌入（延迟加载）====================
        # Profile 嵌入（粗粒度）
        self.usrprf_embeds = None  # [user_num, profile_dim]
        self.itmprf_embeds = None  # [item_num, profile_dim]
        # Intent 嵌入（细粒度）
        self.usrint_embeds = None  # [user_num, llm_dim]
        self.itmint_embeds = None  # [item_num, llm_dim]
        
        # ==================== MLP 映射网络 ====================
        # Profile MLP: profile_dim → hidden_dim
        self.mlp = None  # 延迟初始化（等加载数据后知道维度）
        
        # Intent MLP (学生): llm_dim → hidden_dim
        self.int_mlp = nn.Sequential(
            nn.Linear(self.llm_dim, (self.llm_dim + self.hidden_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.llm_dim + self.hidden_dim) // 2, self.hidden_dim)
        )
        
        # Intent MLP (教师/动量): 结构相同，EMA 更新
        self.int_mlp_m = nn.Sequential(
            nn.Linear(self.llm_dim, (self.llm_dim + self.hidden_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.llm_dim + self.hidden_dim) // 2, self.hidden_dim)
        )
        
        # ==================== 初始化权重 ====================
        self._init_irllrec_weights()
        
        # ==================== 复制参数到动量模型 ====================
        self._copy_params_to_momentum()
        
    def _init_irllrec_weights(self):
        """初始化 IRLLRec 特有的权重"""
        # 初始化 ID Embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.02)
        # 初始化 Intent MLP
        for module in self.int_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        for module in self.int_mlp_m:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    @torch.no_grad()
    def _copy_params_to_momentum(self):
        """复制学生模型参数到教师模型（初始化时调用）"""
        for param, param_m in zip(self.int_mlp.parameters(), 
                                   self.int_mlp_m.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """EMA 更新教师模型参数（每个 batch 调用）"""
        for param, param_m in zip(self.int_mlp.parameters(), 
                                   self.int_mlp_m.parameters()):
            param_m.data = param_m.data * self.momentum + \
                          param.data * (1.0 - self.momentum)
    
    def load_embeddings(self, usrprf_path, itmprf_path, usrint_path, itmint_path):
        """
        加载 LLM 嵌入文件
        
        Args:
            usrprf_path: 用户 Profile 嵌入路径 (usr_emb_np.pkl)，可为 None
            itmprf_path: 物品 Profile 嵌入路径 (itm_emb_np.pkl)，可为 None
            usrint_path: 用户 Intent 嵌入路径 (user_intent_emb_3.pkl 或 usr_intent_emb.pkl)
            itmint_path: 物品 Intent 嵌入路径 (item_intent_emb_3.pkl 或 itm_intent_emb.pkl)
        """
        import os
        
        # ==================== [COMMENTED] Profile 嵌入（粗粒度）- AlphaFuse 中没有这两个文件 ====================
        # 加载 Profile 嵌入（粗粒度）
        # if usrprf_path is not None and os.path.exists(usrprf_path):
        #     with open(usrprf_path, 'rb') as f:
        #         usrprf = pickle.load(f)
        #     self.usrprf_embeds = torch.tensor(usrprf, dtype=torch.float32).to(self.device)
        #     print(f"[IRLLRec] Loaded user profile embedding: {self.usrprf_embeds.shape}")
        #     
        #     # 初始化 Profile MLP（根据实际维度）
        #     actual_profile_dim = self.usrprf_embeds.shape[1]
        #     self.mlp = nn.Sequential(
        #         nn.Linear(actual_profile_dim, (actual_profile_dim + self.hidden_dim) // 2),
        #         nn.LeakyReLU(),
        #         nn.Linear((actual_profile_dim + self.hidden_dim) // 2, self.hidden_dim)
        #     ).to(self.device)
        #     # 初始化权重
        #     for module in self.mlp:
        #         if isinstance(module, nn.Linear):
        #             nn.init.xavier_uniform_(module.weight)
        # else:
        #     print(f"[IRLLRec] Skipping user profile embedding (not available)")
        # 
        # if itmprf_path is not None and os.path.exists(itmprf_path):
        #     with open(itmprf_path, 'rb') as f:
        #         itmprf = pickle.load(f)
        #     self.itmprf_embeds = torch.tensor(itmprf, dtype=torch.float32).to(self.device)
        #     print(f"[IRLLRec] Loaded item profile embedding: {self.itmprf_embeds.shape}")
        # else:
        #     print(f"[IRLLRec] Skipping item profile embedding (not available)")
        # ==================== [END COMMENTED] ====================
        
        # Profile 嵌入在 AlphaFuse 中不可用，跳过
        print(f"[IRLLRec] Profile embeddings (usr_emb_np.pkl, itm_emb_np.pkl) not used in AlphaFuse")
        
        # 加载 Intent 嵌入（细粒度）
        if os.path.exists(usrint_path):
            with open(usrint_path, 'rb') as f:
                usrint = pickle.load(f)
            self.usrint_embeds = torch.tensor(usrint, dtype=torch.float32).to(self.device)
            print(f"[IRLLRec] Loaded user intent embedding: {self.usrint_embeds.shape}")
            
            # 检查维度是否匹配，如果不匹配则重新初始化 MLP
            actual_llm_dim = self.usrint_embeds.shape[1]
            if actual_llm_dim != self.llm_dim:
                print(f"[IRLLRec] Adjusting int_mlp for dim {actual_llm_dim}")
                self.llm_dim = actual_llm_dim
                self.int_mlp = nn.Sequential(
                    nn.Linear(self.llm_dim, (self.llm_dim + self.hidden_dim) // 2),
                    nn.LeakyReLU(),
                    nn.Linear((self.llm_dim + self.hidden_dim) // 2, self.hidden_dim)
                ).to(self.device)
                self.int_mlp_m = nn.Sequential(
                    nn.Linear(self.llm_dim, (self.llm_dim + self.hidden_dim) // 2),
                    nn.LeakyReLU(),
                    nn.Linear((self.llm_dim + self.hidden_dim) // 2, self.hidden_dim)
                ).to(self.device)
                self._init_irllrec_weights()
                self._copy_params_to_momentum()
        else:
            print(f"[IRLLRec] Warning: User intent file not found: {usrint_path}")
        
        if os.path.exists(itmint_path):
            with open(itmint_path, 'rb') as f:
                itmint = pickle.load(f)
            self.itmint_embeds = torch.tensor(itmint, dtype=torch.float32).to(self.device)
            print(f"[IRLLRec] Loaded item intent embedding: {self.itmint_embeds.shape}")
        else:
            print(f"[IRLLRec] Warning: Item intent file not found: {itmint_path}")
    
    def embed_ID(self, x):
        """获取物品 ID 嵌入"""
        return self.item_embeddings(x)
    
    def return_item_emb(self):
        """返回全量物品嵌入"""
        return self.item_embeddings.weight
    
    def intent_decompose(self, embeds, intent_matrix):
        """
        意图分解：将嵌入映射到意图空间
        
        公式 9-10：
        P(c^k | e) = softmax(e @ C)
        r = P @ C.T = softmax(e @ C) @ C.T
        
        Args:
            embeds: [B, D] 输入嵌入
            intent_matrix: [D, K] 意图原型矩阵
        
        Returns:
            [B, D] 意图分解后的嵌入
        """
        # softmax(e @ C) @ C.T
        return torch.softmax(embeds @ intent_matrix, dim=-1) @ intent_matrix.T
    
    def cal_infonce_loss(self, anchor, positive, negatives, temperature):
        """
        InfoNCE 对比损失
        
        Args:
            anchor: [B, D] 锚点嵌入
            positive: [B, D] 正样本嵌入
            negatives: [N, D] 负样本池（包含正样本）
            temperature: 温度参数
        
        Returns:
            InfoNCE 损失
        """
        # L2 归一化
        anchor_norm = F.normalize(anchor, p=2, dim=-1)
        positive_norm = F.normalize(positive, p=2, dim=-1)
        negatives_norm = F.normalize(negatives, p=2, dim=-1)
        
        # 正样本相似度
        pos_sim = (anchor_norm * positive_norm).sum(dim=-1) / temperature  # [B]
        
        # 负样本相似度（与所有 negatives 计算）
        neg_sim = anchor_norm @ negatives_norm.T / temperature  # [B, N]
        
        # InfoNCE: -log(exp(pos) / sum(exp(neg)))
        # numerator = -pos_sim
        # denominator = torch.logsumexp(neg_sim, dim=-1)
        # loss = (numerator + denominator).sum()
        
        # 简化计算：使用 logsumexp 技巧
        logits = neg_sim  # [B, N]
        loss = -pos_sim + torch.logsumexp(logits, dim=-1)
        
        return loss.sum()
    
    def ssl_con_loss(self, embeds1, embeds2):
        """
        SSL 对比损失（用于加噪对比）
        
        Args:
            embeds1: [N, D] 嵌入1
            embeds2: [N, D] 嵌入2
        
        Returns:
            对比损失
        """
        embeds1_norm = F.normalize(embeds1, p=2, dim=-1)
        embeds2_norm = F.normalize(embeds2, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = embeds1_norm @ embeds2_norm.T  # [N, N]
        
        # 对角线是正样本
        pos = torch.diag(sim_matrix)
        
        # 对比损失
        loss = -pos + torch.logsumexp(sim_matrix, dim=-1)
        
        return loss.mean()
    
    def calculate_irllrec_loss(self, seq_output, user_ids, item_ids):
        """
        计算 IRLLRec 的所有蒸馏损失
        
        Args:
            seq_output: [B, D] 序列编码输出（代表用户表示）
            user_ids: [B] 用户 ID
            item_ids: [B] 物品 ID（正样本）
        
        Returns:
            dict: 包含所有损失项的字典
        """
        losses = {}
        B = seq_output.shape[0]
        
        # ============================================================
        # 步骤1：意图分解
        # ============================================================
        # 用户意图分解
        user_int = self.intent_decompose(seq_output, self.user_intent)  # [B, D]
        
        # 物品嵌入和意图分解
        item_embs = self.embed_ID(item_ids)  # [B, D]
        item_int = self.intent_decompose(item_embs, self.item_intent)  # [B, D]
        
        # ============================================================
        # 步骤2：L_kd - Profile 级别对齐（如果有 Profile 嵌入）
        # ============================================================
        if self.usrprf_embeds is not None and self.mlp is not None:
            # 获取 batch 对应的 Profile 嵌入
            usrprf_batch = self.usrprf_embeds[user_ids]  # [B, profile_dim]
            usrprf_mapped = self.mlp(usrprf_batch)  # [B, D]
            
            # 用户侧 Profile 对齐
            kd_loss_user = self.cal_infonce_loss(
                seq_output, usrprf_mapped, 
                self.mlp(self.usrprf_embeds), 
                self.kd_temperature
            )
            
            # 物品侧 Profile 对齐（如果有）
            if self.itmprf_embeds is not None:
                itmprf_batch = self.itmprf_embeds[item_ids]  # [B, profile_dim]
                itmprf_mapped = self.mlp(itmprf_batch)  # [B, D]
                
                kd_loss_item = self.cal_infonce_loss(
                    item_embs, itmprf_mapped,
                    itmprf_mapped,  # 使用 batch 内作为负样本
                    self.kd_temperature
                )
                kd_loss = (kd_loss_user + kd_loss_item) / B
            else:
                kd_loss = kd_loss_user / B
            
            losses['kd_loss'] = kd_loss * self.kd_weight
        else:
            losses['kd_loss'] = torch.tensor(0.0).to(self.device)
        
        # ============================================================
        # 步骤3：L_kd_int - Intent 级别对齐（核心创新）
        # ============================================================
        if self.usrint_embeds is not None:
            # 获取 batch 对应的 Intent 嵌入
            usrint_batch = self.usrint_embeds[user_ids]  # [B, llm_dim]
            usrint_mapped = self.int_mlp(usrint_batch)  # [B, D]
            
            # 用户侧 Intent 对齐：交互意图 ↔ 文本意图
            kd_int_loss_user = self.cal_infonce_loss(
                user_int, usrint_mapped,
                self.int_mlp(self.usrint_embeds),
                self.kd_int_temperature
            )
            
            # 物品侧 Intent 对齐（如果有）
            if self.itmint_embeds is not None:
                itmint_batch = self.itmint_embeds[item_ids]  # [B, llm_dim]
                itmint_mapped = self.int_mlp(itmint_batch)  # [B, D]
                
                kd_int_loss_item = self.cal_infonce_loss(
                    item_int, itmint_mapped,
                    itmint_mapped,
                    self.kd_int_temperature
                )
                kd_int_loss = (kd_int_loss_user + kd_int_loss_item) / B
            else:
                kd_int_loss = kd_int_loss_user / B
            
            losses['kd_int_loss'] = kd_int_loss * self.kd_int_weight
        else:
            losses['kd_int_loss'] = torch.tensor(0.0).to(self.device)
        
        # ============================================================
        # 步骤4：L_kd_int_2 - 加噪对比损失 (Translation Alignment)
        # ============================================================
        if self.usrint_embeds is not None:
            # 获取交互意图和文本意图
            all_user_int = self.intent_decompose(seq_output, self.user_intent)
            all_text_int = self.int_mlp(self.usrint_embeds[user_ids])
            
            # 添加高斯噪声
            noise_r = torch.randn_like(all_user_int)
            noise_z = torch.randn_like(all_text_int)
            
            r_prime = all_user_int + all_user_int * noise_r
            z_prime = all_text_int + all_text_int * noise_z
            
            # 对比损失
            kd_int_2_loss = self.ssl_con_loss(z_prime, r_prime)
            losses['kd_int_2_loss'] = kd_int_2_loss * self.kd_int_weight_2
        else:
            losses['kd_int_2_loss'] = torch.tensor(0.0).to(self.device)
        
        # ============================================================
        # 步骤5：L_ITM - 动量蒸馏损失
        # ============================================================
        if self.usrint_embeds is not None:
            # 更新动量模型
            self._momentum_update()
            
            # 学生模型输出
            student_out = self.int_mlp(self.usrint_embeds[user_ids])
            # 教师模型输出（无梯度）
            with torch.no_grad():
                teacher_out = self.int_mlp_m(self.usrint_embeds[user_ids])
            
            # KL 散度损失（简化版：使用 MSE）
            itm_loss = F.mse_loss(student_out, teacher_out)
            losses['itm_loss'] = itm_loss * self.kd_int_weight_3
        else:
            losses['itm_loss'] = torch.tensor(0.0).to(self.device)
        
        # ============================================================
        # 汇总所有损失
        # ============================================================
        losses['total_irllrec_loss'] = (
            losses['kd_loss'] + 
            losses['kd_int_loss'] + 
            losses['kd_int_2_loss'] + 
            losses['itm_loss']
        )
        
        return losses

    # ==================== [NEW 2024-12-16] 重写 calculate_infonce_loss ====================
    # 问题：原来 IRLLRec 没有重写这个方法，导致蒸馏损失不会被计算
    # 解决：重写方法，在基础 InfoNCE 损失上添加 IRLLRec 特有的蒸馏损失
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, user_ids=None):
        """
        重写 InfoNCE 损失，集成 IRLLRec 特有的蒸馏损失
        
        注意：需要 train.py 传入 user_ids 才能计算完整的蒸馏损失
        如果没有 user_ids，只返回基础 InfoNCE 损失
        """
        # Step 1: 计算基础 InfoNCE 损失（调用父类方法）
        rec_loss = super().calculate_infonce_loss(sequences, target, neg_ratio, temperature)
        
        # Step 2: 如果没有 user_ids 或 Intent 嵌入，只返回基础损失
        if user_ids is None or self.usrint_embeds is None:
            return rec_loss
        
        # Step 3: 获取序列表示
        seq_output = self.forward(sequences)  # [B, D]
        
        # Step 4: 计算 IRLLRec 蒸馏损失
        irll_losses = self.calculate_irllrec_loss(seq_output, user_ids, target)
        
        # Step 5: 返回总损失
        total_loss = rec_loss + irll_losses['total_irllrec_loss']
        
        return total_loss
    # ==================== [END NEW] ====================

# ==================== [END NEW] IRLLRec 模型 ====================
