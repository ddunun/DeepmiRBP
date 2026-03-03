#!/usr/bin/env python3
"""
colab_post_toolkit_tf.py
========================
DeepmiRBP 训练后工具包（TensorFlow / Keras）

功能：
  1) 模型可用性验证 —— 重新读取 RBP 数据、按训练时相同逻辑拆分，
     在测试集上计算 Accuracy / Precision / Recall / F1 / ROC-AUC，
     输出 metrics.json
  2) 给定一条 miRNA 序列，利用 Attention 层 embedding + 余弦相似度
     输出候选 RBP 排名（Top-k），保存为 CSV

⚠️  排名结果仅为 computational evidence（embedding 空间中的相似度），
    需要 CLIP-seq / RIP-seq 等实验数据进一步验证。

使用示例（Colab）：
  !python colab_post_toolkit_tf.py \\
      --repo-dir /content/DeepmiRBP \\
      --weights-path /content/drive/MyDrive/Claude/best_model.keras \\
      --mirna-name hsa-miR-19b-2-5p \\
      --mirna-seq AGUUUUGCAGGUUUGCAUUUCA \\
      --top-k 20 \\
      --per-rbp-sample 500
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  0. 导入                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 屏蔽 TF 冗余日志
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("toolkit")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. 自定义 Attention 层（必须和训练时完全一致）                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class Attention(Layer):
    """Soft Attention — 对 BiLSTM 时序输出加权求和（与训练脚本一致）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. 数据读取（复刻训练脚本逻辑，保证编码 & 拆分一致）                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def _process_sequences(sequences: List[str]) -> Tuple[np.ndarray, set]:
    """字符 → 整数编码 + post-padding（与训练脚本 process_sequences 一致）"""
    if not sequences:
        return np.array([]), set()
    unique_characters = set("".join(sequences))
    letter2number = {ch: i for i, ch in enumerate(sorted(unique_characters), start=1)}
    encoded = [[letter2number.get(c, 0) for c in seq] for seq in sequences]
    return pad_sequences(encoded, padding="post"), unique_characters


def _clean_data(
    sequences: List[str], labels: List
) -> Tuple[List[str], List[int]]:
    """保留 label ∈ {0, 1} 的有效行"""
    clean_seq, clean_lab = [], []
    for seq, lab in zip(sequences, labels):
        if pd.isna(seq) or pd.isna(lab):
            continue
        try:
            v = int(float(lab))
            if v in (0, 1):
                clean_seq.append(str(seq))
                clean_lab.append(v)
        except (ValueError, TypeError):
            pass
    return clean_seq, clean_lab


def _read_single_fa(file_path: str) -> Tuple[np.ndarray, np.ndarray, set]:
    """读取单个 .fa 文件 → (padded_sequences, labels, unique_chars)"""
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        names=["sequence", "label"],
        engine="python",
        on_bad_lines="skip",
    )
    seqs, labs = _clean_data(df["sequence"].tolist(), df["label"].tolist())
    padded, chars = _process_sequences(seqs)
    return padded, np.array(labs, dtype=np.float32), chars


def load_all_rbp(rbp_dir: str) -> Tuple[Dict, set]:
    """
    读取 RBP-Data 目录下所有 .fa 文件，返回:
      data_dict[protein_name] = {'sequences': ndarray, 'labels': ndarray}
      global_unique_characters: set
    """
    if not os.path.isdir(rbp_dir):
        raise FileNotFoundError(f"RBP 数据目录不存在: {rbp_dir}")

    fa_files = sorted(f for f in os.listdir(rbp_dir) if f.endswith(".fa"))
    if not fa_files:
        raise FileNotFoundError(f"RBP 目录下没有 .fa 文件: {rbp_dir}")

    data_dict: Dict = {}
    global_chars: set = set()

    log.info("正在读取 %d 个 RBP 数据文件 ...", len(fa_files))
    for idx, fname in enumerate(fa_files):
        prot = fname.split(".")[0]
        path = os.path.join(rbp_dir, fname)
        try:
            seqs, labs, chars = _read_single_fa(path)
            if len(seqs) == 0:
                log.warning("  文件 %s 无有效数据，跳过", fname)
                continue
            data_dict[prot] = {"sequences": seqs, "labels": labs}
            global_chars.update(chars)
        except Exception as exc:
            log.warning("  跳过 %s : %s", fname, exc)

        if (idx + 1) % 20 == 0 or idx == len(fa_files) - 1:
            log.info("  进度 %d / %d", idx + 1, len(fa_files))

    if not data_dict:
        raise RuntimeError("读取完  RBP 文件后 data_dict 为空，请检查数据")

    n_pos = sum(int(np.sum(d["labels"] == 1)) for d in data_dict.values())
    n_neg = sum(int(np.sum(d["labels"] == 0)) for d in data_dict.values())
    log.info(
        "读取完成: %d 个蛋白, 正样本 %s, 负样本 %s",
        len(data_dict),
        f"{n_pos:,}",
        f"{n_neg:,}",
    )
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"数据只有单类别 (正:{n_pos}, 负:{n_neg})，无法正常评估/排名"
        )
    return data_dict, global_chars


def merge_and_split(
    data_dict: Dict,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    合并所有蛋白数据，统一 padding 到最大长度，做分层 train/test 划分。
    返回 (X_train, X_test, y_train, y_test, max_length)
    """
    all_seqs: List[np.ndarray] = []
    all_labs: List[np.ndarray] = []
    for v in data_dict.values():
        all_seqs.append(v["sequences"])
        all_labs.append(v["labels"])

    # 统一 padding（不同文件的 padded 长度理论上都是 101，但防御性再 pad 一次）
    max_len = max(s.shape[1] for s in all_seqs)
    unified = [
        pad_sequences(s, maxlen=max_len, padding="post", truncating="post")
        for s in all_seqs
    ]

    X = np.concatenate(unified, axis=0)
    y = np.concatenate(all_labs, axis=0)

    log.info("合并数据: X=%s  y=%s  max_length=%d", X.shape, y.shape, max_len)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(
        "拆分: 训练 %s 条 (正 %.1f%%)  测试 %s 条 (正 %.1f%%)",
        f"{len(y_tr):,}",
        np.mean(y_tr) * 100,
        f"{len(y_te):,}",
        np.mean(y_te) * 100,
    )
    return X_tr, X_te, y_tr, y_te, max_len


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. 模型加载                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_model_and_embedding(weights_path: str) -> Tuple[Model, Model, int]:
    """
    加载训练好的 .keras 模型，并构建 embedding_model。
    返回 (model, embedding_model, max_length)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"模型文件不存在: {weights_path}")

    log.info("正在加载模型: %s", weights_path)
    model = tf.keras.models.load_model(
        weights_path, custom_objects={"Attention": Attention}
    )

    # 从模型输入形状推断 max_length
    expected_len = model.input_shape[1]
    log.info(
        "模型加载完成 — 输入长度: %d, 参数量: %s",
        expected_len,
        f"{model.count_params():,}",
    )

    # 提取 Attention 层输出作为 embedding
    try:
        att_output = model.get_layer("attention").output
    except ValueError:
        # 回退：按类名搜索
        att_layers = [l for l in model.layers if isinstance(l, Attention)]
        if not att_layers:
            raise RuntimeError("模型中找不到 Attention 层，无法构建 embedding_model")
        att_output = att_layers[-1].output
        log.warning("未找到名为 'attention' 的层，使用最后一个 Attention 实例")

    embedding_model = Model(
        inputs=model.input, outputs=att_output, name="embedding_model"
    )
    log.info("embedding_model 输出维度: %s", embedding_model.output_shape)

    return model, embedding_model, expected_len


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. 任务 1: 模型可用性验证                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def validate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> Dict:
    """在测试集上评估模型，打印指标并保存 metrics.json"""
    log.info("正在模型推理 (%s 条) ...", f"{len(y_test):,}")

    y_prob = model.predict(X_test, batch_size=256, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC 需要两个类别都存在
    if len(np.unique(y_test)) < 2:
        log.warning("测试集只有单类别，AUC-ROC 无法计算")
        auc = None
    else:
        auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "auc_roc": round(auc, 6) if auc is not None else None,
        "confusion_matrix": cm,
        "test_samples": int(len(y_test)),
        "positive_ratio": round(float(np.mean(y_test)), 4),
    }

    # 打印
    log.info("=" * 55)
    log.info("  模型验证结果 (测试集)")
    log.info("=" * 55)
    log.info("  Accuracy  : %.4f", acc)
    log.info("  Precision : %.4f", prec)
    log.info("  Recall    : %.4f", rec)
    log.info("  F1 Score  : %.4f", f1)
    log.info("  AUC-ROC   : %s", f"{auc:.4f}" if auc else "N/A")
    log.info("  Confusion : %s", cm)
    log.info("=" * 55)

    # 保存 JSON
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
    log.info("指标已保存 → %s", json_path)

    return metrics


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. 任务 2: miRNA → 候选 RBP 排名                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def encode_mirna(mirna_seq: str, max_length: int) -> np.ndarray:
    """
    将单条 miRNA 序列编码为模型输入（与训练脚本 _process_sequences 一致）。
    返回 shape = (1, max_length)
    """
    mirna_seq = mirna_seq.strip().upper()
    if not mirna_seq:
        raise ValueError("miRNA 序列为空")

    # 跟训练脚本完全一致：sorted unique chars → 1-indexed mapping
    unique_chars = sorted(set(mirna_seq))
    letter2num = {ch: i for i, ch in enumerate(unique_chars, start=1)}
    encoded = [letter2num[c] for c in mirna_seq]

    padded = pad_sequences([encoded], maxlen=max_length, padding="post", truncating="post")
    return padded  # (1, max_length)


def rank_rbps_for_mirna(
    mirna_name: str,
    mirna_seq: str,
    data_dict: Dict,
    embedding_model: Model,
    max_length: int,
    per_rbp_sample: int,
    top_k: int,
    output_dir: str,
) -> pd.DataFrame:
    """
    计算 miRNA 与各 RBP 的 embedding 余弦相似度排名。

    方法：
      1. 将 miRNA 序列编码 → embedding_model → miRNA embedding
      2. 对每个 RBP：从正样本中随机抽 per_rbp_sample 条 → embedding_model →
         得到一组 RBP embedding
      3. 计算 miRNA embedding 与每条 RBP embedding 的余弦相似度，
         取 **平均值** 作为该 RBP 的分数（平均相似度比最大值更稳定，
         不易被单条异常序列主导）
      4. 降序排列，输出 Top-k
    """
    log.info("miRNA 名称: %s", mirna_name)
    log.info("miRNA 序列: %s  (长度 %d)", mirna_seq, len(mirna_seq))

    # ── miRNA embedding ──
    mirna_input = encode_mirna(mirna_seq, max_length)
    mirna_emb = embedding_model.predict(mirna_input, verbose=0)  # (1, emb_dim)
    mirna_emb_norm = normalize(mirna_emb, axis=1, norm="l2")
    log.info("miRNA embedding 维度: %s", mirna_emb.shape)

    # ── 逐 RBP 计算相似度 ──
    scores: Dict[str, float] = {}
    rng = np.random.RandomState(42)

    for prot, data in data_dict.items():
        seqs = data["sequences"]
        labs = data["labels"]

        # --- 取正样本 ---
        pos_mask = labs == 1
        pos_seqs = seqs[pos_mask]

        if len(pos_seqs) == 0:
            log.debug("  %s 无正样本，跳过", prot)
            continue

        # 采样
        n_sample = min(per_rbp_sample, len(pos_seqs))
        idx = rng.choice(len(pos_seqs), size=n_sample, replace=False)
        sampled = pos_seqs[idx]

        # 统一 padding 到 max_length（防御性）
        sampled_padded = pad_sequences(
            sampled, maxlen=max_length, padding="post", truncating="post"
        )

        # embedding
        rbp_emb = embedding_model.predict(sampled_padded, batch_size=256, verbose=0)
        rbp_emb_norm = normalize(rbp_emb, axis=1, norm="l2")

        # 余弦相似度: (1, emb_dim) vs (n_sample, emb_dim) → (1, n_sample)
        sim_matrix = cosine_similarity(mirna_emb_norm, rbp_emb_norm)
        avg_sim = float(np.mean(sim_matrix))
        scores[prot] = avg_sim

    if not scores:
        raise RuntimeError("所有 RBP 均无正样本或计算失败，无法生成排名")

    # ── 排序 ──
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 构建 DataFrame
    rows = [
        {"rank": i + 1, "rbp": prot, "cosine_similarity": round(sim, 6)}
        for i, (prot, sim) in enumerate(sorted_items)
    ]
    df_all = pd.DataFrame(rows)
    df_topk = df_all.head(top_k)

    # 打印 top-k
    log.info("")
    log.info("=" * 55)
    log.info("  miRNA → 候选 RBP 排名  (Top %d / %d)", top_k, len(scores))
    log.info("  方法: Attention embedding 平均余弦相似度")
    log.info("=" * 55)
    for _, row in df_topk.iterrows():
        log.info(
            "  #%3d  %-18s  cosine = %.6f",
            row["rank"],
            row["rbp"],
            row["cosine_similarity"],
        )
    log.info("=" * 55)
    log.info(
        "⚠️  以上排名为 embedding 空间中的 computational evidence，\n"
        "    需结合 CLIP-seq / RIP-seq 等实验数据做进一步功能验证。"
    )

    # 保存 CSV（完整排名，不只 top-k）
    csv_name = f"{mirna_name}_rbp_ranking.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df_all.to_csv(csv_path, index=False)
    log.info("完整排名已保存 → %s  (%d 条)", csv_path, len(df_all))

    return df_all


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Drive 挂载 & 输出目录                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def ensure_output_dir(output_dir: str) -> str:
    """挂载 Google Drive（仅 Colab），并确保输出目录存在"""
    try:
        import google.colab  # noqa
        from google.colab import drive

        if not os.path.ismount("/content/drive"):
            log.info("正在挂载 Google Drive ...")
            drive.mount("/content/drive", force_remount=False)
        else:
            log.info("Google Drive 已挂载")
    except ImportError:
        log.info("非 Colab 环境，跳过 Drive 挂载")

    os.makedirs(output_dir, exist_ok=True)
    log.info("输出目录: %s", output_dir)
    return output_dir


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. 命令行参数                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepmiRBP 训练后工具包: 模型验证 + miRNA→RBP 排名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅验证模型
  python colab_post_toolkit_tf.py \\
      --repo-dir /content/DeepmiRBP \\
      --weights-path /content/drive/MyDrive/Claude/best_model.keras

  # 验证 + miRNA 排名
  python colab_post_toolkit_tf.py \\
      --repo-dir /content/DeepmiRBP \\
      --weights-path /content/drive/MyDrive/Claude/best_model.keras \\
      --mirna-name hsa-miR-19b-2-5p \\
      --mirna-seq AGUUUUGCAGGUUUGCAUUUCA \\
      --top-k 20
""",
    )

    p.add_argument(
        "--repo-dir",
        default="/content/DeepmiRBP",
        help="仓库根目录 (默认: /content/DeepmiRBP)",
    )
    p.add_argument(
        "--weights-path",
        required=True,
        help="训练好的 .keras 模型路径",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="测试集比例 (默认: 0.1)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/Claude",
        help="输出目录 (默认: /content/drive/MyDrive/Claude/)",
    )

    # miRNA 相关（可选）
    p.add_argument(
        "--mirna-name",
        default=None,
        help="miRNA 名称，例如 hsa-miR-19b-2-5p（不提供则跳过排名）",
    )
    p.add_argument(
        "--mirna-seq",
        default=None,
        help="miRNA 序列，例如 AGUUUUGCAGGUUUGCAUUUCA",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="输出排名前几位 (默认: 20)",
    )
    p.add_argument(
        "--per-rbp-sample",
        type=int,
        default=500,
        help="从每个 RBP 正样本中抽取多少条计算相似度 (默认: 500)",
    )

    args = p.parse_args(argv)

    # 校验
    if args.mirna_name and not args.mirna_seq:
        p.error("指定了 --mirna-name 但未提供 --mirna-seq")
    if args.mirna_seq and not args.mirna_name:
        p.error("指定了 --mirna-seq 但未提供 --mirna-name")

    return args


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. 主入口                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    log.info("=" * 60)
    log.info("DeepmiRBP Post-Training Toolkit (TensorFlow)")
    log.info("=" * 60)
    log.info("TensorFlow %s  |  GPU: %s",
             tf.__version__,
             "Yes" if tf.config.list_physical_devices("GPU") else "No")

    # ── 输出目录 ──
    output_dir = ensure_output_dir(args.output_dir)

    # ── 数据路径 ──
    rbp_dir = os.path.join(args.repo_dir, "Data", "RBP-Data")
    if not os.path.isdir(rbp_dir):
        log.error("RBP 数据目录不存在: %s", rbp_dir)
        sys.exit(1)

    # ── 加载模型 ──
    model, embedding_model, model_max_len = load_model_and_embedding(
        args.weights_path
    )

    # ── 读取 RBP 数据 ──
    data_dict, global_chars = load_all_rbp(rbp_dir)
    log.info("全局字符集: %s", sorted(global_chars))

    # ── 合并 & 拆分 ──
    X_train, X_test, y_train, y_test, data_max_len = merge_and_split(
        data_dict, args.test_size, args.random_state
    )

    # 验证数据长度与模型期望长度一致
    if data_max_len != model_max_len:
        log.warning(
            "数据 max_length(%d) ≠ 模型期望 input_length(%d)，"
            "将统一 re-pad 到 %d",
            data_max_len,
            model_max_len,
            model_max_len,
        )
        X_train = pad_sequences(
            X_train, maxlen=model_max_len, padding="post", truncating="post"
        )
        X_test = pad_sequences(
            X_test, maxlen=model_max_len, padding="post", truncating="post"
        )
        # 也更新 data_dict 中各蛋白的序列（供 miRNA 排名用）
        for prot in data_dict:
            data_dict[prot]["sequences"] = pad_sequences(
                data_dict[prot]["sequences"],
                maxlen=model_max_len,
                padding="post",
                truncating="post",
            )

    effective_max_len = model_max_len

    # ══════════════════════════════════════════════════════════════
    #  任务 1 : 模型验证
    # ══════════════════════════════════════════════════════════════
    log.info("")
    log.info("▶ 任务 1: 模型可用性验证")
    metrics = validate_model(model, X_test, y_test, output_dir)

    # ══════════════════════════════════════════════════════════════
    #  任务 2 : miRNA → RBP 排名（可选）
    # ══════════════════════════════════════════════════════════════
    if args.mirna_seq:
        log.info("")
        log.info("▶ 任务 2: miRNA → 候选 RBP 排名")
        rank_rbps_for_mirna(
            mirna_name=args.mirna_name,
            mirna_seq=args.mirna_seq,
            data_dict=data_dict,
            embedding_model=embedding_model,
            max_length=effective_max_len,
            per_rbp_sample=args.per_rbp_sample,
            top_k=args.top_k,
            output_dir=output_dir,
        )
    else:
        log.info("未提供 --mirna-seq，跳过任务 2 (RBP 排名)")

    elapsed = time.time() - t0
    log.info("")
    log.info("全部完成，耗时 %.1f 秒", elapsed)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  运行                                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    main()
