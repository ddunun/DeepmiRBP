#!/usr/bin/env python3
"""
DeepmiRBP — 可直接在 Google Colab 上运行的完整训练脚本
=====================================================
功能：
  1. 从 GitHub 克隆 DeepmiRBP 仓库并读取数据
  2. 读取 RBP binding-site 序列（正/负样本）、PSSM 蛋白质特征、miRNA 序列
  3. 构建 BiLSTM + Attention 深度学习模型
  4. 训练、评估，并可视化结果
  5. 使用训练好的模型提取 embedding，计算 RNA-miRNA 相似度

使用方法（Colab）：
  在 Colab 新建 Notebook，创建一个代码单元格，粘贴以下代码即可。
  也可以把本文件上传后  %run DeepmiRBP_Colab.py
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 0 步：环境安装 & 克隆仓库                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
import subprocess, os, sys

def setup_environment():
    """克隆仓库并安装依赖（仅在 Colab 中需要执行一次）"""
    REPO_URL = "https://github.com/sbbi-unl/DeepmiRBP.git"
    REPO_DIR = "/content/DeepmiRBP"

    if not os.path.exists(REPO_DIR):
        print("📥 正在从 GitHub 克隆 DeepmiRBP 仓库 ...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
        print("✅ 仓库克隆完成")
    else:
        print("✅ 仓库已存在，跳过克隆")

    # 确保依赖已安装（Colab 已预装 TensorFlow / sklearn 等）
    try:
        import tensorflow
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "tensorflow", "scikit-learn", "matplotlib", "seaborn", "pandas", "numpy"],
                       check=True)

    # 挂载 Google Drive
    from google.colab import drive
    DRIVE_MOUNT = "/content/drive"
    if not os.path.ismount(DRIVE_MOUNT):
        print("📂 正在挂载 Google Drive ...")
        drive.mount(DRIVE_MOUNT)
        print("✅ Google Drive 挂载完成")
    else:
        print("✅ Google Drive 已挂载")

    # 确保 Drive 保存目录存在
    DRIVE_SAVE_DIR = "/content/drive/MyDrive/Claude"
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

    return REPO_DIR

# 判断运行环境
try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_DIR = setup_environment()
else:
    # 本地或 Codespace 环境直接使用工作区
    REPO_DIR = os.path.dirname(os.path.abspath(__file__))
    # 如果直接运行在 repo 根目录
    if not os.path.isdir(os.path.join(REPO_DIR, "Data")):
        REPO_DIR = "/workspaces/DeepmiRBP"

print(f"📁 数据根目录: {REPO_DIR}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 1 步：导入库                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Embedding, Dropout,
                                      Bidirectional, Concatenate, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

print(f"✅ TensorFlow 版本: {tf.__version__}")
print(f"✅ GPU 可用: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 2 步：定义数据路径                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
DATA_DIR   = os.path.join(REPO_DIR, "Data")
PSSM_DIR   = os.path.join(DATA_DIR, "PSSM")
RBP_DIR    = os.path.join(DATA_DIR, "RBP-Data")
MIRNA_PATH = os.path.join(DATA_DIR, "miRNA", "clean_miRNA_no_duplicates.txt")

# 验证路径
for p, name in [(PSSM_DIR, "PSSM"), (RBP_DIR, "RBP-Data"), (MIRNA_PATH, "miRNA")]:
    exists = os.path.exists(p)
    print(f"  {'✅' if exists else '❌'} {name}: {p}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 3 步：数据读取与预处理函数                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def process_sequences(sequences):
    """将 RNA 字符序列转为数字编码并 padding"""
    if not sequences:
        return np.array([]), set()
    unique_characters = set(''.join(sequences))
    letter2number = {l: i for i, l in enumerate(sorted(unique_characters), start=1)}
    processed_seqs = [[letter2number.get(char, 0) for char in seq] for seq in sequences]
    return pad_sequences(processed_seqs, padding='post'), unique_characters


def clean_data(sequences, labels):
    """清洗数据，只保留标签为 0 或 1 的样本"""
    cleaned_sequences = []
    cleaned_labels = []
    for seq, label in zip(sequences, labels):
        if pd.isna(seq) or pd.isna(label):
            continue
        try:
            cleaned_label = int(float(label))
            if cleaned_label in [0, 1]:
                cleaned_sequences.append(seq)
                cleaned_labels.append(cleaned_label)
        except ValueError:
            pass
    return cleaned_sequences, cleaned_labels


def read_and_process_data(file_path):
    """读取单个 RBP .fa 文件并返回编码序列、标签、字符集"""
    data = pd.read_csv(file_path, sep=r'\s+', names=['sequence', 'label'],
                       engine='python', on_bad_lines='skip')
    sequences, labels = clean_data(data['sequence'].tolist(), data['label'].tolist())
    processed_sequences, unique_characters = process_sequences(sequences)
    labels = np.array(labels).astype(np.float32)
    return processed_sequences, labels, unique_characters


def process_all_rbp_files(directory_path, max_files=None):
    """
    读取所有 RBP .fa 文件。
    参数:
        max_files: 限制读取文件数量（调试时可设为小数字以加快速度）
    """
    data_dict = {}
    global_unique_characters = set()
    file_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.fa')])

    if max_files:
        file_list = file_list[:max_files]

    print(f"\n📖 正在读取 {len(file_list)} 个 RBP 数据文件 ...")
    for i, file_name in enumerate(file_list):
        rbp_name = file_name.split('.')[0]
        file_path = os.path.join(directory_path, file_name)
        try:
            sequences, labels, unique_characters = read_and_process_data(file_path)
            if len(sequences) > 0:
                data_dict[rbp_name] = {'sequences': sequences, 'labels': labels}
                global_unique_characters.update(unique_characters)
        except Exception as e:
            print(f"  ⚠️ 跳过 {file_name}: {e}")

        if (i + 1) % 10 == 0 or i == len(file_list) - 1:
            print(f"  进度: {i + 1}/{len(file_list)}")

    print(f"✅ 成功读取 {len(data_dict)} 个蛋白质的结合数据")
    return data_dict, global_unique_characters


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 4 步：PSSM 处理器                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

class PSSMProcessor:
    """读取 PSSM 文件，调整长度至平均值，返回扁平化向量字典"""

    def __init__(self, path):
        self.path = path
        self.final_array = None

    def load_pssm(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        pssm = []
        for line in lines[3:-6]:
            parts = line.split()
            if len(parts) >= 22:
                pssm.append([int(x) for x in parts[2:22]])
        return np.array(pssm)

    def load_and_adjust_pssms(self):
        pssm_files = sorted([os.path.join(self.path, f)
                             for f in os.listdir(self.path) if f.endswith('.pssm')])
        pssm_data = []

        print(f"\n📖 正在读取 {len(pssm_files)} 个 PSSM 文件 ...")
        for file in pssm_files:
            pssm_matrix = self.load_pssm(file)
            if pssm_matrix.size > 0:
                filename = os.path.basename(file)
                pssm_data.append((filename, pssm_matrix))

        if not pssm_data:
            print("  ⚠️ 没有有效的 PSSM 文件")
            return {}

        # 调整到平均长度
        lengths = [pssm.shape[0] for _, pssm in pssm_data]
        average_length = sum(lengths) // len(lengths)
        print(f"  PSSM 长度统计 — 最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {average_length}")

        adjusted_pssms = []
        for filename, pssm in pssm_data:
            if pssm.shape[0] > average_length:
                adjusted_pssm = pssm[:average_length]
            elif pssm.shape[0] < average_length:
                repeat_times = average_length // pssm.shape[0] + 1
                adjusted_pssm = np.tile(pssm, (repeat_times, 1))[:average_length]
            else:
                adjusted_pssm = pssm
            adjusted_pssms.append((filename, adjusted_pssm.flatten()))

        # 构建字典
        pssm_dict = {}
        for filename, flat_vector in adjusted_pssms:
            if "-" in filename:
                key_part = filename.split('-')[0]
            else:
                key_part = filename.split('_')[0].replace('.pssm', '')
            pssm_dict[key_part] = flat_vector

        print(f"✅ 加载了 {len(pssm_dict)} 个 PSSM 条目")
        return pssm_dict


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 5 步：PSSM 降维 (PCA)                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def reduce_pssm_with_pca(pssm_dict, n_components=25):
    """对 PSSM 向量做 PCA 降维"""
    if not pssm_dict:
        return {}
    keys = list(pssm_dict.keys())
    data = np.array([pssm_dict[k] for k in keys])

    pca = PCA(n_components=min(n_components, data.shape[1], data.shape[0]))
    data_pca = pca.fit_transform(data)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA 保留 {pca.n_components_} 个主成分，解释方差: {explained_var:.1f}%")

    return {keys[i]: data_pca[i] for i in range(len(keys))}


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 6 步：自定义 Attention 层 + 模型构建                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class Attention(Layer):
    """Soft Attention Layer — 对 BiLSTM 的时序输出加权求和"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        base = super().get_config()
        return base


def build_deepmirbp_model(max_length, num_unique_chars,
                          embedding_dim=128,
                          lstm_units=256,
                          num_lstm_layers=3,
                          dropout_rate=0.5,
                          learning_rate=1e-4):
    """
    构建 DeepmiRBP 模型 (BiLSTM + Attention)

    参数:
        max_length:       输入序列最大长度
        num_unique_chars: 字符集大小（用于 Embedding 层）
        embedding_dim:    字符 embedding 维度
        lstm_units:       每层 LSTM 单元数
        num_lstm_layers:  BiLSTM 层数
        dropout_rate:     Dropout 比率
        learning_rate:    学习率

    返回:
        model:           训练用完整模型
        embedding_model: 提取 Attention 层输出的 embedding 模型
    """
    # --- 序列输入分支 ---
    seq_input = Input(shape=(max_length,), name='sequence_input')
    x = Embedding(input_dim=num_unique_chars + 1,
                  output_dim=embedding_dim,
                  input_length=max_length,
                  name='char_embedding')(seq_input)

    for i in range(num_lstm_layers):
        x = Bidirectional(
            LSTM(lstm_units, return_sequences=True, name=f'lstm_{i}'),
            name=f'bilstm_{i}'
        )(x)
        x = Dropout(dropout_rate, name=f'dropout_lstm_{i}')(x)

    # Attention
    attention_output = Attention(name='attention')(x)
    x = Dropout(dropout_rate, name='dropout_attention')(attention_output)

    # 全连接
    x = Dense(64, activation='relu', name='dense_64')(x)
    x = Dropout(dropout_rate, name='dropout_dense')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    # 完整模型
    model = Model(inputs=seq_input, outputs=output, name='DeepmiRBP')
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Embedding 模型（到 Attention 层）
    embedding_model = Model(inputs=seq_input, outputs=attention_output,
                            name='DeepmiRBP_Embedding')

    return model, embedding_model


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 7 步：评估与可视化                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def evaluate_model(model, X, y, dataset_name="Test"):
    """评估模型并打印指标"""
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = 0.0

    print(f"\n{'='*50}")
    print(f"📊 {dataset_name} 评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'y_prob': y_prob, 'y_pred': y_pred}


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title=''):
    """绘制训练 loss 和 accuracy 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'Accuracy — {title}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'Loss — {title}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 8 步：Embedding 提取 & miRNA 相似度分析                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_embeddings(data_dict, embedding_model, max_length, sample_size=500):
    """从训练好的模型中提取 RNA embedding"""
    from sklearn.preprocessing import normalize

    embedding_dict = {}
    for protein, data in data_dict.items():
        sequences = np.array(data['sequences'])
        labels = data['labels']

        # 只取正样本的一部分
        pos_mask = labels == 1
        pos_sequences = sequences[pos_mask]
        if len(pos_sequences) > sample_size:
            indices = np.random.choice(len(pos_sequences), sample_size, replace=False)
            pos_sequences = pos_sequences[indices]

        if len(pos_sequences) == 0:
            continue

        padded = pad_sequences(pos_sequences, maxlen=max_length,
                               padding='post', truncating='post')
        embeddings = embedding_model.predict(padded, verbose=0)
        embedding_dict[protein] = embeddings

    print(f"✅ 提取了 {len(embedding_dict)} 个蛋白质的 embedding")
    return embedding_dict


def compute_mirna_similarity(embedding_dict, mirna_embedding, top_k=20):
    """计算各蛋白质 embedding 与 miRNA embedding 的余弦相似度"""
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity

    mirna_flat = normalize(mirna_embedding, axis=1, norm='l2')

    similarity_scores = {}
    for protein, rna_emb in embedding_dict.items():
        rna_flat = normalize(rna_emb, axis=1, norm='l2')
        sim_matrix = cosine_similarity(rna_flat, mirna_flat)
        similarity_scores[protein] = float(np.mean(sim_matrix))

    # 排序
    sorted_scores = dict(sorted(similarity_scores.items(),
                                key=lambda x: x[1], reverse=True))

    print(f"\n🔬 miRNA 结合蛋白质相似度排名 (Top {top_k}):")
    print("-" * 45)
    for i, (prot, score) in enumerate(sorted_scores.items()):
        if i >= top_k:
            break
        print(f"  {i+1:3d}. {prot:15s}  相似度: {score:.4f}")

    return sorted_scores


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第 9 步：主训练流程                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    """完整的训练和评估流程"""

    # ----------------------------------------------------------------
    # 配置参数（在此调整超参数）
    # ----------------------------------------------------------------
    CONFIG = {
        # 数据
        'max_rbp_files':   None,   # 设为整数以限制文件数（调试用），None=全部
        'test_size':       0.1,    # 测试集比例
        'random_state':    42,

        # 模型
        'embedding_dim':   128,
        'lstm_units':      256,    # 原始为 512，降为 256 以节省 Colab 内存
        'num_lstm_layers': 3,
        'dropout_rate':    0.5,
        'learning_rate':   1e-4,

        # 训练
        'batch_size':      64,
        'epochs':          30,
        'patience':        10,     # EarlyStopping patience
    }

    print("=" * 60)
    print("🚀 DeepmiRBP 训练开始")
    print("=" * 60)

    # -----------------------------------------------------------
    # 9.1 读取 RBP 数据
    # -----------------------------------------------------------
    rbp_data_dict, global_unique_chars = process_all_rbp_files(
        RBP_DIR, max_files=CONFIG['max_rbp_files']
    )

    if not rbp_data_dict:
        print("❌ 没有读取到有效数据，请检查数据路径")
        return

    # 统计正负样本
    total_pos, total_neg = 0, 0
    for name, data in rbp_data_dict.items():
        n_pos = int(np.sum(data['labels'] == 1))
        n_neg = int(np.sum(data['labels'] == 0))
        total_pos += n_pos
        total_neg += n_neg

    print(f"\n📊 数据统计:")
    print(f"  蛋白质数量:   {len(rbp_data_dict)}")
    print(f"  总正样本数:   {total_pos:,}")
    print(f"  总负样本数:   {total_neg:,}")
    print(f"  正负比例:     {total_pos / max(total_neg, 1):.2f}")
    print(f"  字符集:       {sorted(global_unique_chars)}")

    # -----------------------------------------------------------
    # 9.2 合并所有蛋白质数据
    # -----------------------------------------------------------
    all_sequences = []
    all_labels = []
    for key in rbp_data_dict:
        all_sequences.extend(rbp_data_dict[key]['sequences'])
        all_labels.extend(rbp_data_dict[key]['labels'])

    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    print(f"\n  合并后数据 — 序列: {all_sequences.shape}, 标签: {all_labels.shape}")

    # -----------------------------------------------------------
    # 9.3 划分训练/测试集
    # -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences, all_labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=all_labels
    )

    print(f"  训练集: {X_train.shape[0]:,} 条   |   测试集: {X_test.shape[0]:,} 条")
    print(f"  训练集正样本比例: {np.mean(y_train):.3f}")
    print(f"  测试集正样本比例: {np.mean(y_test):.3f}")

    # 计算 max_length
    max_length = all_sequences.shape[1]  # pad_sequences 已统一长度
    num_unique = len(global_unique_chars)
    print(f"  序列最大长度: {max_length}  |  字符集大小: {num_unique}")

    # -----------------------------------------------------------
    # 9.4 构建模型（优先从已有模型继续训练）
    # -----------------------------------------------------------
    PRETRAINED_MODEL_PATH = "/content/deepmirbp_best_model.keras"

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"\n🔄 检测到已有模型，从 {PRETRAINED_MODEL_PATH} 加载并继续训练 ...")
        model = tf.keras.models.load_model(
            PRETRAINED_MODEL_PATH,
            custom_objects={'Attention': Attention}
        )
        # 重新编译以确保优化器状态正确
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=CONFIG['learning_rate']),
            metrics=['accuracy']
        )
        # 构建对应的 embedding 模型（提取 Attention 层输出）
        attention_layer_output = model.get_layer('attention').output
        embedding_model = Model(
            inputs=model.input,
            outputs=attention_layer_output,
            name='DeepmiRBP_Embedding'
        )
        print("✅ 已加载预训练模型，将在此基础上继续训练")
    else:
        print(f"\n🏗️ 未找到预训练模型 ({PRETRAINED_MODEL_PATH})，从头构建 DeepmiRBP 模型 ...")
        model, embedding_model = build_deepmirbp_model(
            max_length=max_length,
            num_unique_chars=num_unique,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=CONFIG['lstm_units'],
            num_lstm_layers=CONFIG['num_lstm_layers'],
            dropout_rate=CONFIG['dropout_rate'],
            learning_rate=CONFIG['learning_rate']
        )

    model.summary()

    # -----------------------------------------------------------
    # 9.5 设置 Callbacks（保存到 Google Drive）
    # -----------------------------------------------------------
    drive_save_dir = "/content/drive/MyDrive/Claude"
    os.makedirs(drive_save_dir, exist_ok=True)
    model_save_path = os.path.join(drive_save_dir, "deepmirbp_best_model.keras")
    # 同时在本地保存一份，方便下次继续训练
    local_save_path = "/content/deepmirbp_best_model.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss',
                        save_best_only=True, verbose=1),
        ModelCheckpoint(filepath=local_save_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    # -----------------------------------------------------------
    # 9.6 训练
    # -----------------------------------------------------------
    print("\n🏋️ 开始训练 ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # -----------------------------------------------------------
    # 9.7 评估
    # -----------------------------------------------------------
    train_results = evaluate_model(model, X_train, y_train, "训练集")
    test_results  = evaluate_model(model, X_test, y_test, "测试集")

    # -----------------------------------------------------------
    # 9.8 可视化
    # -----------------------------------------------------------
    plot_training_history(history, title='DeepmiRBP')
    plot_confusion_matrix(y_test, test_results['y_pred'],
                          title='Test Set Confusion Matrix')
    plot_roc_curve(y_test, test_results['y_prob'],
                   title='Test Set ROC Curve')

    # -----------------------------------------------------------
    # 9.9 miRNA 相似度分析（可选）
    # -----------------------------------------------------------
    if os.path.exists(MIRNA_PATH):
        print("\n🧬 正在进行 miRNA 相似度分析 ...")
        try:
            mirna_seqs, mirna_labels, mirna_chars = read_and_process_data(MIRNA_PATH)
            if len(mirna_seqs) > 0:
                padded_mirna = pad_sequences(mirna_seqs, maxlen=max_length,
                                             padding='post', truncating='post')
                mirna_embedding = embedding_model.predict(padded_mirna, verbose=0)

                rna_embeddings = get_embeddings(rbp_data_dict, embedding_model,
                                                max_length, sample_size=500)

                similarity = compute_mirna_similarity(rna_embeddings,
                                                      mirna_embedding, top_k=20)
        except Exception as e:
            print(f"  ⚠️ miRNA 分析出错: {e}")

    # -----------------------------------------------------------
    # 9.10 PSSM 分析（可选）
    # -----------------------------------------------------------
    if os.path.isdir(PSSM_DIR) and len(os.listdir(PSSM_DIR)) > 0:
        print("\n📐 正在处理 PSSM 数据 ...")
        try:
            processor = PSSMProcessor(PSSM_DIR)
            pssm_dict = processor.load_and_adjust_pssms()
            if pssm_dict:
                pssm_pca_dict = reduce_pssm_with_pca(pssm_dict, n_components=25)
                print(f"  PSSM PCA 降维完成: {len(pssm_pca_dict)} 个蛋白质")
        except Exception as e:
            print(f"  ⚠️ PSSM 处理出错: {e}")

    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"   最佳模型已保存至 Google Drive: {model_save_path}")
    print(f"   本地副本: {local_save_path}（下次运行将自动加载继续训练）")
    print("=" * 60)

    return model, embedding_model, history, test_results


# ╔══════════════════════════════════════════════════════════════════╗
# ║  运行                                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    model, embedding_model, history, results = main()
