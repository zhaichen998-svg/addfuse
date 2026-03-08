"""
test_drought.py
干旱分级五分类测试脚本。

加载最佳模型，在验证集上评估并输出：
  - 总体准确率
  - 每类准确率
  - 混淆矩阵
  - F1-score（每类 + 宏平均）
  - 混淆矩阵可视化（保存为 confusion_matrix.png）

用法:
    python test_drought.py
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from net_drought import DroughtClassifier
from dataset_drought import DroughtDataset

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CSV_PATH   = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT  = '/home/zcl/addfuse/dataset/'

BEST_MODEL_PATH = './models/drought_best.pth'
OUTPUT_DIR      = './test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE  = 8
NUM_CLASSES = 5
NUM_WORKERS = 4
RANDOM_SEED = 42
TEST_SIZE   = 0.2

CLASS_NAMES = [f'Level {i}' for i in range(NUM_CLASSES)]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 获取验证集 IDs
# ---------------------------------------------------------------------------

def get_val_ids(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    ids    = df['ID'].tolist()
    labels = df['label'].tolist()
    _, val_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return val_ids


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """运行推理，返回所有预测和真实标签。"""
    model.eval()
    all_preds  = []
    all_labels = []

    for rgb, tir, ms, labels in tqdm(loader, desc='Testing'):
        rgb    = rgb.to(device)
        tir    = tir.to(device)
        ms     = ms.to(device)
        logits = model(rgb, tir, ms)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# 可视化混淆矩阵
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Drought Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Confusion matrix saved to: {save_path}')


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print(f'Device:      {DEVICE}')
    print(f'Model path:  {BEST_MODEL_PATH}')

    # 验证集
    val_ids = get_val_ids(CSV_PATH, TEST_SIZE, RANDOM_SEED)
    val_dataset = DroughtDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        ids=val_ids,
        augment=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f'Validation samples: {len(val_dataset)}')

    # 模型
    model = DroughtClassifier(
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(val_acc={checkpoint.get('val_acc', 0):.2f}%)")

    # 推理
    preds, labels = run_inference(model, val_loader, DEVICE)

    # 指标
    overall_acc = accuracy_score(labels, preds) * 100
    print(f'\n=== Overall Accuracy: {overall_acc:.2f}% ===\n')

    # 每类准确率
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    print('Per-class Accuracy:')
    for i, acc in enumerate(per_class_acc):
        print(f'  {CLASS_NAMES[i]}: {acc:.2f}%')

    # F1-score
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    f1_macro     = f1_score(labels, preds, average='macro', zero_division=0)
    print('\nPer-class F1-score:')
    for i, f1 in enumerate(f1_per_class):
        print(f'  {CLASS_NAMES[i]}: {f1:.4f}')
    print(f'Macro F1-score: {f1_macro:.4f}')

    # 详细分类报告
    print('\nClassification Report:')
    print(classification_report(labels, preds, target_names=CLASS_NAMES,
                                 zero_division=0))

    # 保存指标到 CSV
    metrics_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Accuracy (%)': per_class_acc.round(2),
        'F1-score': f1_per_class.round(4),
    })
    metrics_df.loc[len(metrics_df)] = ['Overall', round(overall_acc, 2), round(f1_macro, 4)]
    metrics_csv_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f'\nMetrics saved to: {metrics_csv_path}')

    # 可视化混淆矩阵
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)


if __name__ == '__main__':
    main()
