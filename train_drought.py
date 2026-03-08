"""
train_drought.py
干旱分级五分类训练脚本。

用法:
    python train_drought.py

配置:
    csv_path  = '/home/zcl/addfuse/2025label_classic5.csv'
    data_root = '/home/zcl/addfuse/dataset/'
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from net_drought import DroughtClassifier
from dataset_drought import build_dataloaders

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CSV_PATH   = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT  = '/home/zcl/addfuse/dataset/'

BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 1e-4
NUM_CLASSES = 5
NUM_WORKERS = 4
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# 模型保存目录
SAVE_DIR = './models'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_best.pth')

# 学习率调度
LR_STEP_SIZE = 30
LR_GAMMA     = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 训练 / 验证函数
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb    = rgb.to(device)
        tir    = tir.to(device)
        ms     = ms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb, tir, ms)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds    = logits.argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'acc': f'{correct / total * 100:.2f}%'})

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Validation', leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb    = rgb.to(device)
        tir    = tir.to(device)
        ms     = ms.to(device)
        labels = labels.to(device)

        logits = model(rgb, tir, ms)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def main():
    print(f'Device: {DEVICE}')
    print(f'CSV:    {CSV_PATH}')
    print(f'Data:   {DATA_ROOT}')

    # 数据
    train_loader, val_loader = build_dataloaders(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        augment_train=True,
    )
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # 模型
    model = DroughtClassifier(
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f'\nEpoch {epoch}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        elapsed = time.time() - t0
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%  '
              f'[{elapsed:.1f}s]')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, BEST_MODEL_PATH)
            print(f'✅ Best model saved! Val Acc: {val_acc:.2f}%')

    print(f'\nTraining complete. Best Val Acc: {best_val_acc:.2f}%')
    print(f'Best model saved to: {BEST_MODEL_PATH}')


if __name__ == '__main__':
    main()
