"""
dataset_drought.py
干旱分级数据集加载器，支持 ENVI 格式（.dat + .dat.enp）。

数据通道说明：
  RGB (3通道)       : 可见光前3个band
  热红外 (3通道)     : 热红外前3个band
  多光谱 (8通道)     :
      多光谱5通道: NIR, Red, Blue, Green, RedEdge（各取第1个band，即第0个band）
      植被指数3通道: NDVI, GNDVI, SAVI
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import spectral.io.envi as envi
except ImportError:
    raise ImportError("请安装 spectral 库: pip install spectral")


# ---------------------------------------------------------------------------
# 植被指数计算
# ---------------------------------------------------------------------------

def _safe_divide(num, denom, eps=1e-8):
    """防止除零的安全除法。"""
    return num / (denom + eps)


def compute_ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red)"""
    return _safe_divide(nir - red, nir + red)


def compute_gndvi(nir, green):
    """GNDVI = (NIR - Green) / (NIR + Green)"""
    return _safe_divide(nir - green, nir + green)


def compute_savi(nir, red, L=0.5):
    """SAVI = (1 + L) × (NIR - Red) / (NIR + Red + L)  (L=0.5 by default)"""
    return (1 + L) * _safe_divide(nir - red, nir + red + L)


# ---------------------------------------------------------------------------
# ENVI 读取辅助函数
# ---------------------------------------------------------------------------

def read_envi_band(dat_path, band_idx=0):
    """
    读取 ENVI 格式文件中的指定 band，返回 float32 的 2D numpy 数组。

    Args:
        dat_path (str): .dat 文件路径（头文件应为同名 .dat.enp）
        band_idx (int): 要读取的 band 索引（0-based）

    Returns:
        np.ndarray: shape (H, W), dtype float32
    """
    hdr_path = dat_path + '.enp'
    img = envi.open(hdr_path, dat_path)
    # img[:, :, band_idx] 返回 (H, W, 1) 或 (H, W)
    band = img[:, :, band_idx]
    if band.ndim == 3:
        band = band[:, :, 0]
    return band.astype(np.float32)


def read_envi_bands(dat_path, band_indices):
    """
    批量读取 ENVI 格式文件中的多个 band，返回 float32 的 3D numpy 数组。

    Args:
        dat_path (str): .dat 文件路径
        band_indices (list[int]): 要读取的 band 索引列表

    Returns:
        np.ndarray: shape (len(band_indices), H, W), dtype float32
    """
    hdr_path = dat_path + '.enp'
    img = envi.open(hdr_path, dat_path)
    bands = []
    for idx in band_indices:
        band = img[:, :, idx]
        if band.ndim == 3:
            band = band[:, :, 0]
        bands.append(band.astype(np.float32))
    return np.stack(bands, axis=0)  # (C, H, W)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def percentile_normalize(arr, low=2, high=98):
    """
    百分位数归一化，将数值映射到 [0, 1]。
    对每个通道独立处理。

    Args:
        arr (np.ndarray): shape (C, H, W)

    Returns:
        np.ndarray: shape (C, H, W), values in [0, 1]
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        lo = np.percentile(arr[c], low)
        hi = np.percentile(arr[c], high)
        if hi - lo < 1e-8:
            out[c] = np.zeros_like(arr[c])
        else:
            out[c] = np.clip((arr[c] - lo) / (hi - lo), 0.0, 1.0)
    return out


def minmax_normalize(arr):
    """
    Min-Max 归一化，将数值映射到 [0, 1]。
    对每个通道独立处理。
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        lo = arr[c].min()
        hi = arr[c].max()
        if hi - lo < 1e-8:
            out[c] = np.zeros_like(arr[c])
        else:
            out[c] = (arr[c] - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# ID → 文件名映射
# ---------------------------------------------------------------------------

def get_file_paths(sample_id, data_root):
    """
    根据样本 ID 返回各模态文件路径字典。

    文件名规则（按问题描述）：
      ID=1:
        RGB:     {data_root}/_0519_rgb_control/_0519_20m_kejianguang_1.dat
        TIR:     {data_root}/_0519_rehongwai_control/_0519_rehongwai_20m_1.dat
        NIR:     {data_root}/_0519_nir_control/_0519_duoguangpu_20m_840_1.dat
        Red:     {data_root}/_0519_red_control/_0519_duoguangpu_20m_660_1.dat
        Green:   {data_root}/_0519_green_control/_0519_duoguangpu_20m_555_1.dat
        Blue:    {data_root}/_0519_blue_control/_0519_duoguangpu_20m_450_1.dat
        RedEdge: {data_root}/_0519_rededge_control/_0519_duoguangpu_20m_720_1.dat

    Args:
        sample_id (int): 样本编号（从1开始）
        data_root (str): 数据根目录

    Returns:
        dict: 各模态文件路径
    """
    sid = int(sample_id)
    return {
        'rgb': os.path.join(
            data_root, '_0519_rgb_control',
            f'_0519_20m_kejianguang_{sid}.dat'
        ),
        'tir': os.path.join(
            data_root, '_0519_rehongwai_control',
            f'_0519_rehongwai_20m_{sid}.dat'
        ),
        'nir': os.path.join(
            data_root, '_0519_nir_control',
            f'_0519_duoguangpu_20m_840_{sid}.dat'
        ),
        'red': os.path.join(
            data_root, '_0519_red_control',
            f'_0519_duoguangpu_20m_660_{sid}.dat'
        ),
        'green': os.path.join(
            data_root, '_0519_green_control',
            f'_0519_duoguangpu_20m_555_{sid}.dat'
        ),
        'blue': os.path.join(
            data_root, '_0519_blue_control',
            f'_0519_duoguangpu_20m_450_{sid}.dat'
        ),
        'rededge': os.path.join(
            data_root, '_0519_rededge_control',
            f'_0519_duoguangpu_20m_720_{sid}.dat'
        ),
    }


# ---------------------------------------------------------------------------
# Dataset 类
# ---------------------------------------------------------------------------

class DroughtDataset(Dataset):
    """
    干旱分级数据集。

    每个样本包含三组张量：
      rgb (3, H, W)  : 可见光前3个band（归一化到[0,1]）
      tir (3, H, W)  : 热红外前3个band（归一化到[0,1]）
      ms  (8, H, W)  : 多光谱5通道 + 植被指数3通道（归一化到[0,1]）
    以及对应的标签（0-4）。

    Args:
        csv_path (str): 标签 CSV 文件路径，需包含 'ID' 和 'label' 列。
        data_root (str): 数据根目录。
        ids (list[int]): 要使用的样本 ID 列表。
        augment (bool): 是否进行数据增强（翻转）。
        normalize_method (str): 归一化方法，'percentile' 或 'minmax'。
    """

    def __init__(self, csv_path, data_root, ids, augment=False,
                 normalize_method='percentile'):
        self.data_root = data_root
        self.augment = augment
        self.normalize_method = normalize_method

        df = pd.read_csv(csv_path)
        df = df[df['ID'].isin(ids)].reset_index(drop=True)
        self.ids = df['ID'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.ids)

    def _normalize(self, arr):
        if self.normalize_method == 'percentile':
            return percentile_normalize(arr)
        return minmax_normalize(arr)

    def _augment(self, *arrays):
        """随机水平/垂直翻转，对所有数组执行相同操作。"""
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-1).copy() for a in arrays)
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-2).copy() for a in arrays)
        return arrays

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[idx]
        paths = get_file_paths(sample_id, self.data_root)

        # RGB: 前3个band
        rgb = read_envi_bands(paths['rgb'], [0, 1, 2])           # (3, H, W)

        # 热红外: 前3个band
        tir = read_envi_bands(paths['tir'], [0, 1, 2])           # (3, H, W)

        # 多光谱: 每个波段文件只取第0个band
        nir_arr     = read_envi_band(paths['nir'],     band_idx=0)  # (H, W)
        red_arr     = read_envi_band(paths['red'],     band_idx=0)
        green_arr   = read_envi_band(paths['green'],   band_idx=0)
        blue_arr    = read_envi_band(paths['blue'],    band_idx=0)
        rededge_arr = read_envi_band(paths['rededge'], band_idx=0)

        # 植被指数
        ndvi  = compute_ndvi(nir_arr, red_arr)
        gndvi = compute_gndvi(nir_arr, green_arr)
        savi  = compute_savi(nir_arr, red_arr)

        # 多光谱 + VI: shape (8, H, W)
        ms = np.stack([nir_arr, red_arr, blue_arr, green_arr, rededge_arr,
                       ndvi, gndvi, savi], axis=0)

        # 归一化
        rgb = self._normalize(rgb)
        tir = self._normalize(tir)
        ms  = self._normalize(ms)

        # 数据增强
        if self.augment:
            rgb, tir, ms = self._augment(rgb, tir, ms)

        return (
            torch.from_numpy(rgb).float(),
            torch.from_numpy(tir).float(),
            torch.from_numpy(ms).float(),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# 数据集构建函数
# ---------------------------------------------------------------------------

def build_datasets(csv_path, data_root, test_size=0.2, random_state=42,
                   augment_train=True, normalize_method='percentile'):
    """
    从 CSV 文件构建训练集和验证集，使用分层采样。

    Args:
        csv_path (str): 标签 CSV 文件路径（需含 'ID' 和 'label' 列）。
        data_root (str): 数据根目录。
        test_size (float): 验证集比例（默认 0.2）。
        random_state (int): 随机种子（默认 42）。
        augment_train (bool): 训练集是否使用数据增强。
        normalize_method (str): 归一化方法。

    Returns:
        train_dataset, val_dataset
    """
    df = pd.read_csv(csv_path)
    ids = df['ID'].tolist()
    labels = df['label'].tolist()

    train_ids, val_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = DroughtDataset(
        csv_path, data_root, train_ids,
        augment=augment_train,
        normalize_method=normalize_method,
    )
    val_dataset = DroughtDataset(
        csv_path, data_root, val_ids,
        augment=False,
        normalize_method=normalize_method,
    )
    return train_dataset, val_dataset


def build_dataloaders(csv_path, data_root, batch_size=8, num_workers=4,
                      test_size=0.2, random_state=42,
                      augment_train=True, normalize_method='percentile'):
    """
    构建训练/验证 DataLoader。

    Returns:
        train_loader, val_loader
    """
    train_ds, val_ds = build_datasets(
        csv_path, data_root,
        test_size=test_size,
        random_state=random_state,
        augment_train=augment_train,
        normalize_method=normalize_method,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
