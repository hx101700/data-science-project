import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 主题设置：统一风格与调色板
sns.set_theme(style="whitegrid")
sns.set_context("talk")
default_palette = sns.color_palette("Set2", n_colors=8)


def plot_scatter_matrix(df: pd.DataFrame, elements: list[str], label_col: str = 'Label') -> plt.Figure:
    """
    绘制成对散点矩阵 (Pair-wise Scatter-Matrix)

    Args:
        df: 包含 CLR 后特征和标签的 DataFrame
        elements: 地球化学元素列表（不含 'clr_' 前缀）
        label_col: 类别标签列名
    Returns:
        matplotlib.figure.Figure
    """
    clr_cols = [f"clr_{el}" for el in elements]
    hue_levels = sorted(df[label_col].unique())
    palette = default_palette[:len(hue_levels)]  # 截取前 N 色
    pairgrid = sns.pairplot(
        df, vars=clr_cols, hue=label_col, hue_order=hue_levels, palette=palette,
        corner=True, plot_kws={'alpha':0.7, 's':40, 'edgecolor':'w', 'linewidth':0.5}
    )
    pairgrid.figure.suptitle("Pairwise Scatter Matrix (CLR-transformed)", y=1.03, fontsize=18)
    plt.close(pairgrid.figure)
    return pairgrid.figure


def plot_correlation_heatmap(df: pd.DataFrame, elements: list[str], method: str = 'pearson') -> plt.Figure:
    """
    绘制相关性热图 (Correlation Heatmap)

    Args:
        df: 包含 CLR 后特征的 DataFrame
        elements: 地球化学元素列表
        method: 'pearson' 或 'spearman'
    Returns:
        matplotlib.figure.Figure
    """
    clr_cols = [f"clr_{el}" for el in elements]
    corr = df[clr_cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        square=True, cbar_kws={'shrink':0.8, 'label': f'{method.title()} Correlation'},
        ax=ax
    )
    ax.set_title(f"{method.title()} Correlation Heatmap", fontsize=20)
    plt.close(fig)
    return fig


def plot_pca_biplot(df: pd.DataFrame, elements: list[str], label_col: str = 'Label') -> plt.Figure:
    """
    绘制 PCA 双变量散点图 (PC1 vs PC2)

    Args:
        df: 包含 CLR 后特征和标签的 DataFrame
        elements: 用于 PCA 的元素列表
        label_col: 类别标签列名
    Returns:
        matplotlib.figure.Figure
    """
    # 提取 CLR 列和标签
    clr_cols = [f"clr_{el}" for el in elements]
    X = df[clr_cols].values
    y = df[label_col].values

    # 运行 PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    df_pca[label_col] = y

    # 动态获取类别及配色
    hue_levels = sorted(df_pca[label_col].unique())
    palette = default_palette[:len(hue_levels)]
    # 或者用： palette = sns.color_palette("Set2", n_colors=len(hue_levels))

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_pca,
        x='PC1', y='PC2',
        hue=label_col, hue_order=hue_levels, palette=palette,
        s=60, alpha=0.8, edgecolor='k', linewidth=0.5, ax=ax
    )

    # 英文标题与坐标标签
    ax.set_title("PCA Bi-plot (PC1 vs PC2)", fontsize=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=14)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=14)
    ax.legend(title=label_col, loc='best')

    plt.close(fig)
    return fig


def plot_ratio_diagrams(df: pd.DataFrame, ratios: list[tuple[str,str]], against: str) -> plt.Figure:
    """
    绘制地球化学比值图 (Geochemical Ratio Diagrams)

    Args:
        df: 含原始或 CLR 后数据的 DataFrame
        ratios: 比值列表，形如 [(num1, den1), (num2, den2)]
        against: y 轴元素列名
    Returns:
        matplotlib.figure.Figure
    """
    n = len(ratios)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)
    for ax, (num, den) in zip(axes, ratios):
        ratio = df[num] / df[den]
        sns.scatterplot(
            x=ratio, y=df[against], hue=df['Label'], palette=default_palette,
            s=50, alpha=0.7, edgecolor='w', linewidth=0.5, ax=ax
        )
        ax.set_title(f"{num}/{den} vs {against}", fontsize=16)
        ax.set_xlabel(f"{num}/{den}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel(against, fontsize=14)
    fig.tight_layout()
    plt.close(fig)
    return fig
