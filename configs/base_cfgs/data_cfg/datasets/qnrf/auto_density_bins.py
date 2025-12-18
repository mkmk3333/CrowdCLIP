"""
自动确定QNRF数据集的密度区间
支持多种策略：分位数、K-means聚类、均匀分布等
"""
import os
import numpy as np
import scipy.io as io
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import json


class DensityBinCalculator:
    """
    自动计算最优的密度区间
    """
    def __init__(self, num_bins=6):
        self.num_bins = num_bins
        self.density_values = []

    def collect_density_from_dataset(self, root_path='./datasets/UCF-QNRF', split='Train'):
        """
        从数据集中收集所有图像的密度信息
        """
        img_path = os.path.join(root_path, split)
        img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

        all_counts = []
        all_patch_counts = []  # 存储所有可能的patch密度

        print(f"Collecting density information from {len(img_files)} images...")

        for idx, img_file in enumerate(img_files):
            if idx % 50 == 0:
                print(f"Processing {idx}/{len(img_files)}...")

            # 读取图像
            img_full_path = os.path.join(img_path, img_file)
            img_data = cv2.imread(img_full_path)

            # 计算缩放比例（与preprocess_qnrf.py保持一致）
            rate = 1
            if img_data.shape[1] > img_data.shape[0] and img_data.shape[1] >= 2048:
                rate = 2048.0 / img_data.shape[1]
            if img_data.shape[0] > img_data.shape[1] and img_data.shape[0] >= 2048:
                rate = 2048.0 / img_data.shape[0]

            img_data = cv2.resize(img_data, (0, 0), fx=rate, fy=rate)

            # 读取标注
            fname = os.path.basename(img_file).split('.')[0]
            mat_path = os.path.join(img_path, fname + '_ann.mat')
            mat = io.loadmat(mat_path)
            gt_data = mat['annPoints'] * rate

            # 创建密度图
            kpoint = np.zeros((img_data.shape[0], img_data.shape[1]))
            for i in range(len(gt_data)):
                if int(gt_data[i][1]) < img_data.shape[0] and int(gt_data[i][0]) < img_data.shape[1]:
                    kpoint[int(gt_data[i][1]), int(gt_data[i][0])] = 1

            total_count = int(np.sum(kpoint))
            all_counts.append(total_count)

            # 模拟训练时的patch切分，收集各个patch的密度
            height, width = img_data.shape[0], img_data.shape[1]
            center_y, center_x = height // 2, width // 2
            item_height, item_width = height // 12, width // 12

            for i in range(6):  # 6个不同尺度的patch
                y1 = center_y - (i+1) * item_height
                y2 = center_y + (i+1) * item_height
                x1 = center_x - (i+1) * item_width
                x2 = center_x + (i+1) * item_width

                # 确保坐标在有效范围内
                y1, y2 = max(0, y1), min(height, y2)
                x1, x2 = max(0, x1), min(width, x2)

                patch_count = int(np.sum(kpoint[y1:y2, x1:x2]))
                all_patch_counts.append(patch_count)

        self.all_counts = np.array(all_counts)
        self.all_patch_counts = np.array(all_patch_counts)

        print(f"\nDataset statistics:")
        print(f"Total images: {len(all_counts)}")
        print(f"Total patches: {len(all_patch_counts)}")
        print(f"Count range: [{np.min(all_counts)}, {np.max(all_counts)}]")
        print(f"Patch count range: [{np.min(all_patch_counts)}, {np.max(all_patch_counts)}]")
        print(f"Mean count: {np.mean(all_counts):.2f}")
        print(f"Mean patch count: {np.mean(all_patch_counts):.2f}")

        return all_counts, all_patch_counts

    def calculate_bins_quantile(self, use_patch_counts=True):
        """
        方法1：使用分位数方法划分密度区间
        这是最稳定和最常用的方法
        """
        data = self.all_patch_counts if use_patch_counts else self.all_counts

        # 计算六分位数
        percentiles = np.linspace(0, 100, self.num_bins + 1)[1:-1]  # 去掉0和100
        bins = np.percentile(data, percentiles)

        # 使用每个区间的中位数作为代表值
        bin_representatives = []
        data_sorted = np.sort(data)

        # 第一个bin：从最小值到第一个分位点
        mask = data <= bins[0]
        bin_representatives.append(int(np.median(data[mask])))

        # 中间的bins
        for i in range(len(bins) - 1):
            mask = (data > bins[i]) & (data <= bins[i+1])
            bin_representatives.append(int(np.median(data[mask])))

        # 最后一个bin：从最后一个分位点到最大值
        mask = data > bins[-1]
        bin_representatives.append(int(np.median(data[mask])))

        print(f"\n=== Quantile-based bins ===")
        print(f"Bin edges: {[int(b) for b in bins]}")
        print(f"Bin representatives: {bin_representatives}")

        return bin_representatives

    def calculate_bins_kmeans(self, use_patch_counts=True):
        """
        方法2：使用K-means聚类确定密度区间
        这个方法可以找到数据的自然聚类
        """
        data = self.all_patch_counts if use_patch_counts else self.all_counts

        # 进行K-means聚类
        kmeans = KMeans(n_clusters=self.num_bins, random_state=42, n_init=10)
        kmeans.fit(data.reshape(-1, 1))

        # 聚类中心即为代表值
        centers = sorted([int(c[0]) for c in kmeans.cluster_centers_])

        print(f"\n=== K-means clustering bins ===")
        print(f"Cluster centers: {centers}")

        # 统计每个聚类的样本数量
        labels = kmeans.labels_
        for i, center in enumerate(centers):
            count = np.sum(labels == i)
            print(f"Cluster {i} (center={center}): {count} samples ({100*count/len(data):.1f}%)")

        return centers

    def calculate_bins_uniform(self, use_patch_counts=True):
        """
        方法3：均匀分布的密度区间
        简单但可能不符合数据的实际分布
        """
        data = self.all_patch_counts if use_patch_counts else self.all_counts

        min_val, max_val = np.min(data), np.max(data)
        bins = np.linspace(min_val, max_val, self.num_bins + 1)

        # 使用区间中点作为代表值
        bin_representatives = [int((bins[i] + bins[i+1]) / 2) for i in range(self.num_bins)]

        print(f"\n=== Uniform bins ===")
        print(f"Bin edges: {[int(b) for b in bins]}")
        print(f"Bin representatives: {bin_representatives}")

        return bin_representatives

    def calculate_bins_adaptive(self, use_patch_counts=True, min_samples_per_bin=50):
        """
        方法4：自适应方法 - 确保每个bin有足够的样本
        """
        data = self.all_patch_counts if use_patch_counts else self.all_counts
        data_sorted = np.sort(data)

        samples_per_bin = max(min_samples_per_bin, len(data) // self.num_bins)

        bin_representatives = []
        for i in range(self.num_bins):
            start_idx = i * samples_per_bin
            end_idx = min((i + 1) * samples_per_bin, len(data_sorted))
            bin_data = data_sorted[start_idx:end_idx]
            bin_representatives.append(int(np.median(bin_data)))

        print(f"\n=== Adaptive bins (min {min_samples_per_bin} samples per bin) ===")
        print(f"Bin representatives: {bin_representatives}")

        return bin_representatives

    def visualize_distribution(self, bins_dict, save_path='./density_distribution.png'):
        """
        可视化不同方法得到的密度分布和划分
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 绘制整体密度分布
        axes[0].hist(self.all_patch_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Crowd Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Patch Crowd Counts')
        axes[0].grid(True, alpha=0.3)

        # 为每种方法绘制分界线
        colors = ['red', 'green', 'purple', 'orange']
        for idx, (method, bins) in enumerate(bins_dict.items()):
            for bin_val in bins:
                axes[0].axvline(x=bin_val, color=colors[idx % len(colors)],
                              linestyle='--', alpha=0.6, linewidth=1.5,
                              label=f'{method}: {bin_val}' if bin_val == bins[0] else '')

        axes[0].legend()

        # 绘制比较图
        x_pos = np.arange(self.num_bins)
        width = 0.2
        for idx, (method, bins) in enumerate(bins_dict.items()):
            axes[1].bar(x_pos + idx * width, bins, width,
                       label=method, alpha=0.8, color=colors[idx % len(colors)])

        axes[1].set_xlabel('Bin Index')
        axes[1].set_ylabel('Representative Count Value')
        axes[1].set_title('Comparison of Different Binning Methods')
        axes[1].set_xticks(x_pos + width * (len(bins_dict) - 1) / 2)
        axes[1].set_xticklabels([f'Bin {i}' for i in range(self.num_bins)])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

    def save_bins(self, bins, method_name, save_path='./density_bins_config.json'):
        """
        保存密度区间配置到JSON文件
        """
        config = {
            'method': method_name,
            'num_bins': self.num_bins,
            'bins': bins,
            'crowd_count': [str(b) for b in bins],  # 转换为字符串格式，与原代码保持一致
            'statistics': {
                'dataset_mean': float(np.mean(self.all_counts)),
                'dataset_std': float(np.std(self.all_counts)),
                'patch_mean': float(np.mean(self.all_patch_counts)),
                'patch_std': float(np.std(self.all_patch_counts)),
                'min_count': int(np.min(self.all_patch_counts)),
                'max_count': int(np.max(self.all_patch_counts)),
            }
        }

        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguration saved to {save_path}")
        print(f"You can use these values in crowdclip.py:")
        print(f"crowd_count = {config['crowd_count']}")


def main():
    """
    主函数：计算并比较所有方法
    """
    calculator = DensityBinCalculator(num_bins=6)

    # 收集数据集的密度信息
    print("="*60)
    print("Step 1: Collecting density information from QNRF dataset")
    print("="*60)
    calculator.collect_density_from_dataset(
        root_path='./datasets/UCF-QNRF',
        split='Train'
    )

    # 计算不同方法的bins
    print("\n" + "="*60)
    print("Step 2: Calculating density bins using different methods")
    print("="*60)

    bins_quantile = calculator.calculate_bins_quantile(use_patch_counts=True)
    bins_kmeans = calculator.calculate_bins_kmeans(use_patch_counts=True)
    bins_uniform = calculator.calculate_bins_uniform(use_patch_counts=True)
    bins_adaptive = calculator.calculate_bins_adaptive(use_patch_counts=True)

    # 可视化比较
    print("\n" + "="*60)
    print("Step 3: Visualizing and saving results")
    print("="*60)

    bins_dict = {
        'Quantile': bins_quantile,
        'K-means': bins_kmeans,
        'Uniform': bins_uniform,
        'Adaptive': bins_adaptive,
    }

    calculator.visualize_distribution(
        bins_dict,
        save_path='./processed_datasets/UCF-QNRF/density_distribution_comparison.png'
    )

    # 保存推荐的配置（默认使用quantile方法）
    print("\n" + "="*60)
    print("Step 4: Saving recommended configuration")
    print("="*60)
    print("\nRecommendation: Quantile method is recommended for balanced coverage")

    calculator.save_bins(
        bins_quantile,
        'quantile',
        save_path='./processed_datasets/UCF-QNRF/density_bins_config.json'
    )

    # 对比原始的固定值
    print("\n" + "="*60)
    print("Comparison with original fixed values")
    print("="*60)
    original_bins = [20, 55, 90, 125, 160, 195]
    print(f"Original fixed bins: {original_bins}")
    print(f"New quantile bins:   {bins_quantile}")
    print(f"New K-means bins:    {bins_kmeans}")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
