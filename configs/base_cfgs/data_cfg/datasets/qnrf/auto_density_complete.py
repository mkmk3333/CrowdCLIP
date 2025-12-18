"""
完整的自适应密度区间解决方案
同时解决两个问题：
1. 确定最优的区间数量（How many bins?）
2. 确定每个区间的代表值（What are the bin values?）

这是一个端到端的解决方案，一次运行即可完成所有分析和配置
"""
import os
import numpy as np
import scipy.io as io
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class CompleteDensityBinSolution:
    """
    完整的密度区间解决方案
    从数据分析到最终配置的一站式解决方案
    """
    def __init__(self, min_bins=3, max_bins=12, method='auto'):
        """
        Args:
            min_bins: 最小区间数
            max_bins: 最大区间数
            method: 确定最优k的方法
                - 'auto': 自动综合多种方法（推荐）
                - 'silhouette': 使用轮廓系数
                - 'bic': 使用BIC准则
                - 'elbow': 使用肘部法则
                - 'ch_index': 使用Calinski-Harabasz指数
        """
        self.min_bins = max(3, min_bins)
        self.max_bins = min(20, max_bins)
        self.method = method
        self.all_patch_counts = None
        self.optimal_k = None
        self.density_bins = None
        self.evaluation_results = {}

    def collect_density_data(self, root_path='./datasets/UCF-QNRF', split='Train'):
        """
        收集数据集的密度信息
        """
        print("="*70)
        print("STEP 1: Collecting Density Data from QNRF Dataset")
        print("="*70)

        img_path = os.path.join(root_path, split)
        img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

        all_patch_counts = []

        print(f"Processing {len(img_files)} images...")

        for idx, img_file in enumerate(img_files):
            if idx % 50 == 0:
                print(f"  [{idx}/{len(img_files)}] {img_file}")

            # 读取和缩放图像
            img_full_path = os.path.join(img_path, img_file)
            img_data = cv2.imread(img_full_path)

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

            # 多种patch采样策略
            height, width = img_data.shape[0], img_data.shape[1]
            center_y, center_x = height // 2, width // 2
            item_height, item_width = height // 12, width // 12

            # 1. 中心多尺度采样（模拟训练切分）
            for i in range(6):
                y1 = max(0, center_y - (i+1) * item_height)
                y2 = min(height, center_y + (i+1) * item_height)
                x1 = max(0, center_x - (i+1) * item_width)
                x2 = min(width, center_x + (i+1) * item_width)
                patch_count = int(np.sum(kpoint[y1:y2, x1:x2]))
                all_patch_counts.append(patch_count)

            # 2. 随机patch采样（增加多样性）
            for _ in range(4):
                patch_size_h = np.random.randint(height // 4, height * 3 // 4)
                patch_size_w = np.random.randint(width // 4, width * 3 // 4)
                y1 = np.random.randint(0, max(1, height - patch_size_h))
                x1 = np.random.randint(0, max(1, width - patch_size_w))
                y2 = min(height, y1 + patch_size_h)
                x2 = min(width, x1 + patch_size_w)
                patch_count = int(np.sum(kpoint[y1:y2, x1:x2]))
                all_patch_counts.append(patch_count)

        self.all_patch_counts = np.array(all_patch_counts)

        # 打印统计信息
        print(f"\nDataset Statistics:")
        print(f"  Total patch samples: {len(self.all_patch_counts)}")
        print(f"  Density range: [{np.min(self.all_patch_counts)}, {np.max(self.all_patch_counts)}]")
        print(f"  Mean: {np.mean(self.all_patch_counts):.2f}")
        print(f"  Median: {np.median(self.all_patch_counts):.2f}")
        print(f"  Std: {np.std(self.all_patch_counts):.2f}")
        print(f"  25th percentile: {np.percentile(self.all_patch_counts, 25):.2f}")
        print(f"  75th percentile: {np.percentile(self.all_patch_counts, 75):.2f}")

        return self.all_patch_counts

    def find_optimal_k(self):
        """
        自动找到最优的k值（区间数量）
        """
        print("\n" + "="*70)
        print("STEP 2: Finding Optimal Number of Bins (k)")
        print("="*70)

        data = self.all_patch_counts.reshape(-1, 1)
        k_range = range(self.min_bins, self.max_bins + 1)

        if self.method == 'auto':
            # 综合多种方法
            print("\nUsing AUTO mode - evaluating multiple criteria...\n")

            optimal_ks = []

            # 1. Silhouette Score（轮廓系数）
            print("1. Evaluating Silhouette Scores...")
            silhouette_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
                print(f"   k={k:2d}: Silhouette={score:.4f}")

            k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
            optimal_ks.append(k_silhouette)
            print(f"   → Best k by Silhouette: {k_silhouette}\n")

            self.evaluation_results['silhouette'] = {
                'k_range': list(k_range),
                'scores': silhouette_scores,
                'optimal_k': k_silhouette
            }

            # 2. BIC (Bayesian Information Criterion)
            print("2. Evaluating BIC (Gaussian Mixture Model)...")
            bic_scores = []
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
                gmm.fit(data)
                bic = gmm.bic(data)
                bic_scores.append(bic)
                print(f"   k={k:2d}: BIC={bic:.2f}")

            k_bic = list(k_range)[np.argmin(bic_scores)]
            optimal_ks.append(k_bic)
            print(f"   → Best k by BIC: {k_bic}\n")

            self.evaluation_results['bic'] = {
                'k_range': list(k_range),
                'scores': bic_scores,
                'optimal_k': k_bic
            }

            # 3. Calinski-Harabasz Index
            print("3. Evaluating Calinski-Harabasz Index...")
            ch_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = calinski_harabasz_score(data, labels)
                ch_scores.append(score)
                print(f"   k={k:2d}: CH Index={score:.2f}")

            k_ch = list(k_range)[np.argmax(ch_scores)]
            optimal_ks.append(k_ch)
            print(f"   → Best k by CH Index: {k_ch}\n")

            self.evaluation_results['ch_index'] = {
                'k_range': list(k_range),
                'scores': ch_scores,
                'optimal_k': k_ch
            }

            # 4. Elbow Method (二阶导数)
            print("4. Evaluating Elbow Method...")
            inertias = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
                print(f"   k={k:2d}: Inertia={kmeans.inertia_:.2f}")

            # 计算二阶导数
            inertias = np.array(inertias)
            if len(inertias) > 2:
                second_diff = np.diff(np.diff(inertias))
                k_elbow = list(k_range)[np.argmax(second_diff) + 1]
            else:
                k_elbow = self.min_bins
            optimal_ks.append(k_elbow)
            print(f"   → Best k by Elbow: {k_elbow}\n")

            self.evaluation_results['elbow'] = {
                'k_range': list(k_range),
                'inertias': inertias.tolist(),
                'optimal_k': k_elbow
            }

            # 综合决策：使用中位数
            print("-" * 70)
            print("Summary of all methods:")
            print(f"  Silhouette Score:     k = {k_silhouette}")
            print(f"  BIC (GMM):            k = {k_bic}")
            print(f"  CH Index:             k = {k_ch}")
            print(f"  Elbow Method:         k = {k_elbow}")
            print(f"\n  All recommendations: {optimal_ks}")

            # 使用中位数作为最终推荐
            self.optimal_k = int(np.median(optimal_ks))

            print(f"\n{'='*70}")
            print(f"CONSENSUS RECOMMENDATION: k = {self.optimal_k}")
            print(f"{'='*70}")

        elif self.method == 'silhouette':
            print("\nUsing SILHOUETTE method...\n")
            silhouette_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
                print(f"k={k:2d}: Silhouette Score={score:.4f}")

            self.optimal_k = list(k_range)[np.argmax(silhouette_scores)]
            print(f"\nOptimal k: {self.optimal_k}")

        elif self.method == 'bic':
            print("\nUsing BIC method...\n")
            bic_scores = []
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
                gmm.fit(data)
                bic = gmm.bic(data)
                bic_scores.append(bic)
                print(f"k={k:2d}: BIC={bic:.2f}")

            self.optimal_k = list(k_range)[np.argmin(bic_scores)]
            print(f"\nOptimal k: {self.optimal_k}")

        elif self.method == 'ch_index':
            print("\nUsing Calinski-Harabasz Index method...\n")
            ch_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = calinski_harabasz_score(data, labels)
                ch_scores.append(score)
                print(f"k={k:2d}: CH Index={score:.2f}")

            self.optimal_k = list(k_range)[np.argmax(ch_scores)]
            print(f"\nOptimal k: {self.optimal_k}")

        elif self.method == 'elbow':
            print("\nUsing Elbow method...\n")
            inertias = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
                print(f"k={k:2d}: Inertia={kmeans.inertia_:.2f}")

            inertias = np.array(inertias)
            if len(inertias) > 2:
                second_diff = np.diff(np.diff(inertias))
                self.optimal_k = list(k_range)[np.argmax(second_diff) + 1]
            else:
                self.optimal_k = self.min_bins
            print(f"\nOptimal k: {self.optimal_k}")

        return self.optimal_k

    def calculate_density_bins(self, strategy='kmeans'):
        """
        使用最优的k值计算密度区间的代表值

        Args:
            strategy: 计算策略
                - 'kmeans': 使用K-means聚类中心（推荐，与k的确定方法一致）
                - 'quantile': 使用分位数方法
                - 'hybrid': 混合方法
        """
        print("\n" + "="*70)
        print(f"STEP 3: Calculating Density Bins (k={self.optimal_k})")
        print("="*70)

        data = self.all_patch_counts.reshape(-1, 1)

        if strategy == 'kmeans':
            print(f"\nUsing K-MEANS clustering with k={self.optimal_k}...\n")

            # 使用K-means找到聚类中心
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=20)
            kmeans.fit(data)

            # 聚类中心即为密度代表值
            centers = sorted([int(c[0]) for c in kmeans.cluster_centers_])
            self.density_bins = centers

            # 统计每个聚类的样本数
            labels = kmeans.labels_
            print("Cluster Analysis:")
            for i, center in enumerate(centers):
                count = np.sum(labels == i)
                percentage = 100 * count / len(data)
                print(f"  Bin {i+1}: center={center:4d}, samples={count:5d} ({percentage:5.2f}%)")

        elif strategy == 'quantile':
            print(f"\nUsing QUANTILE method with k={self.optimal_k}...\n")

            # 计算分位数
            percentiles = np.linspace(0, 100, self.optimal_k + 1)[1:-1]
            bin_edges = np.percentile(data, percentiles)

            # 计算每个区间的中位数作为代表值
            bin_representatives = []
            data_sorted = np.sort(data.flatten())

            # 第一个bin
            mask = data.flatten() <= bin_edges[0]
            bin_representatives.append(int(np.median(data.flatten()[mask])))

            # 中间的bins
            for i in range(len(bin_edges) - 1):
                mask = (data.flatten() > bin_edges[i]) & (data.flatten() <= bin_edges[i+1])
                bin_representatives.append(int(np.median(data.flatten()[mask])))

            # 最后一个bin
            mask = data.flatten() > bin_edges[-1]
            bin_representatives.append(int(np.median(data.flatten()[mask])))

            self.density_bins = bin_representatives

            print("Bin Analysis:")
            for i, bin_val in enumerate(self.density_bins):
                print(f"  Bin {i+1}: representative={bin_val:4d}")

        elif strategy == 'hybrid':
            print(f"\nUsing HYBRID method with k={self.optimal_k}...\n")

            # 先用K-means找聚类
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(data)

            # 对每个聚类，使用中位数而不是均值
            bin_representatives = []
            for i in range(self.optimal_k):
                cluster_data = data[labels == i]
                bin_representatives.append(int(np.median(cluster_data)))

            self.density_bins = sorted(bin_representatives)

            print("Hybrid Bin Analysis:")
            for i, bin_val in enumerate(self.density_bins):
                count = np.sum(labels == i)
                percentage = 100 * count / len(data)
                print(f"  Bin {i+1}: value={bin_val:4d}, samples={count:5d} ({percentage:5.2f}%)")

        print(f"\n{'='*70}")
        print(f"FINAL DENSITY BINS: {self.density_bins}")
        print(f"{'='*70}")

        return self.density_bins

    def visualize_results(self, save_path='./complete_density_analysis.png'):
        """
        可视化完整的分析结果
        """
        print("\n" + "="*70)
        print("STEP 4: Generating Visualizations")
        print("="*70)

        fig = plt.figure(figsize=(18, 12))

        # 1. 数据分布直方图
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(self.all_patch_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Crowd Count', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Density Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 带有density bins标记的分布图
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(self.all_patch_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        for bin_val in self.density_bins:
            ax2.axvline(x=bin_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Crowd Count', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'Distribution with {self.optimal_k} Density Bins', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Silhouette scores (如果有)
        if 'silhouette' in self.evaluation_results:
            ax3 = plt.subplot(3, 3, 3)
            result = self.evaluation_results['silhouette']
            ax3.plot(result['k_range'], result['scores'], 'go-', linewidth=2, markersize=8)
            ax3.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2,
                       label=f"Optimal k={self.optimal_k}")
            ax3.set_xlabel('Number of Bins (k)', fontsize=11)
            ax3.set_ylabel('Silhouette Score', fontsize=11)
            ax3.set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. BIC scores (如果有)
        if 'bic' in self.evaluation_results:
            ax4 = plt.subplot(3, 3, 4)
            result = self.evaluation_results['bic']
            ax4.plot(result['k_range'], result['scores'], 'ro-', linewidth=2, markersize=8)
            ax4.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2,
                       label=f"Optimal k={self.optimal_k}")
            ax4.set_xlabel('Number of Bins (k)', fontsize=11)
            ax4.set_ylabel('BIC Score', fontsize=11)
            ax4.set_title('BIC Analysis (GMM)', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. CH Index (如果有)
        if 'ch_index' in self.evaluation_results:
            ax5 = plt.subplot(3, 3, 5)
            result = self.evaluation_results['ch_index']
            ax5.plot(result['k_range'], result['scores'], 'co-', linewidth=2, markersize=8)
            ax5.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2,
                       label=f"Optimal k={self.optimal_k}")
            ax5.set_xlabel('Number of Bins (k)', fontsize=11)
            ax5.set_ylabel('CH Index', fontsize=11)
            ax5.set_title('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Elbow Method (如果有)
        if 'elbow' in self.evaluation_results:
            ax6 = plt.subplot(3, 3, 6)
            result = self.evaluation_results['elbow']
            ax6.plot(result['k_range'], result['inertias'], 'bo-', linewidth=2, markersize=8)
            ax6.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2,
                       label=f"Optimal k={self.optimal_k}")
            ax6.set_xlabel('Number of Bins (k)', fontsize=11)
            ax6.set_ylabel('Inertia', fontsize=11)
            ax6.set_title('Elbow Method', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. 密度bins的可视化
        ax7 = plt.subplot(3, 3, 7)
        ax7.bar(range(len(self.density_bins)), self.density_bins, color='orange', alpha=0.7)
        ax7.set_xlabel('Bin Index', fontsize=11)
        ax7.set_ylabel('Density Value', fontsize=11)
        ax7.set_title('Density Bin Values', fontsize=12, fontweight='bold')
        ax7.set_xticks(range(len(self.density_bins)))
        ax7.set_xticklabels([f'Bin {i+1}' for i in range(len(self.density_bins))], rotation=45)
        ax7.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, v in enumerate(self.density_bins):
            ax7.text(i, v + max(self.density_bins)*0.02, str(v),
                    ha='center', va='bottom', fontweight='bold')

        # 8. 样本分布到各个bin
        ax8 = plt.subplot(3, 3, 8)
        data = self.all_patch_counts.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        bin_counts = [np.sum(labels == i) for i in range(self.optimal_k)]
        ax8.bar(range(len(bin_counts)), bin_counts, color='green', alpha=0.7)
        ax8.set_xlabel('Bin Index', fontsize=11)
        ax8.set_ylabel('Number of Samples', fontsize=11)
        ax8.set_title('Sample Distribution Across Bins', fontsize=12, fontweight='bold')
        ax8.set_xticks(range(len(bin_counts)))
        ax8.set_xticklabels([f'Bin {i+1}' for i in range(len(bin_counts))], rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')

        # 添加百分比标签
        for i, v in enumerate(bin_counts):
            percentage = 100 * v / len(data)
            ax8.text(i, v + max(bin_counts)*0.02, f'{v}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=9)

        # 9. 总结信息
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
        COMPLETE DENSITY ANALYSIS RESULTS
        ═══════════════════════════════════

        Optimal Number of Bins: {self.optimal_k}

        Density Bin Values:
        """

        for i, bin_val in enumerate(self.density_bins):
            count = bin_counts[i]
            percentage = 100 * count / len(data)
            summary_text += f"\n        Bin {i+1}: {bin_val:4d} ({count:5d} samples, {percentage:4.1f}%)"

        summary_text += f"""

        ───────────────────────────────────
        Original Fixed Bins (for comparison):
        [20, 55, 90, 125, 160, 195]

        New Adaptive Bins:
        {self.density_bins}
        ───────────────────────────────────

        Total Samples: {len(data)}
        Data Range: [{np.min(data):.0f}, {np.max(data):.0f}]
        Mean: {np.mean(data):.1f}
        Std: {np.std(data):.1f}
        """

        ax9.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    def save_configuration(self, save_path='./adaptive_density_config.json'):
        """
        保存完整的配置到JSON文件
        """
        print("\n" + "="*70)
        print("STEP 5: Saving Configuration")
        print("="*70)

        config = {
            'method': self.method,
            'optimal_k': self.optimal_k,
            'density_bins': self.density_bins,
            'crowd_count': [str(b) for b in self.density_bins],  # 字符串格式，兼容CLIP
            'evaluation_results': self.evaluation_results,
            'data_statistics': {
                'n_samples': len(self.all_patch_counts),
                'mean': float(np.mean(self.all_patch_counts)),
                'std': float(np.std(self.all_patch_counts)),
                'min': float(np.min(self.all_patch_counts)),
                'max': float(np.max(self.all_patch_counts)),
                'median': float(np.median(self.all_patch_counts)),
                'q25': float(np.percentile(self.all_patch_counts, 25)),
                'q75': float(np.percentile(self.all_patch_counts, 75))
            },
            'comparison': {
                'original_fixed_bins': [20, 55, 90, 125, 160, 195],
                'new_adaptive_bins': self.density_bins,
                'original_k': 6,
                'new_k': self.optimal_k
            }
        }

        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguration saved to: {save_path}")
        print(f"\nTo use in crowdclip.py:")
        print(f"  crowd_count = {config['crowd_count']}")
        print(f"\nOr load from config file:")
        print(f"  with open('{save_path}', 'r') as f:")
        print(f"      config = json.load(f)")
        print(f"      crowd_count = config['crowd_count']")

        return config

    def run_complete_analysis(self, bin_calculation_strategy='kmeans'):
        """
        运行完整的分析流程
        """
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*15 + "COMPLETE DENSITY BIN ANALYSIS" + " "*24 + "║")
        print("╚" + "="*68 + "╝\n")

        # Step 1: 收集数据
        self.collect_density_data()

        # Step 2: 找到最优k
        self.find_optimal_k()

        # Step 3: 计算density bins
        self.calculate_density_bins(strategy=bin_calculation_strategy)

        # Step 4: 可视化
        self.visualize_results(
            save_path='./processed_datasets/UCF-QNRF/complete_density_analysis.png'
        )

        # Step 5: 保存配置
        config = self.save_configuration(
            save_path='./processed_datasets/UCF-QNRF/adaptive_density_config.json'
        )

        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*20 + "ANALYSIS COMPLETE!" + " "*28 + "║")
        print("╚" + "="*68 + "╝\n")

        print("Summary:")
        print(f"  ✓ Optimal k: {self.optimal_k}")
        print(f"  ✓ Density bins: {self.density_bins}")
        print(f"  ✓ Visualization: ./processed_datasets/UCF-QNRF/complete_density_analysis.png")
        print(f"  ✓ Configuration: ./processed_datasets/UCF-QNRF/adaptive_density_config.json")

        print("\nNext Steps:")
        print("  1. Review the visualization to verify the bins make sense")
        print("  2. Use the configuration file with preprocess_qnrf_adaptive.py")
        print("  3. Train the model with the new density bins")

        return config


def main():
    """
    主函数
    """
    # 创建完整解决方案
    solution = CompleteDensityBinSolution(
        min_bins=3,
        max_bins=12,
        method='auto'  # 'auto', 'silhouette', 'bic', 'ch_index', 'elbow'
    )

    # 运行完整分析
    config = solution.run_complete_analysis(
        bin_calculation_strategy='kmeans'  # 'kmeans', 'quantile', 'hybrid'
    )

    print("\n" + "="*70)
    print("You can now use these results with:")
    print("="*70)
    print("\n# Option 1: Direct use in code")
    print(f"crowd_count = {config['crowd_count']}")
    print("\n# Option 2: Load from config file")
    print("import json")
    print("with open('./processed_datasets/UCF-QNRF/adaptive_density_config.json', 'r') as f:")
    print("    config = json.load(f)")
    print("    crowd_count = config['crowd_count']")
    print("    num_bins = config['optimal_k']")


if __name__ == '__main__':
    main()
