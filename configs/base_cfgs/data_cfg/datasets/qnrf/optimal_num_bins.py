"""
自动确定QNRF数据集的最优密度区间数量
支持多种评估方法来找到最佳的bin数量
"""
import os
import numpy as np
import scipy.io as io
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class OptimalBinsFinder:
    """
    自动寻找最优的密度区间数量
    """
    def __init__(self, min_bins=3, max_bins=12):
        """
        Args:
            min_bins: 最小区间数（至少3个）
            max_bins: 最大区间数（建议不超过12个，避免过度细分）
        """
        self.min_bins = max(3, min_bins)
        self.max_bins = min(20, max_bins)
        self.all_patch_counts = None
        self.results = {}

    def collect_density_from_dataset(self, root_path='./datasets/UCF-QNRF', split='Train'):
        """
        从数据集中收集所有patch的密度信息
        """
        img_path = os.path.join(root_path, split)
        img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

        all_patch_counts = []

        print(f"Collecting density information from {len(img_files)} images...")

        for idx, img_file in enumerate(img_files):
            if idx % 50 == 0:
                print(f"Processing {idx}/{len(img_files)}...")

            # 读取图像
            img_full_path = os.path.join(img_path, img_file)
            img_data = cv2.imread(img_full_path)

            # 计算缩放比例
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

            # 模拟多种patch切分方式，收集密度分布
            height, width = img_data.shape[0], img_data.shape[1]

            # 1. 中心多尺度切分（模拟训练切分）
            center_y, center_x = height // 2, width // 2
            item_height, item_width = height // 12, width // 12

            for i in range(6):
                y1 = max(0, center_y - (i+1) * item_height)
                y2 = min(height, center_y + (i+1) * item_height)
                x1 = max(0, center_x - (i+1) * item_width)
                x2 = min(width, center_x + (i+1) * item_width)
                patch_count = int(np.sum(kpoint[y1:y2, x1:x2]))
                all_patch_counts.append(patch_count)

            # 2. 随机采样额外的patch（增加多样性）
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

        print(f"\nDataset statistics:")
        print(f"Total patch samples: {len(self.all_patch_counts)}")
        print(f"Density range: [{np.min(self.all_patch_counts)}, {np.max(self.all_patch_counts)}]")
        print(f"Mean density: {np.mean(self.all_patch_counts):.2f}")
        print(f"Median density: {np.median(self.all_patch_counts):.2f}")
        print(f"Std density: {np.std(self.all_patch_counts):.2f}")

        return self.all_patch_counts

    def method1_elbow_kmeans(self):
        """
        方法1：肘部法则 - 使用K-means的惯性（inertia）找到最优k
        惯性是所有样本到其所属聚类中心的距离平方和
        """
        print("\n" + "="*60)
        print("Method 1: Elbow Method (K-means Inertia)")
        print("="*60)

        data = self.all_patch_counts.reshape(-1, 1)
        inertias = []
        k_range = range(self.min_bins, self.max_bins + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            print(f"k={k:2d}, Inertia={kmeans.inertia_:10.2f}")

        # 计算肘部点（二阶导数最大的点）
        # 使用差分法近似二阶导数
        inertias = np.array(inertias)
        if len(inertias) > 2:
            first_diff = np.diff(inertias)
            second_diff = np.diff(first_diff)
            # 找到二阶导数最大（最负）的点
            elbow_idx = np.argmax(second_diff) + 1  # +1因为diff操作减少了一个元素
            optimal_k = list(k_range)[elbow_idx]
        else:
            optimal_k = self.min_bins

        self.results['elbow_kmeans'] = {
            'k_range': list(k_range),
            'inertias': inertias.tolist(),
            'optimal_k': optimal_k,
            'second_diff': second_diff.tolist() if len(inertias) > 2 else []
        }

        print(f"\nOptimal k by Elbow Method: {optimal_k}")
        return optimal_k

    def method2_silhouette(self):
        """
        方法2：轮廓系数 - 衡量聚类的紧密度和分离度
        轮廓系数范围[-1, 1]，值越大表示聚类效果越好
        """
        print("\n" + "="*60)
        print("Method 2: Silhouette Score")
        print("="*60)

        data = self.all_patch_counts.reshape(-1, 1)
        silhouette_scores = []
        k_range = range(self.min_bins, self.max_bins + 1)

        for k in k_range:
            if k >= len(data):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            # 计算轮廓系数
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            print(f"k={k:2d}, Silhouette Score={score:.4f}")

        # 选择轮廓系数最大的k
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]

        self.results['silhouette'] = {
            'k_range': list(k_range)[:len(silhouette_scores)],
            'scores': silhouette_scores,
            'optimal_k': optimal_k
        }

        print(f"\nOptimal k by Silhouette Score: {optimal_k}")
        return optimal_k

    def method3_bic_gmm(self):
        """
        方法3：贝叶斯信息准则(BIC) - 使用高斯混合模型
        BIC考虑了模型复杂度，值越小越好
        """
        print("\n" + "="*60)
        print("Method 3: BIC with Gaussian Mixture Model")
        print("="*60)

        data = self.all_patch_counts.reshape(-1, 1)
        bic_scores = []
        aic_scores = []
        k_range = range(self.min_bins, self.max_bins + 1)

        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
            gmm.fit(data)
            bic = gmm.bic(data)
            aic = gmm.aic(data)
            bic_scores.append(bic)
            aic_scores.append(aic)
            print(f"k={k:2d}, BIC={bic:10.2f}, AIC={aic:10.2f}")

        # 选择BIC最小的k
        optimal_idx_bic = np.argmin(bic_scores)
        optimal_k_bic = list(k_range)[optimal_idx_bic]

        # 选择AIC最小的k
        optimal_idx_aic = np.argmin(aic_scores)
        optimal_k_aic = list(k_range)[optimal_idx_aic]

        self.results['bic_gmm'] = {
            'k_range': list(k_range),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'optimal_k_bic': optimal_k_bic,
            'optimal_k_aic': optimal_k_aic
        }

        print(f"\nOptimal k by BIC: {optimal_k_bic}")
        print(f"Optimal k by AIC: {optimal_k_aic}")
        return optimal_k_bic

    def method4_variance_ratio(self):
        """
        方法4：方差比率 - Calinski-Harabasz指数和Davies-Bouldin指数
        CH指数：类间方差/类内方差，越大越好
        DB指数：平均类内距离/类间距离，越小越好
        """
        print("\n" + "="*60)
        print("Method 4: Variance Ratio Indices")
        print("="*60)

        data = self.all_patch_counts.reshape(-1, 1)
        ch_scores = []
        db_scores = []
        k_range = range(self.min_bins, self.max_bins + 1)

        for k in k_range:
            if k >= len(data):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            # Calinski-Harabasz指数（越大越好）
            ch_score = calinski_harabasz_score(data, labels)
            ch_scores.append(ch_score)

            # Davies-Bouldin指数（越小越好）
            db_score = davies_bouldin_score(data, labels)
            db_scores.append(db_score)

            print(f"k={k:2d}, CH Index={ch_score:8.2f}, DB Index={db_score:.4f}")

        # CH指数最大的k
        optimal_idx_ch = np.argmax(ch_scores)
        optimal_k_ch = list(k_range)[optimal_idx_ch]

        # DB指数最小的k
        optimal_idx_db = np.argmin(db_scores)
        optimal_k_db = list(k_range)[optimal_idx_db]

        self.results['variance_ratio'] = {
            'k_range': list(k_range)[:len(ch_scores)],
            'ch_scores': ch_scores,
            'db_scores': db_scores,
            'optimal_k_ch': optimal_k_ch,
            'optimal_k_db': optimal_k_db
        }

        print(f"\nOptimal k by CH Index: {optimal_k_ch}")
        print(f"Optimal k by DB Index: {optimal_k_db}")
        return optimal_k_ch

    def method5_distribution_analysis(self):
        """
        方法5：分布分析 - 基于数据的统计特性
        考虑数据的分位数范围、密度分布的峰值等
        """
        print("\n" + "="*60)
        print("Method 5: Distribution Analysis")
        print("="*60)

        data = self.all_patch_counts

        # 计算数据的统计特性
        data_range = np.max(data) - np.min(data)
        std_dev = np.std(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)

        # 方法5a: 基于Sturges规则（用于直方图）
        n = len(data)
        k_sturges = int(np.ceil(np.log2(n) + 1))

        # 方法5b: 基于Scott规则
        h_scott = 3.5 * std_dev / (n ** (1/3))  # 最优bin宽度
        k_scott = int(np.ceil(data_range / h_scott))

        # 方法5c: 基于Freedman-Diaconis规则
        h_fd = 2 * iqr / (n ** (1/3))
        k_fd = int(np.ceil(data_range / h_fd))

        # 方法5d: 基于变异系数
        cv = std_dev / np.mean(data)  # 变异系数
        # 变异系数越大，数据越离散，需要更多bins
        k_cv = int(np.clip(6 + cv * 4, self.min_bins, self.max_bins))

        print(f"Data range: {data_range:.2f}")
        print(f"Std dev: {std_dev:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(f"Coefficient of Variation: {cv:.4f}")
        print(f"\nk by Sturges rule: {k_sturges}")
        print(f"k by Scott rule: {k_scott}")
        print(f"k by Freedman-Diaconis rule: {k_fd}")
        print(f"k by Coefficient of Variation: {k_cv}")

        # 取这些方法的中位数作为推荐值
        suggested_ks = [k_sturges, k_scott, k_fd, k_cv]
        # 限制在合理范围内
        suggested_ks = [max(self.min_bins, min(self.max_bins, k)) for k in suggested_ks]
        optimal_k = int(np.median(suggested_ks))

        self.results['distribution_analysis'] = {
            'k_sturges': k_sturges,
            'k_scott': k_scott,
            'k_fd': k_fd,
            'k_cv': k_cv,
            'optimal_k': optimal_k
        }

        print(f"\nRecommended k by Distribution Analysis: {optimal_k}")
        return optimal_k

    def method6_gap_statistic(self):
        """
        方法6：Gap统计量 - 比较聚类结果与随机数据的差异
        Gap统计量越大，说明聚类效果越好
        """
        print("\n" + "="*60)
        print("Method 6: Gap Statistic")
        print("="*60)

        data = self.all_patch_counts.reshape(-1, 1)
        k_range = range(self.min_bins, min(self.max_bins + 1, 11))  # Gap统计量计算较慢，限制范围
        n_refs = 10  # 参考数据集数量

        gaps = []
        sk = []

        for k in k_range:
            print(f"Computing Gap statistic for k={k}...")

            # 原始数据的聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            disp_orig = np.log(kmeans.inertia_)

            # 生成参考数据集（均匀分布）
            ref_disps = []
            for _ in range(n_refs):
                # 在数据范围内生成均匀分布的参考数据
                ref_data = np.random.uniform(
                    low=np.min(data),
                    high=np.max(data),
                    size=data.shape
                )
                kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_ref.fit(ref_data)
                ref_disps.append(np.log(kmeans_ref.inertia_))

            ref_disps = np.array(ref_disps)
            gap = np.mean(ref_disps) - disp_orig
            sk_val = np.std(ref_disps) * np.sqrt(1 + 1.0/n_refs)

            gaps.append(gap)
            sk.append(sk_val)

            print(f"k={k}, Gap={gap:.4f}, sk={sk_val:.4f}")

        # 找到满足 Gap(k) >= Gap(k+1) - s(k+1) 的最小k
        gaps = np.array(gaps)
        sk = np.array(sk)

        optimal_k = list(k_range)[0]
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i+1] - sk[i+1]:
                optimal_k = list(k_range)[i]
                break

        self.results['gap_statistic'] = {
            'k_range': list(k_range),
            'gaps': gaps.tolist(),
            'sk': sk.tolist(),
            'optimal_k': optimal_k
        }

        print(f"\nOptimal k by Gap Statistic: {optimal_k}")
        return optimal_k

    def visualize_all_methods(self, save_path='./optimal_k_analysis.png'):
        """
        可视化所有方法的结果
        """
        fig = plt.figure(figsize=(18, 12))

        # 1. Elbow Method
        if 'elbow_kmeans' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            result = self.results['elbow_kmeans']
            ax1.plot(result['k_range'], result['inertias'], 'bo-', linewidth=2, markersize=8)
            ax1.axvline(x=result['optimal_k'], color='r', linestyle='--', linewidth=2,
                       label=f"Optimal k={result['optimal_k']}")
            ax1.set_xlabel('Number of Bins (k)', fontsize=11)
            ax1.set_ylabel('Inertia', fontsize=11)
            ax1.set_title('Elbow Method (K-means)', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Silhouette Score
        if 'silhouette' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            result = self.results['silhouette']
            ax2.plot(result['k_range'], result['scores'], 'go-', linewidth=2, markersize=8)
            ax2.axvline(x=result['optimal_k'], color='r', linestyle='--', linewidth=2,
                       label=f"Optimal k={result['optimal_k']}")
            ax2.set_xlabel('Number of Bins (k)', fontsize=11)
            ax2.set_ylabel('Silhouette Score', fontsize=11)
            ax2.set_title('Silhouette Score', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. BIC/AIC
        if 'bic_gmm' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            result = self.results['bic_gmm']
            ax3.plot(result['k_range'], result['bic_scores'], 'ro-', linewidth=2,
                    markersize=8, label='BIC')
            ax3.plot(result['k_range'], result['aic_scores'], 'mo-', linewidth=2,
                    markersize=8, label='AIC')
            ax3.axvline(x=result['optimal_k_bic'], color='r', linestyle='--', linewidth=2,
                       alpha=0.7, label=f"Optimal k(BIC)={result['optimal_k_bic']}")
            ax3.set_xlabel('Number of Bins (k)', fontsize=11)
            ax3.set_ylabel('Information Criterion', fontsize=11)
            ax3.set_title('BIC/AIC (GMM)', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Calinski-Harabasz Index
        if 'variance_ratio' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            result = self.results['variance_ratio']
            ax4.plot(result['k_range'], result['ch_scores'], 'co-', linewidth=2, markersize=8)
            ax4.axvline(x=result['optimal_k_ch'], color='r', linestyle='--', linewidth=2,
                       label=f"Optimal k={result['optimal_k_ch']}")
            ax4.set_xlabel('Number of Bins (k)', fontsize=11)
            ax4.set_ylabel('CH Index', fontsize=11)
            ax4.set_title('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Davies-Bouldin Index
        if 'variance_ratio' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            result = self.results['variance_ratio']
            ax5.plot(result['k_range'], result['db_scores'], 'yo-', linewidth=2, markersize=8)
            ax5.axvline(x=result['optimal_k_db'], color='r', linestyle='--', linewidth=2,
                       label=f"Optimal k={result['optimal_k_db']}")
            ax5.set_xlabel('Number of Bins (k)', fontsize=11)
            ax5.set_ylabel('DB Index', fontsize=11)
            ax5.set_title('Davies-Bouldin Index', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Gap Statistic
        if 'gap_statistic' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            result = self.results['gap_statistic']
            ax6.plot(result['k_range'], result['gaps'], 'ko-', linewidth=2, markersize=8)
            ax6.errorbar(result['k_range'], result['gaps'], yerr=result['sk'],
                        fmt='o', color='black', alpha=0.5)
            ax6.axvline(x=result['optimal_k'], color='r', linestyle='--', linewidth=2,
                       label=f"Optimal k={result['optimal_k']}")
            ax6.set_xlabel('Number of Bins (k)', fontsize=11)
            ax6.set_ylabel('Gap Statistic', fontsize=11)
            ax6.set_title('Gap Statistic', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Distribution histogram
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(self.all_patch_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax7.set_xlabel('Crowd Count', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('Density Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # 8. Summary of all optimal k values
        ax8 = plt.subplot(3, 3, 8)
        optimal_ks = []
        method_names = []

        if 'elbow_kmeans' in self.results:
            optimal_ks.append(self.results['elbow_kmeans']['optimal_k'])
            method_names.append('Elbow')
        if 'silhouette' in self.results:
            optimal_ks.append(self.results['silhouette']['optimal_k'])
            method_names.append('Silhouette')
        if 'bic_gmm' in self.results:
            optimal_ks.append(self.results['bic_gmm']['optimal_k_bic'])
            method_names.append('BIC')
        if 'variance_ratio' in self.results:
            optimal_ks.append(self.results['variance_ratio']['optimal_k_ch'])
            method_names.append('CH Index')
        if 'gap_statistic' in self.results:
            optimal_ks.append(self.results['gap_statistic']['optimal_k'])
            method_names.append('Gap Stat')
        if 'distribution_analysis' in self.results:
            optimal_ks.append(self.results['distribution_analysis']['optimal_k'])
            method_names.append('Dist Analysis')

        colors = ['blue', 'green', 'red', 'cyan', 'black', 'orange']
        ax8.bar(range(len(optimal_ks)), optimal_ks, color=colors[:len(optimal_ks)], alpha=0.7)
        ax8.set_xticks(range(len(optimal_ks)))
        ax8.set_xticklabels(method_names, rotation=45, ha='right')
        ax8.set_ylabel('Optimal k', fontsize=11)
        ax8.set_title('Comparison of Optimal k', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        # 添加值标签
        for i, v in enumerate(optimal_ks):
            ax8.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

        # 9. Consensus recommendation
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # 计算推荐值
        recommended_k = int(np.median(optimal_ks))
        mean_k = np.mean(optimal_ks)
        mode_k = Counter(optimal_ks).most_common(1)[0][0]

        summary_text = f"""
        OPTIMAL K RECOMMENDATION
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━

        All Methods Summary:
        • Elbow Method: {self.results.get('elbow_kmeans', {}).get('optimal_k', 'N/A')}
        • Silhouette: {self.results.get('silhouette', {}).get('optimal_k', 'N/A')}
        • BIC (GMM): {self.results.get('bic_gmm', {}).get('optimal_k_bic', 'N/A')}
        • CH Index: {self.results.get('variance_ratio', {}).get('optimal_k_ch', 'N/A')}
        • Gap Statistic: {self.results.get('gap_statistic', {}).get('optimal_k', 'N/A')}
        • Dist Analysis: {self.results.get('distribution_analysis', {}).get('optimal_k', 'N/A')}

        Statistical Summary:
        • Median: {recommended_k}
        • Mean: {mean_k:.1f}
        • Mode: {mode_k}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        RECOMMENDED: k = {recommended_k}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━

        This is based on the median of
        all evaluation methods.
        """

        ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

        return recommended_k

    def get_final_recommendation(self):
        """
        综合所有方法，给出最终推荐
        """
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)

        all_optimal_ks = []

        # 收集所有方法的结果
        if 'elbow_kmeans' in self.results:
            all_optimal_ks.append(('Elbow Method', self.results['elbow_kmeans']['optimal_k']))

        if 'silhouette' in self.results:
            all_optimal_ks.append(('Silhouette Score', self.results['silhouette']['optimal_k']))

        if 'bic_gmm' in self.results:
            all_optimal_ks.append(('BIC (GMM)', self.results['bic_gmm']['optimal_k_bic']))
            all_optimal_ks.append(('AIC (GMM)', self.results['bic_gmm']['optimal_k_aic']))

        if 'variance_ratio' in self.results:
            all_optimal_ks.append(('CH Index', self.results['variance_ratio']['optimal_k_ch']))
            all_optimal_ks.append(('DB Index', self.results['variance_ratio']['optimal_k_db']))

        if 'gap_statistic' in self.results:
            all_optimal_ks.append(('Gap Statistic', self.results['gap_statistic']['optimal_k']))

        if 'distribution_analysis' in self.results:
            all_optimal_ks.append(('Distribution Analysis', self.results['distribution_analysis']['optimal_k']))

        # 打印所有结果
        print("\nAll methods' recommendations:")
        for method, k in all_optimal_ks:
            print(f"  {method:25s}: k = {k}")

        # 统计分析
        k_values = [k for _, k in all_optimal_ks]
        recommended_k = int(np.median(k_values))
        mean_k = np.mean(k_values)
        std_k = np.std(k_values)
        min_k = np.min(k_values)
        max_k = np.max(k_values)

        # 找出最常见的k值
        k_counter = Counter(k_values)
        mode_k, mode_count = k_counter.most_common(1)[0]

        print(f"\nStatistical Summary:")
        print(f"  Median:  {recommended_k}")
        print(f"  Mean:    {mean_k:.2f}")
        print(f"  Std:     {std_k:.2f}")
        print(f"  Range:   [{min_k}, {max_k}]")
        print(f"  Mode:    {mode_k} (appears {mode_count} times)")

        print(f"\n{'='*60}")
        print(f"RECOMMENDED: k = {recommended_k}")
        print(f"{'='*60}")

        # 给出建议范围
        conservative_k = max(self.min_bins, recommended_k - 1)
        aggressive_k = min(self.max_bins, recommended_k + 1)

        print(f"\nSuggested range for experiments:")
        print(f"  Conservative: k = {conservative_k} (fewer bins, simpler model)")
        print(f"  Recommended:  k = {recommended_k} (based on statistical analysis)")
        print(f"  Aggressive:   k = {aggressive_k} (more bins, finer granularity)")

        return {
            'recommended_k': recommended_k,
            'conservative_k': conservative_k,
            'aggressive_k': aggressive_k,
            'all_methods': all_optimal_ks,
            'statistics': {
                'median': recommended_k,
                'mean': mean_k,
                'std': std_k,
                'min': min_k,
                'max': max_k,
                'mode': mode_k
            }
        }

    def save_results(self, recommendation, save_path='./optimal_k_results.json'):
        """
        保存所有结果到JSON文件
        """
        output = {
            'recommendation': recommendation,
            'detailed_results': self.results,
            'data_statistics': {
                'n_samples': len(self.all_patch_counts),
                'mean': float(np.mean(self.all_patch_counts)),
                'std': float(np.std(self.all_patch_counts)),
                'min': float(np.min(self.all_patch_counts)),
                'max': float(np.max(self.all_patch_counts)),
                'median': float(np.median(self.all_patch_counts)),
                'q25': float(np.percentile(self.all_patch_counts, 25)),
                'q75': float(np.percentile(self.all_patch_counts, 75))
            }
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {save_path}")


def main():
    """
    主函数：运行所有方法并给出最终推荐
    """
    print("="*60)
    print("OPTIMAL NUMBER OF DENSITY BINS ANALYSIS")
    print("="*60)

    # 创建分析器
    finder = OptimalBinsFinder(min_bins=3, max_bins=12)

    # 收集数据
    print("\nStep 1: Collecting density data from QNRF dataset...")
    finder.collect_density_from_dataset(
        root_path='./datasets/UCF-QNRF',
        split='Train'
    )

    # 运行所有评估方法
    print("\nStep 2: Running all evaluation methods...")

    finder.method1_elbow_kmeans()
    finder.method2_silhouette()
    finder.method3_bic_gmm()
    finder.method4_variance_ratio()
    finder.method5_distribution_analysis()
    finder.method6_gap_statistic()

    # 可视化结果
    print("\nStep 3: Visualizing results...")
    finder.visualize_all_methods(
        save_path='./processed_datasets/UCF-QNRF/optimal_k_analysis.png'
    )

    # 获取最终推荐
    print("\nStep 4: Computing final recommendation...")
    recommendation = finder.get_final_recommendation()

    # 保存结果
    print("\nStep 5: Saving results...")
    finder.save_results(
        recommendation,
        save_path='./processed_datasets/UCF-QNRF/optimal_k_results.json'
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review the visualization: ./processed_datasets/UCF-QNRF/optimal_k_analysis.png")
    print(f"2. Check detailed results: ./processed_datasets/UCF-QNRF/optimal_k_results.json")
    print(f"3. Use k={recommendation['recommended_k']} to run auto_density_bins.py")
    print(f"   Example: calculator = DensityBinCalculator(num_bins={recommendation['recommended_k']})")


if __name__ == '__main__':
    main()
