"""
改进的QNRF数据预处理脚本
支持自适应密度感知的patch切分策略
"""
import os
import cv2
import numpy as np
import scipy.io as io
from PIL import Image
import json


def find_best_patches_for_density_bins(image, gt, density_bins, num_patches_per_bin=1,
                                       patch_size_ratio=0.3, stride_ratio=0.1):
    """
    为每个密度区间找到最匹配的patch

    Args:
        image: PIL Image对象
        gt: ground truth密度图（二值化）
        density_bins: 目标密度值列表，例如[20, 55, 90, 125, 160, 195]
        num_patches_per_bin: 每个密度bin生成几个patch
        patch_size_ratio: patch尺寸相对于图像的比例
        stride_ratio: 滑动窗口的步长比例

    Returns:
        patches_info: 列表，每个元素是(crop_box, actual_count, target_bin)
    """
    width, height = image.size
    patch_width = int(width * patch_size_ratio)
    patch_height = int(height * patch_size_ratio)
    stride_w = int(width * stride_ratio)
    stride_h = int(height * stride_ratio)

    # 存储所有可能的patch及其密度
    all_patches = []

    # 滑动窗口提取所有可能的patch
    for y in range(0, height - patch_height + 1, stride_h):
        for x in range(0, width - patch_width + 1, stride_w):
            box = (x, y, x + patch_width, y + patch_height)
            gt_crop = gt[y:y+patch_height, x:x+patch_width]
            count = int(np.sum(gt_crop))
            all_patches.append({
                'box': box,
                'count': count,
                'center': (x + patch_width//2, y + patch_height//2)
            })

    # 为每个密度bin找到最接近的patch
    selected_patches = []
    for target_density in density_bins:
        # 找到密度最接近目标值的patch
        candidates = sorted(all_patches, key=lambda p: abs(p['count'] - target_density))

        # 选择最佳的patch，同时避免过度重叠
        for candidate in candidates[:num_patches_per_bin * 3]:  # 从前几个候选中选择
            # 检查是否与已选择的patch重叠过多
            overlap = False
            for selected in selected_patches:
                if _calculate_iou(candidate['box'], selected['box']) > 0.5:
                    overlap = True
                    break

            if not overlap:
                selected_patches.append({
                    'box': candidate['box'],
                    'count': candidate['count'],
                    'target': target_density
                })
                break

        # 如果没有找到不重叠的patch，使用最佳候选
        if len(selected_patches) < len([p for p in selected_patches if p['target'] == target_density]) + 1:
            selected_patches.append({
                'box': candidates[0]['box'],
                'count': candidates[0]['count'],
                'target': target_density
            })

    return selected_patches


def _calculate_iou(box1, box2):
    """计算两个box的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def cut_image_train_adaptive(image, gt, save_path, fname, f, density_bins,
                              strategy='sliding_window'):
    """
    自适应的训练数据切分

    Args:
        image: PIL Image
        gt: ground truth密度图
        save_path: 保存路径
        fname: 文件名
        f: 文件句柄
        density_bins: 密度区间列表
        strategy: 切分策略
            - 'sliding_window': 滑动窗口寻找最匹配的patch
            - 'multi_scale': 多尺度中心裁剪（原始方法的改进版）
            - 'density_aware': 密度感知的自适应裁剪
    """
    width, height = image.size

    if strategy == 'sliding_window':
        # 使用滑动窗口找到最匹配各密度区间的patch
        patches_info = find_best_patches_for_density_bins(
            image, gt, density_bins,
            patch_size_ratio=0.4,
            stride_ratio=0.1
        )

        for index, patch_info in enumerate(patches_info):
            img_crop = image.crop(patch_info['box'])
            img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
            img_crop.save(img_save)

            f.write('{} {} {}'.format(
                fname + '_' + str(index) + '.jpg',
                patch_info['count'],
                patch_info['target']  # 记录目标密度
            ))
            f.write('\n')

    elif strategy == 'multi_scale':
        # 改进的多尺度方法：根据density_bins动态调整尺度
        center_x = int(width / 2)
        center_y = int(height / 2)

        # 根据密度bins智能选择patch大小
        total_count = int(np.sum(gt))

        for index, target_density in enumerate(density_bins):
            # 估算需要的patch尺寸（假设密度均匀分布）
            if total_count > 0:
                ratio = min(1.0, max(0.1, target_density / total_count))
                scale_factor = np.sqrt(ratio)
            else:
                scale_factor = (index + 1) / len(density_bins)

            item_width = int(width * scale_factor / 2)
            item_height = int(height * scale_factor / 2)

            box = (
                max(0, center_x - item_width),
                max(0, center_y - item_height),
                min(width, center_x + item_width),
                min(height, center_y + item_height)
            )

            y1, y2 = max(0, center_y - item_height), min(height, center_y + item_height)
            x1, x2 = max(0, center_x - item_width), min(width, center_x + item_width)

            gt_crop = gt[y1:y2, x1:x2]
            gt_count = int(np.sum(gt_crop))

            img_crop = image.crop(box)
            img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
            img_crop.save(img_save)

            f.write('{} {} {}'.format(
                fname + '_' + str(index) + '.jpg',
                gt_count,
                target_density
            ))
            f.write('\n')

    elif strategy == 'density_aware':
        # 密度感知策略：分析图像的密度分布热图
        patches_info = _extract_density_aware_patches(image, gt, density_bins)

        for index, patch_info in enumerate(patches_info):
            img_crop = image.crop(patch_info['box'])
            img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
            img_crop.save(img_save)

            f.write('{} {} {}'.format(
                fname + '_' + str(index) + '.jpg',
                patch_info['count'],
                patch_info['target']
            ))
            f.write('\n')


def _extract_density_aware_patches(image, gt, density_bins):
    """
    基于密度分布的智能patch提取
    将图像分成网格，分析每个区域的密度，然后组合成最佳patch
    """
    width, height = image.size
    grid_size = 8  # 将图像分成8x8的网格
    cell_w, cell_h = width // grid_size, height // grid_size

    # 计算每个网格单元的密度
    density_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            density_grid[i, j] = np.sum(gt[y1:y2, x1:x2])

    patches_info = []

    for target_density in density_bins:
        # 尝试不同的patch大小
        best_patch = None
        best_diff = float('inf')

        for patch_grid_size in [2, 3, 4, 6]:  # patch占据的网格数
            for i in range(grid_size - patch_grid_size + 1):
                for j in range(grid_size - patch_grid_size + 1):
                    # 计算这个patch的总密度
                    patch_density = np.sum(
                        density_grid[i:i+patch_grid_size, j:j+patch_grid_size]
                    )

                    diff = abs(patch_density - target_density)
                    if diff < best_diff:
                        best_diff = diff
                        box = (
                            j * cell_w,
                            i * cell_h,
                            (j + patch_grid_size) * cell_w,
                            (i + patch_grid_size) * cell_h
                        )
                        best_patch = {
                            'box': box,
                            'count': int(patch_density),
                            'target': target_density
                        }

        if best_patch:
            patches_info.append(best_patch)

    return patches_info


def cut_image_test(image, gt, save_path, fname, f, patch_num):
    """测试数据切分（保持原样）"""
    width, height = image.size
    item_width = int(width / patch_num)
    item_height = int(height / patch_num)
    index = 0
    for i in range(0, patch_num):
        for j in range(0, patch_num):
            img = image.copy()
            kpoint = gt.copy()
            box = (j * item_width, i * item_height, (j + 1) * item_width, (j + 1) * item_height)
            gt_crop = kpoint[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
            gt_count = int(np.sum(gt_crop))
            img_crop = img.crop(box)
            img_save = os.path.join(save_path, fname + '_' + str(index) + '.jpg')
            img_crop.save(img_save)
            f.write('{} {}'.format(fname + '_' + str(index) + '.jpg', gt_count))
            f.write('\n')
            index += 1


def main():
    # 加载自动计算的密度bins配置
    config_path = './processed_datasets/UCF-QNRF/density_bins_config.json'

    if os.path.exists(config_path):
        print(f"Loading density bins from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        density_bins = config['bins']
        print(f"Using auto-calculated density bins: {density_bins}")
    else:
        print(f"Config file not found. Using default bins.")
        print("Please run auto_density_bins.py first to calculate optimal bins.")
        density_bins = [20, 55, 90, 125, 160, 195]
        print(f"Using default density bins: {density_bins}")

    # 选择切分策略
    # 可选: 'sliding_window', 'multi_scale', 'density_aware'
    strategy = 'sliding_window'  # 推荐使用滑动窗口策略
    print(f"Using strategy: {strategy}")

    # 数据集路径
    root = './datasets/UCF-QNRF'
    img_test_path = root + '/Test/'
    img_train_path = root + '/Train/'
    save_test_img_path = root + '/test_data/images_2048/'
    save_train_img_path = root + '/train_data/images_2048/'

    if not os.path.exists(save_train_img_path):
        os.makedirs(save_train_img_path)
    if not os.path.exists(save_test_img_path):
        os.makedirs(save_test_img_path)

    img_train = []
    img_test = []
    for file_name in os.listdir(img_train_path):
        if file_name.split('.')[1] == 'jpg':
            img_train.append(file_name)
    for file_name in os.listdir(img_test_path):
        if file_name.split('.')[1] == 'jpg':
            img_test.append(file_name)
    img_train.sort()
    img_test.sort()
    print(f"Train images: {len(img_train)}, Test images: {len(img_test)}")

    save_path_test = './processed_datasets/UCF-QNRF/test_data/'
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)
    if not os.path.exists(save_path_test + 'data_list'):
        os.makedirs(save_path_test + 'data_list')
    f_count_test = open("./processed_datasets/UCF-QNRF/test_data/data_list/test.txt", "w+")

    save_path_train = './processed_datasets/UCF-QNRF/train_data/'
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
    if not os.path.exists(save_path_train + 'data_list'):
        os.makedirs(save_path_train + 'data_list')
    f_count_train = open("./processed_datasets/UCF-QNRF/train_data/data_list/train.txt", "w+")

    # 处理训练数据
    print("\nProcessing training data...")
    for k in range(len(img_train)):
        Img_data = cv2.imread(img_train_path + img_train[k])
        if k % 50 == 0:
            print(f"Processing train image {k}/{len(img_train)}: {img_train[k]}")

        rate = 1
        if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
            rate = 2048.0 / Img_data.shape[1]
        if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
            rate = 2048.0 / Img_data.shape[0]

        Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)
        new_img_path = (save_train_img_path + img_train[k])
        cv2.imwrite(new_img_path, Img_data)

        fname = os.path.basename(img_train_path + img_train[k]).split('.')[0]
        mat_path = os.path.join(img_train_path, fname + '_ann.mat')
        mat = io.loadmat(mat_path)
        Gt_data = mat['annPoints'] * rate
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        for i in range(0, len(Gt_data)):
            if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

        image = Image.open(save_train_img_path + img_train[k])
        cut_image_train_adaptive(
            image, kpoint, save_path_train, fname, f_count_train,
            density_bins=density_bins,
            strategy=strategy
        )

    # 处理测试数据
    print("\nProcessing test data...")
    for k in range(len(img_test)):
        Img_data = cv2.imread(img_test_path + img_test[k])
        if k % 50 == 0:
            print(f"Processing test image {k}/{len(img_test)}: {img_test[k]}")

        rate = 1
        if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
            rate = 2048.0 / Img_data.shape[1]
        if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
            rate = 2048.0 / Img_data.shape[0]

        Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)
        new_img_path = (save_test_img_path + img_test[k])
        cv2.imwrite(new_img_path, Img_data)

        fname = os.path.basename(img_test_path + img_test[k]).split('.')[0]
        mat_path = os.path.join(img_test_path, fname + '_ann.mat')
        mat = io.loadmat(mat_path)
        Gt_data = mat['annPoints'] * rate
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        for i in range(0, len(Gt_data)):
            if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

        image = Image.open(save_test_img_path + img_test[k])
        cut_image_test(image, kpoint, save_path_test, fname, f_count_test, patch_num=4)

    f_count_train.close()
    f_count_test.close()

    print("\nPreprocessing completed!")
    print(f"Density bins used: {density_bins}")
    print(f"Strategy used: {strategy}")


if __name__ == '__main__':
    main()
