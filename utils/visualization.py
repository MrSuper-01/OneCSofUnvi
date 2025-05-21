import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
import seaborn as sns
from utils.tools import draw_sem_seg_by_cv2_sum

def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/videos', 'test', image_filename))
    img_ori = cv2.resize(img_ori, (256, 256))
    img_ori0 = cv2.imread(os.path.join(opt.data_path +'/videos', 'test', image_filename))
    img_ori0 = cv2.resize(img_ori0, (256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    # table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
    #                           [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    table = np.array([[0, 0, 0], [0, 0, 155]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) 
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
    fulldir = opt.result_path + "/vis/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


# 1127重写
def save_mask_visualization(seg, image_filename, opt):
    """
    可视化分割掩码并叠加到对应的原始图像上。
    Args:
        seg: 分割结果，形状为 (1, H, W) 或 (H, W)
        image_filename: 原始图像文件名
        opt: 配置对象，包含数据路径和结果路径等
    """
    # 构建完整路径
    img_path = os.path.join(opt.data_path, 'videos', 'test', image_filename)

    # 加载原始图像
    try:
        img_ori = np.load(img_path)
    except FileNotFoundError:
        raise ValueError(f"原始图像文件未找到: {img_path}")

    # 检查图像是否为空
    if img_ori is None or img_ori.size == 0:
        raise ValueError(f"原始图像为空: {img_path}")

    # print(f"Shape of loaded image: {img_ori.shape}")

    # 如果是四维数组（多帧），选择第一帧和第一个通道
    if img_ori.ndim == 4:
        img_ori = img_ori[0, 0]  # (H, W)
    elif img_ori.ndim == 3:
        img_ori = img_ori[0]  # (H, W)
    else:
        raise ValueError(f"不支持的图像形状: {img_ori.shape}")

    # print(f"Selected first frame and channel, new shape: {img_ori.shape}")

    # 检查掩码维度
    if seg.ndim == 3:
        seg = seg[0]  # 从 (1, H, W) 提取到 (H, W)

    # 检查掩码值是否有效
    unique_values = np.unique(seg)
    # print(f"Unique values in seg: {unique_values}")
    if len(unique_values) == 1 and unique_values[0] == 0:
        # print(f"No segmentation region found for {image_filename}, skipping visualization.")
        return

    # 调整图像为 uint8 格式
    img_ori = (img_ori * 255).astype(np.uint8)

    # 调整掩码为彩色
    mask_colored = cv2.applyColorMap((seg * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 叠加掩码和原始图像
    overlay = cv2.addWeighted(cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR), 0.7, mask_colored, 0.3, 0)

    # 构建保存路径
    save_dir = os.path.join(opt.result_path, "mask_vis", opt.modelname)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{image_filename}.png")

    # 保存结果
    cv2.imwrite(save_path, overlay)
    # print(f"Mask visualization saved at: {save_path}")



def visual_segmentation_npy(pred, gt, image_filename, opt, img_ori, frameidx:int):
    palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]
    img_ori = img_ori[0,...]
    if not isinstance(img_ori, np.ndarray):
        img_ori = img_ori.detach().cpu().numpy()
    img_ori, gt, pred = img_ori.astype(np.uint8), gt.astype(np.uint8), pred.astype(np.uint8)

    img = draw_sem_seg_by_cv2_sum(img_ori, gt, pred, palette)      
    img = cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2BGR)

    fulldir = opt.result_path + "vis/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename.split('.')[0] + f'_{frameidx}.png', img)

# def visual_segmentation_sets(seg, image_filename, opt):
#     img_path = os.path.join(opt.data_subpath + '/img', image_filename)
#     img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
#     img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
#     img_ori = cv2.resize(img_ori, dsize=(256, 256))
#     img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
#     overlay = img_ori * 0
#     img_r = img_ori[:, :, 0]
#     img_g = img_ori[:, :, 1]
#     img_b = img_ori[:, :, 2]
#     table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
#                               [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
#     seg0 = seg[0, :, :]
            
#     for i in range(1, opt.classes):
#         img_r[seg0 == i] = table[i - 1, 0]
#         img_g[seg0 == i] = table[i - 1, 1]
#         img_b[seg0 == i] = table[i - 1, 2]
            
#     overlay[:, :, 0] = img_r
#     overlay[:, :, 1] = img_g
#     overlay[:, :, 2] = img_b
#     overlay = np.uint8(overlay)
 
#     img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
#     #img = img_ori0
          
#     fulldir = opt.result_path + "/" + opt.modelname + "/"
#     #fulldir = opt.result_path + "/" + "GT" + "/"
#     if not os.path.isdir(fulldir):
#         os.makedirs(fulldir)
#     cv2.imwrite(fulldir + image_filename, img)

# def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt):
#     img_path = os.path.join(opt.data_subpath, 'videos', 'test', image_filename)
#     img_ori = cv2.imread(img_path)
#     img_ori0 = cv2.imread(img_path)

#     if img_ori is None or img_ori.size == 0:
#         raise ValueError(f"Input image is empty for file: {image_filename} at path: {img_path}")

#     if img_ori is None or img_ori.size == 0:
#         raise ValueError(f"Input image is empty for file: {image_filename}")
#     print(f"Shape of img_ori: {img_ori.shape if img_ori is not None else 'None'}")
#     img_ori = cv2.resize(img_ori, dsize=(256, 256))
#     img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
#     overlay = img_ori * 0
#     img_r = img_ori[:, :, 0]
#     img_g = img_ori[:, :, 1]
#     img_b = img_ori[:, :, 2]
#     table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
#                               [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
#     seg0 = seg[0, :, :]
            
#     for i in range(1, opt.classes):
#         img_r[seg0 == i] = table[i - 1, 0]
#         img_g[seg0 == i] = table[i - 1, 1]
#         img_b[seg0 == i] = table[i - 1, 2]
            
#     overlay[:, :, 0] = img_r
#     overlay[:, :, 1] = img_g
#     overlay[:, :, 2] = img_b
#     overlay = np.uint8(overlay)
 
#     img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
#     #img = img_ori0
    
#     pt = np.array(pt.cpu())
#     N = pt.shape[0]
#     # for i in range(N):
#     #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 6, (0,0,0), -1)
#     #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 5, (0,0,255), -1)
#     #     cv2.line(img, (int(pt[i, 0]-3), int(pt[i, 1])), (int(pt[i, 0])+3, int(pt[i, 1])), (0, 0, 0), 1)
#     #     cv2.line(img, (int(pt[i, 0]), int(pt[i, 1])-3), (int(pt[i, 0]), int(pt[i, 1])+3), (0, 0, 0), 1)
          
#     fulldir = opt.result_path + "/PT10-" + opt.modelname + "/"
#     #fulldir = opt.result_path + "/PT3-" + "img" + "/"
#     if not os.path.isdir(fulldir):
#         os.makedirs(fulldir)
#     cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt):
    # 构建完整路径
    img_path = os.path.join(opt.data_subpath, 'videos', 'test', image_filename) # 这里需要修改

    # 尝试使用 numpy 加载 npy 文件
    try:
        img_ori = np.load(img_path)
        img_ori0 = img_ori.copy()  # 为后续叠加操作保存一个副本
    except FileNotFoundError:
        raise ValueError(f"File not found at path: {img_path}")

    if img_ori is None or img_ori.size == 0:
        raise ValueError(f"Input image is empty for file: {image_filename} at path: {img_path}")
    # print(f"Shape of img_ori: {img_ori.shape if img_ori is not None else 'None'}")

    # 如果是四维数组，选择第一个帧
    if img_ori.ndim == 4:
        img_ori = img_ori[0]  # 选择第一个帧，新的形状为 (C, H, W)
        img_ori0 = img_ori0[0]

    # 将三维数组从 (C, H, W) 转换为 (H, W, C)
    if img_ori.ndim == 3:
        if img_ori.shape[0] in [1, 3]:  # 如果通道数是 1 或 3
            img_ori = np.transpose(img_ori, (1, 2, 0))  # 转换为 (H, W, C)
            img_ori0 = np.transpose(img_ori0, (1, 2, 0))
        else:
            # 如果通道数不是 1 或 3，则只选择第一个通道
            img_ori = img_ori[0]  # 选择第一个通道
            img_ori0 = img_ori0[0]

    # 进行 resize 操作
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))

    # 检查是否为三通道，否则转换
    if img_ori.ndim == 2:  # 如果是单通道
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)
        img_ori0 = cv2.cvtColor(img_ori0, cv2.COLOR_GRAY2BGR)

    # 获取各个通道
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]

    overlay = np.zeros_like(img_ori, dtype=np.uint8)

    # 创建调色板
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                      [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]  # 获取分割结果

    # 使用布尔索引更新每个通道的值
    for i in range(1, min(opt.classes, len(table) + 1)):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]

    # 将修改后的通道重新组合成彩色叠加图像
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b

    # 将叠加的分割结果与原始图像混合
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0)

    # 确保 pt 是一个二维数组，形状为 (N, 2)
    pt = np.array(pt.cpu())
    if pt.ndim == 3 and pt.shape[1] == 1:
        pt = pt.squeeze(1)  # 去掉多余的维度，转换为 (N, 2)
    elif pt.ndim == 2 and pt.shape[1] != 2:
        raise ValueError(f"Unexpected shape for pt: {pt.shape}. Expected a (N, 2) array for point coordinates.")

    # 在图像上绘制点
    N = pt.shape[0]
    for i in range(N):
        if len(pt[i]) == 2:
            x, y = pt[i]  # 获取每个点的 x 和 y 坐标
            cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), -1)  # 黑色边框
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色填充
        else:
            raise ValueError(f"Point {i} does not have two coordinates: {pt[i]}")

    # 修改保存路径，确保文件名有合适的扩展名
    if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_filename += '.png'  # 默认保存为 .png 格式

    fulldir = os.path.join(opt.result_path, "PT10-" + opt.modelname)
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    save_path = os.path.join(fulldir, image_filename)

    # 保存图像
    cv2.imwrite(save_path, img)



def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)