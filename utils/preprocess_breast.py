import argparse
import os
import json
import SimpleITK as sitk
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path
from glob import glob

SEED = 42
RESIZE_SIZE = (256, 256)
SPLIT_RATIOS = [0.7, 0.1, 0.2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='G:/WorkSpace/Medical_image_database/cardiac_data/breast/test_nii')
    parser.add_argument('-o', '--output_dir', type=str, default='G:/WorkSpace/Medical_image_database/cardiac_data/breast/output_breast')
    parser.add_argument('-f', '--split_file', type=str)
    args = parser.parse_args()

    return args

def sitk_load_resize(filepath: str | Path, resize_size) -> tuple[np.ndarray, dict]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    image = resampleXYSize(image, *resize_size)
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def generate_list(begin=1, end=100):
    number_list = ['patient' + str(i).zfill(4) for i in range(begin, end + 1)]
    return number_list

def random_split(data: list, ratios: list):
    total_length = len(data)
    split_points = [int(ratio * total_length) for ratio in ratios]
    random.seed(SEED)
    random.shuffle(data)
    splits = []
    start = 0
    for point in split_points:
        splits.append(data[start:start + point])
        start += point
    return splits

def resampleXYSize(sitkImage, new_xsize, new_ysize):
    # 重采样函数
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = xspacing / (new_xsize / float(xsize))
    new_spacing_y = yspacing / (new_ysize / float(ysize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    # 根据新的spacing 计算新的size
    newsize = (new_xsize, new_ysize, zsize)
    newspace = (new_spacing_x, new_spacing_y, zspacing)
    sitkImage = sitk.Resample(sitkImage, newsize, euler3d, sitk.sitkNearestNeighbor, origin, newspace, direction)
    return sitkImage

def preprocess_data(input_path, output_path, split_file):
    # hyperparam
    resize_size = RESIZE_SIZE
    # generate patient list
    patient_name_list = generate_list(1, 100)

    # split dataset
    if split_file:
        # from split file
        with open(split_file, 'r') as f:
            data = json.load(f)
            train_split, val_split, test_split = data['train'], data['val'], data['test']
    else:
        # random split
        data_split = random_split(data=patient_name_list, ratios=SPLIT_RATIOS)
        train_split, val_split, test_split = data_split

    # create folder
    if not os.path.exists(output_path + '/videos'):
        os.makedirs(output_path + '/videos/train')
        os.makedirs(output_path + '/videos/val')
        os.makedirs(output_path + '/videos/test')
    if not os.path.exists(output_path + '/annotations'):
        os.makedirs(output_path + '/annotations/train')
        os.makedirs(output_path + '/annotations/val')
        os.makedirs(output_path + '/annotations/test')

    for idx, patient_name in enumerate(patient_name_list):
        patient_root = Path(input_path)
        patient_dir = patient_root / patient_name

        # 查找 .nii 和 _gt.nii 文件
        seq_file = patient_dir / f"{patient_name}.nii"
        seq_gt_file = patient_dir / f"{patient_name}_gt.nii"

        if not seq_file.exists() or not seq_gt_file.exists():
            # print(f"跳过 {patient_name}，因为找不到相应的 .nii 或 _gt.nii 文件")
            continue

        seq, seq_info = sitk_load_resize(seq_file, resize_size)
        seq_gt, seq_gt_info = sitk_load_resize(seq_gt_file, resize_size)

        assert seq_info['spacing'] == seq_gt_info['spacing']

        # to rgb
        seq = np.repeat(seq[np.newaxis, :, :, :], 3, axis=0)

        if seq.shape[1] != seq_gt.shape[0]:
            # print(f"帧数不匹配: {patient_name}, 图像帧数: {seq.shape[1]}, 掩码帧数: {seq_gt.shape[0]}")
            continue

        frame_pairs_mask = {}
        for idxx, frame_gt in enumerate(seq_gt):
            frame_pairs_mask[str(idxx)] = frame_gt

        # save
        if patient_name in train_split:
            video_save_path = os.path.join(output_path, 'videos/train')
            anno_save_path = os.path.join(output_path, 'annotations/train')
        elif patient_name in val_split:
            video_save_path = os.path.join(output_path, 'videos/val')
            anno_save_path = os.path.join(output_path, 'annotations/val')
        elif patient_name in test_split:
            video_save_path = os.path.join(output_path, 'videos/test')
            anno_save_path = os.path.join(output_path, 'annotations/test')

        # save video
        seq_save_pattern = f"{patient_name}.npy"
        seq_gt_save_pattern = f"{patient_name}_gt.npz"

        np.save(os.path.join(video_save_path, seq_save_pattern), seq)
        # save anno
        np.savez(
            os.path.join(anno_save_path, seq_gt_save_pattern),
            fnum_mask=frame_pairs_mask,
            spacing=seq_info['spacing']
        )
        # print(idx + 1, seq_save_pattern)

    # create split txt
    output_file_train = open(output_path + '/breast_train_filenames.txt', 'w')
    output_file_val = open(output_path + '/breast_val_filenames.txt', 'w')
    output_file_test = open(output_path + '/breast_test_filenames.txt', 'w')

    for name in train_split:
        output_file_train.write(name + '\n')
    for name in val_split:
        output_file_val.write(name + '\n')
    for name in test_split:
        output_file_test.write(name + '\n')

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise ValueError('Input directory does not exist.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_data(input_path=args.input_dir,
                    output_path=args.output_dir,
                    split_file=args.split_file)
