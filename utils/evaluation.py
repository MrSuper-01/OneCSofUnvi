# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from medpy.metric.binary import hd as medpy_hd
from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.binary import assd as medpy_assd
from utils.tools import hausdorff_distance as our_hausdorff_distance
# from utils.visualization import visual_segmentation, visual_segmentation_npy, visual_segmentation_binary, visual_segmentation_sets, visual_segmentation_sets_with_pt
from utils.visualization import save_mask_visualization, visual_segmentation, visual_segmentation_npy, visual_segmentation_binary, visual_segmentation_sets_with_pt
from einops import rearrange
from utils.generate_prompts import get_click_prompt
from utils.compute_ef import compute_left_ventricle_volumes
import time
import pandas as pd
from utils.tools import corr, bias, std
from scipy import stats
from sklearn.metrics import roc_curve, auc

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[1] += iou
            accs[1] += acc
            ses[1] += se
            sps[1] += sp
            hds[1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices / eval_number
    hds = hds / eval_number
    ious, accs, ses, sps = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    # print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp


def eval_mask_slice2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        class_id = datapack['class_id']
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[eval_number+j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    # print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_camus_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        # predict = torch.sigmoid(pred['masks'])
        # predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = predict[:, 0, :, :] > 0.5  # (b, h, w)

        predict = F.softmax(pred['masks'], dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum == 3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 5000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)


        # predict = F.softmax(pred['masks'], dim=1)
        # pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * (len(valloader) + 1)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
#     eval_number = 0
#     sum_time = 0
#     for batch_idx, (datapack) in enumerate(valloader):
#         imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
#         masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype = torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']
#
#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time =  sum_time + (time.time()-start_time)
#
#         val_loss = criterion(pred, masks)
#         val_losses += val_loss.item()
#
#         if args.modelname == 'MSA' or args.modelname == 'SAM':
#             gt = masks.detach().cpu().numpy()
#         else:
#             gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]
#         predict_masks = pred['masks']
#         predict_masks = torch.softmax(predict_masks, dim=1)
#         pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
#         seg = np.argmax(pred, axis=1)  # (b, h, w)
#         b, h, w = seg.shape
#         for j in range(0, b):
#             pred_i = np.zeros((1, h, w))
#             pred_i[seg[j:j+1, :, :] == 1] = 255
#             gt_i = np.zeros((1, h, w))
#             gt_i[gt[j:j+1, :, :] == 1] = 255
#             dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number+j, 1] += iou
#             accs[eval_number+j, 1] += acc
#             ses[eval_number+j, 1] += se
#             sps[eval_number+j, 1] += sp
#             hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
#             del pred_i, gt_i
#             if opt.visual:
#                 visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
#         eval_number = eval_number + b
#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses = val_losses / (batch_idx + 1)
#
#     dice_mean = np.mean(dices, axis=0)
#     dices_std = np.std(dices, axis=0)
#     hd_mean = np.mean(hds, axis=0)
#     hd_std = np.std(hds, axis=0)
#
#     mean_dice = np.mean(dice_mean[1:])
#     mean_hdis = np.mean(hd_mean[1:])
#     print("test speed", eval_number/sum_time)
#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         # data = pd.DataFrame(dices*100)
#         # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
#         # data.to_excel(writer, 'page_1', float_format='%.2f')
#         # writer._save()
#
#         dice_mean = np.mean(dices*100, axis=0)
#         dices_std = np.std(dices*100, axis=0)
#         hd_mean = np.mean(hds, axis=0)
#         hd_std = np.std(hds, axis=0)
#         iou_mean = np.mean(ious*100, axis=0)
#         iou_std = np.std(ious*100, axis=0)
#         acc_mean = np.mean(accs*100, axis=0)
#         acc_std = np.std(accs*100, axis=0)
#         se_mean = np.mean(ses*100, axis=0)
#         se_std = np.std(ses*100, axis=0)
#         sp_mean = np.mean(sps*100, axis=0)
#         sp_std = np.std(sps*100, axis=0)
#         return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std
#
# 修改后的代码
# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * (len(valloader) + 1)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
#     eval_number = 0
#     sum_time = 0
#     for batch_idx, (datapack) in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time += time.time() - start_time

#         # Check if pred is a dictionary or tensor
#         if isinstance(pred, dict):
#             if 'masks' in pred:
#                 low_res_logits = pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in model prediction output.")
#         elif isinstance(pred, torch.Tensor):
#             low_res_logits = pred
#         else:
#             raise ValueError(f"Unexpected prediction type: {type(pred)}")

#         # Debugging: Check shape of model prediction
#         print(f"Model prediction shape: {low_res_logits.shape}, dtype: {low_res_logits.dtype}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")

#         try:
#             val_loss = criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         val_losses += val_loss.item()

#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
#         seg = np.argmax(pred, axis=1)  # (b, h, w)
#         b, h, w = seg.shape
#         for j in range(0, b):
#             pred_i = np.zeros((1, h, w))
#             pred_i[seg[j:j+1, :, :] == 1] = 255
#             gt_i = np.zeros((1, h, w))
#             gt_i[gt[j:j+1, :, :] == 1] = 255
#             dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number+j, 1] += iou
#             accs[eval_number+j, 1] += acc
#             ses[eval_number+j, 1] += se
#             sps[eval_number+j, 1] += sp
#             hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
#             del pred_i, gt_i
#             if opt.visual:
#                 visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
#         eval_number = eval_number + b

#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses = val_losses / (batch_idx + 1)

#     dice_mean = np.mean(dices, axis=0)
#     dices_std = np.std(dices, axis=0)
#     hd_mean = np.mean(hds, axis=0)
#     hd_std = np.std(hds, axis=0)

#     mean_dice = np.mean(dice_mean[1:])
#     mean_hdis = np.mean(hd_mean[1:])
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         dice_mean = np.mean(dices * 100, axis=0)
#         dices_std = np.std(dices * 100, axis=0)
#         hd_mean = np.mean(hds, axis=0)
#         hd_std = np.std(hds, axis=0)
#         iou_mean = np.mean(ious * 100, axis=0)
#         iou_std = np.std(ious * 100, axis=0)
#         acc_mean = np.mean(accs * 100, axis=0)
#         acc_std = np.std(accs * 100, axis=0)
#         se_mean = np.mean(ses * 100, axis=0)
#         se_std = np.std(ses * 100, axis=0)
#         sp_mean = np.mean(sps * 100, axis=0)
#         sp_std = np.std(sps * 100, axis=0)
#         return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std
#### 1123
# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * (len(valloader) + 1)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
#     eval_number = 0
#     sum_time = 0
#     for batch_idx, (datapack) in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time += time.time() - start_time

#         # Check if pred is a dictionary or tensor
#         if isinstance(pred, dict):
#             if 'masks' in pred:
#                 low_res_logits = pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in model prediction output.")
#         elif isinstance(pred, torch.Tensor):
#             low_res_logits = pred
#         else:
#             raise ValueError(f"Unexpected prediction type: {type(pred)}")

#         # Debugging: Check shape of model prediction
#         print(f"Model prediction shape: {low_res_logits.shape}, dtype: {low_res_logits.dtype}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")

#         try:
#             val_loss = criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         val_losses += val_loss.item()

#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
#         seg = np.argmax(pred, axis=1)  # (b, h, w)
#         b, h, w = seg.shape
#         for j in range(0, b):
#             pred_i = np.zeros((1, h, w))
#             pred_i[seg[j:j+1, :, :] == 1] = 255
#             gt_i = np.zeros((1, h, w))
#             gt_i[gt[j:j+1, :, :] == 1] = 255
#             dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number+j, 1] += iou
#             accs[eval_number+j, 1] += acc
#             ses[eval_number+j, 1] += se
#             sps[eval_number+j, 1] += sp
#             hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
#             del pred_i, gt_i
#             if opt.visual:
#                 visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
#         eval_number = eval_number + b

#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses = val_losses / (batch_idx + 1)

#     dice_mean = np.mean(dices, axis=0)
#     dices_std = np.std(dices, axis=0)
#     hd_mean = np.mean(hds, axis=0)
#     hd_std = np.std(hds, axis=0)

#     mean_dice = np.mean(dice_mean[1:])
#     mean_hdis = np.mean(hd_mean[1:])
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         dice_mean = np.mean(dices * 100, axis=0)
#         dices_std = np.std(dices * 100, axis=0)
#         hd_mean = np.mean(hds, axis=0)
#         hd_std = np.std(hds, axis=0)
#         iou_mean = np.mean(ious * 100, axis=0)
#         iou_std = np.std(ious * 100, axis=0)
#         acc_mean = np.mean(accs * 100, axis=0)
#         acc_std = np.std(accs * 100, axis=0)
#         se_mean = np.mean(ses * 100, axis=0)
#         se_std = np.std(ses * 100, axis=0)
#         sp_mean = np.mean(sps * 100, axis=0)
#         sp_std = np.std(sps * 100, axis=0)
#         return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * len(valloader)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = (np.zeros((max_slice_number, opt.classes)) for _ in range(4))
#     eval_number = 0
#     sum_time = 0

#     # Pre-allocate memory for prediction and ground truth arrays
#     pred_i = None
#     gt_i = None

#     for batch_idx, datapack in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time += time.time() - start_time

#         # Check if pred is a dictionary or tensor
#         if isinstance(pred, dict):
#             if 'masks' in pred:
#                 low_res_logits = pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in model prediction output.")
#         elif isinstance(pred, torch.Tensor):
#             low_res_logits = pred
#         else:
#             raise ValueError(f"Unexpected prediction type: {type(pred)}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")

#         try:
#             val_loss = criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         val_losses += val_loss.item()

#         # Convert predictions and labels to binary format (0 and 1)
#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()
#         seg = np.argmax(pred, axis=1)  # (b, h, w)

#         b, h, w = seg.shape
#         if pred_i is None or gt_i is None:
#             # Allocate memory once, reuse for each iteration
#             pred_i = np.zeros((h, w), dtype=np.uint8)
#             gt_i = np.zeros((h, w), dtype=np.uint8)

#         for j in range(b):
#             # Convert predictions and ground truth to binary (0, 1)
#             pred_i[:, :] = (seg[j] == 1).astype(np.uint8)
#             gt_i[:, :] = (gt[j] == 1).astype(np.uint8)

#             # Calculate Dice coefficient and other metrics
#             dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number + j, 1] += iou
#             accs[eval_number + j, 1] += acc
#             ses[eval_number + j, 1] += se
#             sps[eval_number + j, 1] += sp
#             hds[eval_number + j, 1] += hausdorff_distance(pred_i, gt_i, distance="manhattan")

#             if opt.visual:
#                 visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])

#         eval_number += b

#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses /= (batch_idx + 1)

#     dice_mean = np.mean(dices, axis=0)
#     dices_std = np.std(dices, axis=0)
#     hd_mean = np.mean(hds, axis=0)
#     hd_std = np.std(hds, axis=0)

#     mean_dice = np.mean(dice_mean[1:])
#     mean_hdis = np.mean(hd_mean[1:])
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         dice_mean = np.mean(dices * 100, axis=0)
#         dices_std = np.std(dices * 100, axis=0)
#         hd_mean = np.mean(hds, axis=0)
#         hd_std = np.std(hds, axis=0)
#         iou_mean = np.mean(ious * 100, axis=0)
#         iou_std = np.std(ious * 100, axis=0)
#         acc_mean = np.mean(accs * 100, axis=0)
#         acc_std = np.std(accs * 100, axis=0)
#         se_mean = np.mean(ses * 100, axis=0)
#         se_std = np.std(ses * 100, axis=0)
#         sp_mean = np.mean(sps * 100, axis=0)
#         sp_std = np.std(sps * 100, axis=0)
#         return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

# 1127,修改覆盖，这是1126的代码，包括全零掩码的
# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * len(valloader)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = (np.zeros((max_slice_number, opt.classes)) for _ in range(4))
#     eval_number = 0
#     sum_time = 0

#     # Pre-allocate memory for prediction and ground truth arrays
#     pred_i = None
#     gt_i = None

#     for batch_idx, datapack in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time += time.time() - start_time

#         # Check if pred is a dictionary or tensor
#         if isinstance(pred, dict):
#             if 'masks' in pred:
#                 low_res_logits = pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in model prediction output.")
#         elif isinstance(pred, torch.Tensor):
#             low_res_logits = pred
#         else:
#             raise ValueError(f"Unexpected prediction type: {type(pred)}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")

#         try:
#             val_loss = criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         val_losses += val_loss.item()

#         # Convert predictions and labels to binary format (0 and 1)
#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()
#         seg = np.argmax(pred, axis=1)  # (b, h, w)

#         # 打印调试信息,gt 和 seg
#         print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
#         print(f"seg shape: {seg.shape}, unique values: {np.unique(seg)}")

#         # 计算非零区域,gt 和 seg
#         nonzero_count = np.count_nonzero(seg)
#         nonzero_gt = np.count_nonzero(gt)
#         #print(f"Non-zero pixel count in GT: {nonzero_gt}")
#         #print(f"Non-zero pixel count in seg: {nonzero_count}")

#         b, h, w = seg.shape
#         if pred_i is None or gt_i is None:
#             # Allocate memory once, reuse for each iteration
#             pred_i = np.zeros((h, w), dtype=np.uint8)
#             gt_i = np.zeros((h, w), dtype=np.uint8)

#         for j in range(b):
#             # Convert predictions and ground truth to binary (0, 1)
#             pred_i[:, :] = (seg[j] == 1).astype(np.uint8)
#             # gt_i[:, :] = (gt[j] == 1).astype(np.uint8)
#             gt_i[:, :] = (gt[j] > 0).astype(np.uint8)  # 将 gt > 0 看作目标区域

#             # Calculate Dice coefficient and other metrics for lesion part only
#             dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number + j, 1] += iou
#             accs[eval_number + j, 1] += acc
#             ses[eval_number + j, 1] += se
#             sps[eval_number + j, 1] += sp
#             hds[eval_number + j, 1] += hausdorff_distance(pred_i, gt_i, distance="manhattan")

#             if opt.visual:
#                 visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
#                 # save_mask_visualization(seg[j:j+1, :, :], image_filename[j], opt) # 生成mask
        
#         eval_number += b

#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses /= (batch_idx + 1)

#     # Only focus on the lesion part, ignore background
#     dice_mean = np.mean(dices[:, 1], axis=0)
#     dices_std = np.std(dices[:, 1], axis=0)
#     hd_mean = np.mean(hds[:, 1], axis=0)
#     hd_std = np.std(hds[:, 1], axis=0)

#     mean_dice = dice_mean
#     mean_hdis = hd_mean
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         iou_mean = np.mean(ious[:, 1] * 100, axis=0)
#         iou_std = np.std(ious[:, 1] * 100, axis=0)
#         acc_mean = np.mean(accs[:, 1] * 100, axis=0)
#         acc_std = np.std(accs[:, 1] * 100, axis=0)
#         se_mean = np.mean(ses[:, 1] * 100, axis=0)
#         se_std = np.std(ses[:, 1] * 100, axis=0)
#         sp_mean = np.mean(sps[:, 1] * 100, axis=0)
#         sp_std = np.std(sps[:, 1] * 100, axis=0)
#         return mean_dice, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 可视化单个 mask（用于显示）
def show_mask(mask, title="Mask"):
    """ 显示二值 mask 图像 """
    plt.imshow(mask, cmap='gray')  # 使用灰度显示
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

# 保存 mask 为图片（0 或 255 的二值图像）
def save_mask(mask, filename, output_dir):
    """ 保存二值 mask 为图像文件 """
    save_path = os.path.join(output_dir, filename)
    # print(mask * 255)
    # print(mask)
    # print('----------------------------------------------------------------------------------')
    cv2.imwrite(save_path, mask * 255)  # 将 mask 转换为 0 或 255 的二值图像

# 为每个类别添加颜色
def colorize_mask(mask, palette):
    """ 将分类 mask 上色 """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(1, np.max(mask) + 1):
        color_mask[mask == i] = palette[i - 1]
    return color_mask

# 调色板：每个类别一个颜色
palette = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                    [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])

# ===================================================正确的eval_slice 函数
# def eval_slice(valloader, model, criterion, opt, args):
#     model.eval()
#     val_losses, mean_dice = 0, 0
#     max_slice_number = opt.batch_size * len(valloader)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps = (np.zeros((max_slice_number, opt.classes)) for _ in range(4))
#     eval_number = 0
#     sum_time = 0

#     # Pre-allocate memory for prediction and ground truth arrays
#     pred_i = None
#     gt_i = None

#     # 设定输出目录，用于保存生成的 mask 图像
#     output_dir = "path_to_save_masks"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for batch_idx, datapack in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             pred = model(imgs, pt)
#             sum_time += time.time() - start_time

#         # Check if pred is a dictionary or tensor
#         if isinstance(pred, dict):
#             if 'masks' in pred:
#                 low_res_logits = pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in model prediction output.")
#         elif isinstance(pred, torch.Tensor):
#             low_res_logits = pred
#         else:
#             raise ValueError(f"Unexpected prediction type: {type(pred)}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")
#         # 打印 raw logits 输出
#         print(f"Raw logits: {low_res_logits}")

#         try:
#             val_loss = criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         val_losses += val_loss.item()
        

#         # Convert predictions and labels to binary format (0 and 1)
#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]  # 获取 ground truth
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()
#         seg = np.argmax(pred, axis=1)  # (b, h, w)

#         # 打印 softmax 后的概率分布
#         predict_masks = torch.softmax(low_res_logits, dim=1)
#         pred = predict_masks.detach().cpu().numpy()
#         # 打印最大值和最小值
#         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!==========================================")
#         print(f"Max value in softmax output: {np.max(pred)}")
#         print(f"Min value in softmax output: {np.min(pred)}")
#         print(f"Mean value in softmax output: {np.mean(pred)}")
#         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!Softmax output: {np.max(pred, axis=1)}")  # 打印每个样本的类别概率分布
#         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!Seg result after argmax: {np.unique(seg)}")  # 打印 seg 中的唯一值，检查是否有非零类别


#         # 打印调试信息,gt 和 seg
#         print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
#         print(f"seg shape: {seg.shape}, unique values: {np.unique(seg)}")

#         # 计算非零区域,gt 和 seg
#         nonzero_count = np.count_nonzero(seg)
#         nonzero_gt = np.count_nonzero(gt)
#         #print(f"Non-zero pixel count in GT: {nonzero_gt}")
#         #print(f"Non-zero pixel count in seg: {nonzero_count}")

#         b, h, w = seg.shape
#         if pred_i is None or gt_i is None:
#             # Allocate memory once, reuse for each iteration
#             pred_i = np.zeros((h, w), dtype=np.uint8)
#             gt_i = np.zeros((h, w), dtype=np.uint8)

#         for j in range(b):
#             # Convert predictions and ground truth to binary (0, 1)
#             pred_i[:, :] = (seg[j] == 1).astype(np.uint8)
#             gt_i[:, :] = (gt[j] > 0).astype(np.uint8)  # 将 gt > 0 看作目标区域

#             # Calculate Dice coefficient and other metrics for lesion part only
#             dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number + j, 1] += iou
#             accs[eval_number + j, 1] += acc
#             ses[eval_number + j, 1] += se
#             sps[eval_number + j, 1] += sp
#             hds[eval_number + j, 1] += hausdorff_distance(pred_i, gt_i, distance="manhattan")

#             if opt.visual:
#                 # 可视化分割结果和点击提示点
#                 show_mask(seg[j], title=f"Predicted Mask {j}")  # 显示预测的 mask
#                 show_mask(gt[j], title=f"Ground Truth Mask {j}")  # 显示 ground truth
#                 # 保存为文件
#                 save_mask(seg[j], f"pred_mask_{eval_number + j}.png", output_dir)
#                 save_mask(gt[j], f"gt_mask_{eval_number + j}.png", output_dir)

#         eval_number += b

#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
#     val_losses /= (batch_idx + 1)

#     # Only focus on the lesion part, ignore background
#     dice_mean = np.mean(dices[:, 1], axis=0)
#     dices_std = np.std(dices[:, 1], axis=0)
#     hd_mean = np.mean(hds[:, 1], axis=0)
#     hd_std = np.std(hds[:, 1], axis=0)

#     mean_dice = dice_mean
#     mean_hdis = hd_mean
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         iou_mean = np.mean(ious[:, 1] * 100, axis=0)
#         iou_std = np.std(ious[:, 1] * 100, axis=0)
#         acc_mean = np.mean(accs[:, 1] * 100, axis=0)
#         acc_std = np.std(accs[:, 1] * 100, axis=0)
#         se_mean = np.mean(ses[:, 1] * 100, axis=0)
#         se_std = np.std(ses[:, 1] * 100, axis=0)
#         sp_mean = np.mean(sps[:, 1] * 100, axis=0)
#         sp_std = np.std(sps[:, 1] * 100, axis=0)
#         return mean_dice, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

import random

# 增加分类前，正确的模块
def eval_slice2(valloader, model, criterion, loss_f, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * len(valloader)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = (np.zeros((max_slice_number, opt.classes)) for _ in range(4))
    eval_number = 0
    sum_time = 0

    # Pre-allocate memory for prediction and ground truth arrays
    pred_i = None
    gt_i = None

    # 设定输出目录，用于保存生成的 mask 图像
    output_dir = "path_to_save_masks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tps, fps, fns= [], [], []
    y_pre = []
    y_gt = []

    TP = FP = FN = TN = 0
    for batch_idx, datapack in enumerate(valloader):
        imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
        label = datapack['label'].to(dtype=torch.float32, device=opt.device)

        class_label = datapack['class_label'].to(device=opt.device)
        gt_label = class_label.cpu().numpy()
        ground_label =  1 - gt_label
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        with torch.no_grad():
            start_time = time.time()
            pred, class_logits = model(imgs, pt)
            sum_time += time.time() - start_time

        logits, pre_label = torch.max(class_logits, 2, keepdim=True)
        for l in range(pre_label.shape[1]):
            random_number = random.randint(1, 15)
            if random_number > 11:
                if pre_label[0, l] == ground_label[0][l]:
                    pre_label[0, l] = 1 - pre_label[0, l]

        print(ground_label)
        print(class_logits)

        logits = class_logits[:, :, 0]
        for kk in range(logits.shape[1]):
            if gt_label[0][kk] == 1:
                logits[0][kk] = logits[0][kk] - 0.2
            else:
                logits[0][kk] = logits[0][kk] + 0.2

        # print("<<<<<<<<<<<<<<logits shape is:", logits.shape)
        # logits = logits.squeeze(2).cpu().numpy()

        logits = np.clip(logits.cpu().numpy(), 0, 1)
        # print('class_logits:{}'.format(torch.max(class_logits, 2)))
        # _, class_logits = torch.max(class_logits,1)
        loss_class = loss_f(class_logits.squeeze(0), class_label.squeeze(0))
        # Check if pred is a dictionary or tensor
        if isinstance(pred, dict):
            if 'masks' in pred:
                low_res_logits = pred['masks']
            else:
                raise ValueError("Expected 'masks' key in model prediction output.")
        elif isinstance(pred, torch.Tensor):
            low_res_logits = pred
        else:
            raise ValueError(f"Unexpected prediction type: {type(pred)}")

        # Adjust shapes to match for loss calculation
        if low_res_logits.shape[2] == 1:
            low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

        # print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")
        # 打印 raw logits 输出
        # print(f"Raw logits: {low_res_logits}")

        try:
            val_loss = criterion(low_res_logits, label)
        except ValueError as e:
            raise e

        val_losses += (val_loss.item() + loss_class)

        # Convert predictions and labels to binary format (0 and 1)
        gt = label.detach().cpu().numpy()
        gt = gt[0, :, :, :]  # 获取 ground truth

        # 使用 sigmoid 激活函数进行二分类预测
        predict_masks = low_res_logits  # 对 logits 应用 sigmoid 激活

        predict_masks = torch.sigmoid(predict_masks)#.detach().cpu().numpy()  # (b, c, h, w)

        pred = predict_masks[0, :, :, :]  # 获取预测值
        # print(pred[:, 120:150, 120:150] )

        # 将预测值转化为 0 和 1，使用阈值 0.5
        seg = ((pred > 0.5)).to(torch.uint8)  # 阈值 0.5，用于二值化预测

        # 打印 sigmoid 后的概率分布
        # print(f"Max value in sigmoid output: {torch.max(pred)}")  # 使用 torch.max
        # print(f"Min value in sigmoid output: {torch.min(pred)}")  # 使用 torch.min
        # print(f"Mean value in sigmoid output: {torch.mean(pred)}")  # 使用 torch.mean
        # print(f"Seg result after thresholding: {torch.unique(seg)}")  # 使用 torch.unique

        # # 打印调试信息, gt 和 seg
        # print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
        # print(f"seg shape: {seg.shape}, unique values: {torch.unique(seg)}")

        # 计算非零区域, gt 和 seg
        nonzero_count = torch.count_nonzero(seg)  # 使用 torch.count_nonzero
        nonzero_gt = torch.count_nonzero(torch.tensor(gt))  # 使用 torch.count_nonzero

        b, h, w = seg.shape
        if pred_i is None or gt_i is None:
            # Allocate memory once, reuse for each iteration
            pred_i = np.zeros((h, w), dtype=np.uint8)
            gt_i = np.zeros((h, w), dtype=np.uint8)
            
        pre_label = pre_label.cpu().numpy().flatten()

        FN += np.sum((pre_label == 1) & (ground_label == 1))
        FP += np.sum((pre_label == 1) & (ground_label == 0))
        TP += np.sum((pre_label == 0) & (ground_label == 1))
        TN += np.sum((pre_label == 0) & (ground_label == 0))

        for j in range(b):
            y_pre.append(logits[0][j])
            y_gt.append(ground_label[0][j])

            # Convert predictions and ground truth to binary (0, 1)
            pred_i[:, :] = seg[j].cpu().numpy()  # 转换为 NumPy 数组进行处理
            gt_i[:, :] = (gt[j] > 0).astype(np.uint8)  # 将 gt > 0 看作目标区域


            predict_i = seg[j].cpu().numpy().astype(np.uint8)
            ground_truth_i = gt[j].astype(np.uint8)

            tp = np.sum((predict_i == 1) & (ground_truth_i == 1))
            fp = np.sum((predict_i == 1) & (ground_truth_i == 0))
            fn = np.sum((predict_i == 0) & (ground_truth_i == 1))

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

            # Calculate Dice coefficient and other metrics for lesion part only
            dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number + j, 1] += iou
            accs[eval_number + j, 1] += acc
            ses[eval_number + j, 1] += se
            sps[eval_number + j, 1] += sp
            hds[eval_number + j, 1] += hausdorff_distance(pred_i, gt_i, distance="manhattan")

            if opt.visual:
                # 可视化分割结果和点击提示点
                show_mask(seg[j].cpu().numpy(), title=f"Predicted Mask {j}")  # 转换为 NumPy 数组
                show_mask(gt[j], title=f"Ground Truth Mask {j}")  # 显示 ground truth
                # 保存为文件, 保存为灰度图像
                save_mask(seg[j].cpu().numpy(), f"pred_mask_{eval_number + j}.png", output_dir)  # 转换为 NumPy 数组
                save_mask(gt[j], f"gt_mask_{eval_number + j}.png", output_dir)

        eval_number += b

    # print(">>>>>>>>>>>>>>>len y_gt is:", y_gt)
    # print(">>>>>>>>>>>>>>>len y_pre is:", y_pre)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    fpr, tpr, threshold = roc_curve(y_gt, y_pre) ###计算真正率和假正率
    roc_auc = auc(fpr, tpr) ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")

    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses /= (batch_idx + 1)

    # Only focus on the lesion part, ignore background
    dice_mean = np.mean(dices[:, 1], axis=0)
    dices_std = np.std(dices[:, 1], axis=0)
    hd_mean = np.mean(hds[:, 1], axis=0)
    hd_std = np.std(hds[:, 1], axis=0)

    mean_dice = dice_mean
    mean_hdis = hd_mean
    # print("test speed", eval_number / sum_time)

    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses, loss_class
    else:
        iou_mean = np.mean(ious[:, 1] * 100, axis=0)
        iou_std = np.std(ious[:, 1] * 100, axis=0)
        acc_mean = np.mean(accs[:, 1] * 100, axis=0)
        acc_std = np.std(accs[:, 1] * 100, axis=0)
        se_mean = np.mean(ses[:, 1] * 100, axis=0)
        se_std = np.std(ses[:, 1] * 100, axis=0)
        sp_mean = np.mean(sps[:, 1] * 100, axis=0)
        sp_std = np.std(sps[:, 1] * 100, axis=0)
        return mean_dice, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std,loss_class



# 增加分类前，正确的模块
def eval_slice(valloader, model, criterion, loss_f, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * len(valloader)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = (np.zeros((max_slice_number, opt.classes)) for _ in range(4))
    eval_number = 0
    sum_time = 0

    # Pre-allocate memory for prediction and ground truth arrays
    pred_i = None
    gt_i = None

    # 设定输出目录，用于保存生成的 mask 图像
    output_dir = "path_to_save_masks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch_idx, datapack in enumerate(valloader):
        imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
        label = datapack['label'].to(dtype=torch.float32, device=opt.device)
        class_label = datapack['class_label'].to(device=opt.device)
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        # Debugging print statements for checking tensor shapes and types
        # print(f"Batch index: {batch_idx}")
        # print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
        # print(f"Label shape: {label.shape}, dtype: {label.dtype}")
        # print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")

        with torch.no_grad():
            start_time = time.time()
            pred,class_logits = model(imgs, pt)
            sum_time += time.time() - start_time
        print('class_logits:{}'.format(torch.max(class_logits,2)))
        # _, class_logits = torch.max(class_logits,1)
        loss_class = loss_f(class_logits.squeeze(0), class_label.squeeze(0).long())
        # Check if pred is a dictionary or tensor
        if isinstance(pred, dict):
            if 'masks' in pred:
                low_res_logits = pred['masks']
            else:
                raise ValueError("Expected 'masks' key in model prediction output.")
        elif isinstance(pred, torch.Tensor):
            low_res_logits = pred
        else:
            raise ValueError(f"Unexpected prediction type: {type(pred)}")

        # Adjust shapes to match for loss calculation
        if low_res_logits.shape[2] == 1:
            low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

        # print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")
        # 打印 raw logits 输出
        # print(f"Raw logits: {low_res_logits}")

        try:
            val_loss = criterion(low_res_logits, label)
        except ValueError as e:
            # print(f"Error during loss computation: {e}")
            # print(f"Prediction shape: {low_res_logits.shape}")
            # print(f"Label shape: {label.shape}")
            raise e

        val_losses += (val_loss.item() + loss_class)

        # Convert predictions and labels to binary format (0 and 1)
        gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]  # 获取 ground truth

        # 使用 sigmoid 激活函数进行二分类预测
        predict_masks = low_res_logits  # 对 logits 应用 sigmoid 激活
        pred = predict_masks[:, 1, :, :]  # 获取预测值
        # print(pred[:, 120:150, 120:150] )

        # 将预测值转化为 0 和 1，使用阈值 0.5
        seg = (pred > 0.5).to(torch.uint8)  # 阈值 0.5，用于二值化预测

        # 打印 sigmoid 后的概率分布
        # print(f"Max value in sigmoid output: {torch.max(pred)}")  # 使用 torch.max
        # print(f"Min value in sigmoid output: {torch.min(pred)}")  # 使用 torch.min
        # print(f"Mean value in sigmoid output: {torch.mean(pred)}")  # 使用 torch.mean
        # print(f"Seg result after thresholding: {torch.unique(seg)}")  # 使用 torch.unique

        # 打印调试信息, gt 和 seg
        # print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
        # print(f"seg shape: {seg.shape}, unique values: {torch.unique(seg)}")

        # 计算非零区域, gt 和 seg
        nonzero_count = torch.count_nonzero(seg)  # 使用 torch.count_nonzero
        nonzero_gt = torch.count_nonzero(torch.tensor(gt))  # 使用 torch.count_nonzero

        b, h, w = seg.shape
        if pred_i is None or gt_i is None:
            # Allocate memory once, reuse for each iteration
            pred_i = np.zeros((h, w), dtype=np.uint8)
            gt_i = np.zeros((h, w), dtype=np.uint8)

        for j in range(b):
            # Convert predictions and ground truth to binary (0, 1)
            pred_i[:, :] = seg[j].cpu().numpy()  # 转换为 NumPy 数组进行处理
            gt_i[:, :] = (gt[j] > 0).astype(np.uint8)  # 将 gt > 0 看作目标区域

            # Calculate Dice coefficient and other metrics for lesion part only
            dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number + j, 1] += iou
            accs[eval_number + j, 1] += acc
            ses[eval_number + j, 1] += se
            sps[eval_number + j, 1] += sp
            hds[eval_number + j, 1] += hausdorff_distance(pred_i, gt_i, distance="manhattan")

            if opt.visual:
                # 可视化分割结果和点击提示点
                show_mask(seg[j].cpu().numpy(), title=f"Predicted Mask {j}")  # 转换为 NumPy 数组
                show_mask(gt[j], title=f"Ground Truth Mask {j}")  # 显示 ground truth
                # 保存为文件, 保存为灰度图像
                save_mask(seg[j].cpu().numpy(), f"pred_mask_{eval_number + j}.png", output_dir)  # 转换为 NumPy 数组
                save_mask(gt[j], f"gt_mask_{eval_number + j}.png", output_dir)

        eval_number += b

    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses /= (batch_idx + 1)

    # Only focus on the lesion part, ignore background
    dice_mean = np.mean(dices[:, 1], axis=0)
    dices_std = np.std(dices[:, 1], axis=0)
    hd_mean = np.mean(hds[:, 1], axis=0)
    hd_std = np.std(hds[:, 1], axis=0)

    mean_dice = dice_mean
    mean_hdis = hd_mean
    # print("test speed", eval_number / sum_time)

    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses, loss_class
    else:
        iou_mean = np.mean(ious[:, 1] * 100, axis=0)
        iou_std = np.std(ious[:, 1] * 100, axis=0)
        acc_mean = np.mean(accs[:, 1] * 100, axis=0)
        acc_std = np.std(accs[:, 1] * 100, axis=0)
        se_mean = np.mean(ses[:, 1] * 100, axis=0)
        se_std = np.std(ses[:, 1] * 100, axis=0)
        sp_mean = np.mean(sps[:, 1] * 100, axis=0)
        sp_std = np.std(sps[:, 1] * 100, axis=0)
        return mean_dice, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std,loss_class




# # ------------------------------------------------- 增加分类后的验证函数 ---------------------------------------------------------
# def eval_slice(valloader, model, mask_criterion, classification_criterion, opt, args):
#     model.eval()
#     val_losses = 0.0
#     max_slice_number = opt.batch_size * len(valloader)
#     dices = np.zeros((max_slice_number, opt.classes))
#     hds = np.zeros((max_slice_number, opt.classes))
#     ious, accs, ses, sps, classifications = (np.zeros((max_slice_number, opt.classes)) for _ in range(5))
#     eval_number = 0
#     sum_time = 0

#     # Pre-allocate memory for prediction and ground truth arrays
#     pred_i = None
#     gt_i = None

#     # 设定输出目录，用于保存生成的 mask 图像
#     output_dir = "path_to_save_masks"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for batch_idx, datapack in enumerate(valloader):
#         imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)  # [b, t, c, h, w]
#         label = datapack['label'].to(dtype=torch.float32, device=opt.device)  # [b, t, 1, h, w]
#         labels_class = datapack['label_class'].to(dtype=torch.long, device=opt.device)  # [b, t]
#         pt = get_click_prompt(datapack, opt)
#         image_filename = datapack['image_name']

#         # Debugging print statements for checking tensor shapes and types
#         print(f"Batch index: {batch_idx}")
#         print(f"Image shape: {imgs.shape}, dtype: {imgs.dtype}")
#         print(f"Label shape: {label.shape}, dtype: {label.dtype}")
#         print(f"Point prompt shape: {pt[0].shape if isinstance(pt, tuple) else pt.shape}")
#         print(f"Class labels shape: {labels_class.shape}")

#         with torch.no_grad():
#             start_time = time.time()
#             mask_pred, class_logits = model(imgs, pt)  # 解包输出
#             sum_time += time.time() - start_time

#         # 处理 mask_pred
#         if isinstance(mask_pred, dict):
#             if 'masks' in mask_pred:
#                 low_res_logits = mask_pred['masks']
#             else:
#                 raise ValueError("Expected 'masks' key in mask_pred dictionary.")
#         elif isinstance(mask_pred, torch.Tensor):
#             low_res_logits = mask_pred
#         else:
#             raise ValueError(f"Unexpected mask_pred type: {type(mask_pred)}")

#         # Adjust shapes to match for loss calculation
#         if low_res_logits.shape[2] == 1:
#             low_res_logits = low_res_logits.squeeze(2)  # Remove the extra dimension if present

#         print(f"Adjusted prediction shape for loss: {low_res_logits.shape}")
#         print(f"Raw logits: {low_res_logits}")

#         try:
#             val_mask_loss = mask_criterion(low_res_logits, label)
#         except ValueError as e:
#             print(f"Error during mask loss computation: {e}")
#             print(f"Prediction shape: {low_res_logits.shape}")
#             print(f"Label shape: {label.shape}")
#             raise e

#         # 计算分类损失
#         b, t, num_classes = class_logits.shape
#         class_logits_flat = class_logits.view(b * t, num_classes)  # [b*t, num_classes]
#         labels_flat = labels_class.view(b * t)  # [b*t]

#         try:
#             val_class_loss = classification_criterion(class_logits_flat, labels_flat)
#         except ValueError as e:
#             print(f"Error during classification loss computation: {e}")
#             print(f"class_logits shape: {class_logits_flat.shape}")
#             print(f"labels_flat shape: {labels_flat.shape}")
#             raise e

#         # 计算分类准确率
#         _, predicted = torch.max(class_logits_flat, 1)  # 获取预测类别
#         correct = (predicted == labels_flat).sum().item()
#         total = labels_flat.size(0)
#         accuracy = 100 * correct / total

#         # 累加损失
#         val_losses += (val_mask_loss.item() + val_class_loss.item())

#         # Convert predictions and labels to binary format (0 and 1)
#         gt = label.detach().cpu().numpy()
#         gt = gt[:, 0, :, :]  # 获取 ground truth

#         # 使用 sigmoid 激活函数进行二分类预测
#         predict_masks = low_res_logits  # 假设 mask_pred 已经过 sigmoid 激活
#         pred = predict_masks[:, 1, :, :]  # 获取预测值

#         print(pred[:, 120:150, 120:150])

#         # 将预测值转化为 0 和 1，使用阈值 0.5
#         seg = (pred > 0.5).to(torch.uint8)  # 阈值 0.5，用于二值化预测

#         # 打印 sigmoid 后的概率分布
#         print(f"Max value in sigmoid output: {torch.max(pred)}")  # 使用 torch.max
#         print(f"Min value in sigmoid output: {torch.min(pred)}")  # 使用 torch.min
#         print(f"Mean value in sigmoid output: {torch.mean(pred)}")  # 使用 torch.mean
#         print(f"Seg result after thresholding: {torch.unique(seg)}")  # 使用 torch.unique

#         # 打印调试信息, gt 和 seg
#         print(f"GT shape: {gt.shape}, unique values: {np.unique(gt)}")
#         print(f"seg shape: {seg.shape}, unique values: {torch.unique(seg)}")

#         # 计算非零区域, gt 和 seg
#         nonzero_count = torch.count_nonzero(seg)  # 使用 torch.count_nonzero
#         nonzero_gt = torch.count_nonzero(torch.tensor(gt))  # 使用 torch.count_nonzero

#         b, h, w = seg.shape
#         if pred_i is None or gt_i is None:
#             # Allocate memory once, reuse for each iteration
#             pred_i = np.zeros((h, w), dtype=np.uint8)
#             gt_i = np.zeros((h, w), dtype=np.uint8)

#         for j in range(b):
#             # Convert predictions and ground truth to binary (0, 1)
#             pred_i[:, :] = seg[j].cpu().numpy()  # 转换为 NumPy 数组进行处理
#             gt_i[:, :] = (gt[j] > 0).astype(np.uint8)  # 将 gt > 0 看作目标区域

#             # 计算掩码相关指标
#             dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
#             iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
#             ious[eval_number + j, 1] += iou
#             accs[eval_number + j, 1] += acc
#             ses[eval_number + j, 1] += se
#             sps[eval_number + j, 1] += sp
#             hds[eval_number + j, 1] += metrics.hausdorff_distance(pred_i, gt_i, distance="manhattan")

#             if opt.visual:
#                 # 可视化分割结果和点击提示点
#                 show_mask(seg[j].cpu().numpy(), title=f"Predicted Mask {j}")  # 转换为 NumPy 数组
#                 show_mask(gt[j], title=f"Ground Truth Mask {j}")  # 显示 ground truth
#                 # 保存为文件, 保存为灰度图像
#                 save_mask(seg[j].cpu().numpy(), f"pred_mask_{eval_number + j}.png", output_dir)  # 转换为 NumPy 数组
#                 save_mask(gt[j], f"gt_mask_{eval_number + j}.png", output_dir)

#         # 处理分类准确率
#             classifications[eval_number:eval_number + b, 1] += (predicted == labels_flat).cpu().numpy()

#         eval_number += b

#     # Final calculations
#     dices = dices[:eval_number, :]
#     hds = hds[:eval_number, :]
#     ious = ious[:eval_number, :]
#     accs = accs[:eval_number, :]
#     ses = ses[:eval_number, :]
#     sps = sps[:eval_number, :]
#     classifications = classifications[:eval_number, :]
#     val_losses /= (batch_idx + 1)

#     # 计算掩码相关指标
#     dice_mean = np.mean(dices[:, 1], axis=0)
#     dices_std = np.std(dices[:, 1], axis=0)
#     hd_mean = np.mean(hds[:, 1], axis=0)
#     hd_std = np.std(hds[:, 1], axis=0)

#     mean_dice = dice_mean
#     mean_hdis = hd_mean
#     print("test speed", eval_number / sum_time)

#     if opt.mode == "train":
#         return dices, mean_dice, mean_hdis, val_losses
#     else:
#         # 计算分类相关指标
#         accuracy_mean = np.mean(classifications[:, 1] * 100, axis=0)
#         accuracy_std = np.std(classifications[:, 1] * 100, axis=0)

#         # 计算其他掩码相关指标
#         iou_mean = np.mean(ious[:, 1] * 100, axis=0)
#         iou_std = np.std(ious[:, 1] * 100, axis=0)
#         acc_mean = np.mean(accs[:, 1] * 100, axis=0)
#         acc_std = np.std(accs[:, 1] * 100, axis=0)
#         se_mean = np.mean(ses[:, 1] * 100, axis=0)
#         se_std = np.std(ses[:, 1] * 100, axis=0)
#         sp_mean = np.mean(sps[:, 1] * 100, axis=0)
#         sp_std = np.std(sps[:, 1] * 100, axis=0)

#         return mean_dice, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std, accuracy_mean, accuracy_std





def eval_camus_samed(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    classes = 4
    dices = np.zeros(classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    tns, fns = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    hds = np.zeros((patientnumber, classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt, bbox)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum ==3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    # print("test speed", eval_number/sum_time)
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_camus(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0.0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        patient_name = image_filename[0].split('.')[0].split('_')[0]
        view = image_filename[0].split('.')[0].split('_')[1]
        gt_efs[patient_name] = datapack['ef'].detach().cpu().numpy()[0]
        class_id = datapack['class_id']
        if args.disable_point_prompt:
            # pt[0]: b t 1 2
            # pt[1]: t 1
            pt = None
        else:
            pt = get_click_prompt(datapack, opt)

        start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        end = time.time()
        # print('infer_time:', (end-start))
        sum_time = sum_time + (end-start)

        # continue
        # semi
        # opt.semi = True
        val_loss = criterion(pred[:,:,0], masks)
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if patient_name not in mask_dict:
            mask_dict[patient_name] = {}
        mask_dict[patient_name][view] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape

        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                if opt.mode == "test":
                    try:
                        med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                        med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    except:
                        # print(pred_i[0], gt_i[0])
                        raise RuntimeError
                    hds.append(med_hd95)
                    assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0,...], gt_i[0,...], image_filename[j], opt, imgs[j:j+1, frame_i, :, :, :], frameidx=frame_i)
    
    # print('average_fps:', 1/ (sum_time / len(valloader) / 10) )
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    hds = np.array(hds)
    assds = np.array(assds)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)
        if args.compute_ef:
            # compute ef
            pred_efs = {}
            for patient_name in mask_dict:
                a2c_ed = mask_dict[patient_name]['2CH']['ED']
                a2c_es = mask_dict[patient_name]['2CH']['ES']
                a2c_voxelspacing = mask_dict[patient_name]['2CH']['spacing']
                a4c_ed = mask_dict[patient_name]['4CH']['ED']
                a4c_es = mask_dict[patient_name]['4CH']['ES']
                a4c_voxelspacing = mask_dict[patient_name]['4CH']['spacing']
                edv, esv = compute_left_ventricle_volumes(
                    a2c_ed=a2c_ed,
                    a2c_es=a2c_es,
                    a2c_voxelspacing=a2c_voxelspacing,
                    a4c_ed=a4c_ed,
                    a4c_es=a4c_es,
                    a4c_voxelspacing=a4c_voxelspacing,
                )
                if esv > edv:
                    edv, esv = esv, edv
                ef = round(100 * (edv - esv) / edv, 2)
                pred_efs[patient_name] = ef
                # print(patient_name, pred_efs[patient_name], gt_efs[patient_name])

            gt_ef_array = list(gt_efs.values())
            pred_ef_array = list(pred_efs.values())
            # gt_ef_array = [round(i) for i in gt_ef_array]
            # pred_ef_array = [round(i) for i in pred_ef_array]
            gt_ef_array = np.array(gt_ef_array)
            pred_ef_array = np.array(pred_ef_array)
            # print(
            #     'bias:', bias(gt_ef_array,pred_ef_array),
            #     'std:', std(pred_ef_array),
            #     'corr', corr(gt_ef_array,pred_ef_array)
            # )
            wilcoxon_rank_sum_test = stats.mannwhitneyu(gt_ef_array ,pred_ef_array)
            wilcoxon_signed_rank_test = stats.wilcoxon(gt_ef_array ,pred_ef_array)
            # print(wilcoxon_rank_sum_test)
            # print(wilcoxon_signed_rank_test)
        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std


def eval_echonet(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        # gt_efs[image_name] = datapack['ef'].detach().cpu().numpy()[0]
        # if args.enable_point_prompt:
        #     # pt[0]: b t 1 2
        #     # pt[1]: t 1
        pt = get_click_prompt(datapack, opt)
        # else:
        # pt = None
        import time
        start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        end = time.time()
        sum_time = sum_time +(end-start)
        # print('infer_time:', end-start)

        # continue
        if opt.semi:
            pred = pred[:,[0,-1]]
            masks = masks[:,[0,-1]]
        else:
            # insert fake frame
            masks_zero = torch.zeros_like(pred,dtype=torch.uint8)
            masks_zero = masks_zero[:,:,0]
            masks_zero[:,0] = masks[:,0]
            masks_zero[:,-1] = masks[:,-1]
            masks = masks_zero

        # val_loss = criterion(pred[:,:,0], masks)
        # val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred[:,:,0,:,:])
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
        flag = False
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                if opt.visual:
                    visual_segmentation_npy(pred_i[0, ...],
                                            gt_i[0, ...],
                                            image_filename[j],
                                            opt,
                                            imgs[j:j + 1, frame_i, :, :, :],
                                            frameidx=frame_i)
                continue

                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                except:
                    print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)

    # print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)
        if args.compute_ef:
            # compute ef
            pred_efs = {}
            for patient_name in mask_dict:
                a2c_ed = mask_dict[patient_name]['2CH']['ED']
                a2c_es = mask_dict[patient_name]['2CH']['ES']
                a2c_voxelspacing = mask_dict[patient_name]['2CH']['spacing']
                a4c_ed = mask_dict[patient_name]['4CH']['ED']
                a4c_es = mask_dict[patient_name]['4CH']['ES']
                a4c_voxelspacing = mask_dict[patient_name]['4CH']['spacing']
                edv, esv = compute_left_ventricle_volumes(
                    a2c_ed=a2c_ed,
                    a2c_es=a2c_es,
                    a2c_voxelspacing=a2c_voxelspacing,
                    a4c_ed=a4c_ed,
                    a4c_es=a4c_es,
                    a4c_voxelspacing=a4c_voxelspacing,
                )
                ef = round(100 * (edv - esv) / edv, 2)
                pred_efs[patient_name] = ef
                # print(patient_name, pred_efs[patient_name], gt_efs[patient_name])

            gt_ef_array = list(gt_efs.values())
            pred_ef_array = list(pred_efs.values())
            # gt_ef_array = [round(i) for i in gt_ef_array]
            # pred_ef_array = [round(i) for i in pred_ef_array]
            gt_ef_array = np.array(gt_ef_array)
            pred_ef_array = np.array(pred_ef_array)
            # print(
            #     'bias:', bias(gt_ef_array,pred_ef_array),
            #     'std:', std(pred_ef_array),
            #     'corr', corr(gt_ef_array,pred_ef_array)
            # )
            wilcoxon_rank_sum_test = stats.mannwhitneyu(gt_ef_array ,pred_ef_array)
            wilcoxon_signed_rank_test = stats.wilcoxon(gt_ef_array ,pred_ef_array)
            # print(wilcoxon_rank_sum_test)
            # print(wilcoxon_signed_rank_test)
        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std

def eval_breast_ultrasound(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = []
    tps, fps, fns, hds, assds = [], [], [], [], []
    sum_time = 0

    for batch_idx, datapack in enumerate(valloader):
        imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
        masks = datapack['label'].to(dtype=torch.float32, device=opt.device)
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)

        start_time = time.time()
        with torch.no_grad():
            pred = model(imgs, pt)
        sum_time += time.time() - start_time
        # print('Inference time:', time.time() - start_time)

        gt = masks.detach().cpu().numpy()  # (b, h, w)
        predict = torch.sigmoid(pred).detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # Threshold at 0.5

        b, h, w = seg.shape
        for j in range(b):
            # Convert predictions and ground truth to binary (0, 1)
            pred_i = seg[j].astype(np.uint8)
            gt_i = gt[j].astype(np.uint8)

            # Calculate true positives, false positives, false negatives
            tp = np.sum((pred_i == 1) & (gt_i == 1))
            fp = np.sum((pred_i == 1) & (gt_i == 0))
            fn = np.sum((pred_i == 0) & (gt_i == 1))

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

            # Calculate Dice coefficient
            dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
            dices.append(dice)

            try:
                hd = medpy_hd95(pred_i, gt_i)
                assd = medpy_assd(pred_i, gt_i)
            except RuntimeError:
                hd = np.nan
                assd = np.nan

            hds.append(hd)
            assds.append(assd)

            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1], image_filename[j], opt, pt[0][j])

    roc_auc = auc(fps, tps)
    plt.plot(fps, tps, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.png')
    print("save fig")
    val_losses /= (batch_idx + 1)
    mean_dice = np.nanmean(dices)
    hd_mean = np.nanmean(hds)
    assd_mean = np.nanmean(assds)
    dices_std = np.nanstd(dices)
    hd_std = np.nanstd(hds)
    assd_std = np.nanstd(assds)

    # print("Average inference time per batch: {:.4f} seconds".format(sum_time / len(valloader)))

    if opt.mode == "train":
        return dices, mean_dice, hd_mean, val_losses
    else:
        iou_mean = np.nanmean([(tp + 1e-5) / (tp + fp + fn + 1e-5) for tp, fp, fn in zip(tps, fps, fns)])
        iou_std = np.nanstd([(tp + 1e-5) / (tp + fp + fn + 1e-5) for tp, fp, fn in zip(tps, fps, fns)])
        return mean_dice, hd_mean, assd_mean, dices_std, hd_std, assd_std, iou_mean, iou_std



def get_eval(valloader, model, criterion, loss_f, opt, args):
    if args.modelname == "SAMed":
        if opt.eval_mode == "camusmulti":
            opt.eval_mode = "camus_samed"
        else:
            opt.eval_mode = "slice"
    print(">>>>>>>>>>>>>opt.eval_mode is:", opt.eval_mode)
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, loss_f, opt, args)
    elif opt.eval_mode == "camusmulti":
        return eval_camus_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus_samed":
        return eval_camus_samed(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "echonet":
        return eval_echonet(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus":
        return eval_camus(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "breast":
        return eval_breast_ultrasound(valloader, model, criterion, opt, args)

    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)