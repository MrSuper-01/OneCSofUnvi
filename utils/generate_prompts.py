import torch
import random

def generate_click_prompt(img, msk, pt_label = 1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

# def get_click_prompt(datapack, opt):
#     if 'pt' not in datapack:
#         imgs, pt, masks = generate_click_prompt(imgs, masks)
#     else:
#         pt = datapack['pt']
#         point_labels = datapack['p_label']
#
#     point_coords = pt
#     coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=opt.device)
#     if isinstance(point_labels[0], torch.Tensor):
#         point_labels = [point_label.tolist()[0] for point_label in point_labels]
#     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=opt.device)
#     labels_torch = labels_torch.view(-1,1)
#     if len(pt.shape) == 2:
#         coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#     pt = (coords_torch, labels_torch)
#     return pt

# def get_click_prompt(datapack, opt):
#     if 'pt' not in datapack:
#         imgs, pt, masks = generate_click_prompt(datapack['image'], datapack['label'])
#     else:
#         pt = datapack['pt']
#         point_labels = datapack['p_label']
#
#     # 打印 point_labels 的内容和类型
#     print("point_labels before conversion:", point_labels)
#     print("type of point_labels:", type(point_labels))
#
#     # 将 point_labels 中的张量转换为 Python 数值
#     point_labels_flat = []
#     for p_label in point_labels:
#         if isinstance(p_label, list):
#             # 处理嵌套的列表情况
#             point_labels_flat.extend([int(val.item()) for val in p_label])
#         elif isinstance(p_label, torch.Tensor):
#             point_labels_flat.append(int(p_label.item()))
#         else:
#             point_labels_flat.append(int(p_label))
#
#     # 打印 point_labels_flat 的内容和类型，在进一步处理之前
#     print("point_labels after flattening:", point_labels_flat)
#     print("type of point_labels_flat:", type(point_labels_flat))
#
#     point_coords = pt
#     coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=opt.device)
#     labels_torch = torch.as_tensor(point_labels_flat, dtype=torch.int, device=opt.device)
#     labels_torch = labels_torch.view(-1, 1)
#
#     if len(pt.shape) == 2:
#         coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#
#     pt = (coords_torch, labels_torch)
#     return pt


def get_click_prompt(datapack, opt):
    if 'pt' not in datapack:
        imgs, pt, masks = generate_click_prompt(datapack['image'], datapack['label'])
    else:
        pt = datapack['pt']
        point_labels = datapack['p_label']

    # 打印 point_labels 的内容和类型
    # print("point_labels before conversion:", point_labels)
    # print("type of point_labels:", type(point_labels))

    # 将 point_labels 中的张量转换为 Python 数值
    point_labels_flat = []
    for p_label in point_labels:
        if isinstance(p_label, list):
            # 处理嵌套的列表情况，并且只取张量中的第一个值
            point_labels_flat.extend([int(val[0].item()) for val in p_label])
        elif isinstance(p_label, torch.Tensor):
            point_labels_flat.append(int(p_label.item()))
        else:
            point_labels_flat.append(int(p_label))

    # 打印 point_labels_flat 的内容和类型，在进一步处理之前
    # print("point_labels after flattening:", point_labels_flat)
    # print("type of point_labels_flat:", type(point_labels_flat))

    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=opt.device)
    labels_torch = torch.as_tensor(point_labels_flat, dtype=torch.int, device=opt.device)
    labels_torch = labels_torch.view(-1, 1)

    if len(pt.shape) == 2:
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

    pt = (coords_torch, labels_torch)
    return pt

