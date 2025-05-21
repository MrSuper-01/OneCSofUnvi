# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# from turtle import shape
# import torch
# import numpy as np
# from torch import nn
# from torch.nn import functional as F
# from torchvision import models

# from typing import Any, Dict, List, Tuple

# from .image_encoder import ImageEncoderViT
# from .mask_decoder import MaskDecoder
# from .prompt_encoder import PromptEncoder
# from .mem import Mem
# from einops import rearrange
# import torchvision.models as models
# from timm.models.layers import trunc_normal_



# # ------------------------------------------------------------- RESNET分类--------------------------------------------
# # class ClassificationHead(nn.Module):
# #     def __init__(self, in_features, num_classes=2):
# #         super(ClassificationHead, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(in_features, 512),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(512, num_classes)
# #         )
    
# #     def forward(self, x):
# #         x = torch.flatten(x, 1)  # [batch, features]
# #         x = self.fc(x)
# #         return x


# # class ResNetClassifier(nn.Module):
# #     def __init__(self, in_channels: int, num_classes: int):
# #         super(ResNetClassifier, self).__init__()
# #         self.classifier = models.resnet18(pretrained=True)
    
# #     def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
# #         out = self.classifier(combined_features)
# #         print('resnet:{}'.format(np.shape(out)))
# #         return out

# class BN_Conv2d(nn.Module):
#     """
#     BN_CONV, default activation is ReLU
#     """

#     def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
#                  dilation=1, groups=1, bias=False, activation=True) -> object:
#         super(BN_Conv2d, self).__init__()
#         layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                             padding=padding, dilation=dilation, groups=groups, bias=bias),
#                   nn.BatchNorm2d(out_channels)]
#         if activation:
#             layers.append(nn.ReLU(inplace=True))
#         self.seq = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.seq(x)
        
# class BasicBlock(nn.Module):
#     """
#     basic building block for ResNet-18, ResNet-34
#     """
#     message = "basic"

#     def __init__(self, in_channels, out_channels, strides, is_se=False):
#         super(BasicBlock, self).__init__()
#         self.is_se = is_se
#         self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
#         self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
#         if self.is_se:
#             self.se = SE(out_channels, 16)

#         # fit input with residual output
#         self.short_cut = nn.Sequential()
#         if strides is not 1:
#             self.short_cut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.is_se:
#             coefficient = self.se(out)
#             out = out * coefficient
#         out = out + self.short_cut(x)
#         return F.relu(out)


# class ResNetClassifier(nn.Module):
#     """
#     building ResNet_34
#     """

#     def __init__(self, block: object, groups: object, num_classes=1000) -> object:
#         super(ResNetClassifier, self).__init__()
#         self.channels = 180  # out channels from the first convolutional layer
#         self.block = block

#         self.conv1 = nn.Conv2d(257, 180, 7, stride=4, padding=3, bias=False)
#         self.bn = nn.BatchNorm2d(self.channels)
#         self.pool1 = nn.MaxPool2d(3, 2, 1)
#         self.conv2_x = self._make_conv_x(channels=160, blocks=groups[0], strides=2, index=2)
#         self.conv3_x = self._make_conv_x(channels=120, blocks=groups[1], strides=2, index=3)
#         # self.conv4_x = self._make_conv_x(channels=96, blocks=groups[2], strides=2, index=4)
#         # self.conv5_x = self._make_conv_x(channels=720, blocks=groups[3], strides=2, index=5)
#         self.pool2 = nn.AvgPool2d(7)
#         patches = 120 if self.block.message == "basic" else 120 * 4
        
#         self.fc = nn.Linear(patches, num_classes) 

#     def _make_conv_x(self, channels, blocks, strides, index):
#         """
#         making convolutional group
#         :param channels: output channels of the conv-group
#         :param blocks: number of blocks in the conv-group
#         :param strides: strides
#         :return: conv-group
#         """
#         list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
#         conv_x = nn.Sequential()
#         for i in range(len(list_strides)):
#             layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
#             conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
#             self.channels = channels if self.block.message == "basic" else channels * 4
#         return conv_x

#     def forward(self, x):
#         out = self.conv1(x)
#         out = F.relu(self.bn(out))
#         out = self.pool1(out)
#         out = self.conv2_x(out)
#         out = self.conv3_x(out)
#         # out = self.conv4_x(out)
#         # out = self.conv5_x(out)
#         out = self.pool2(out)
#         out = out.view(out.size(0), -1)
#         out = F.softmax(self.fc(out))
#         return out
        
# # -------------------------------------------------------- 3D 卷积分割 ↓ -------------------------------------------------------
# class TimeToChannel3DConvNet(nn.Module):
#     def __init__(self, in_channels=3, depth=10, height=256, width=256):
#         super(TimeToChannel3DConvNet, self).__init__()
        
#         # 3D 卷积层
#         self.conv1 = nn.Conv3d(
#             in_channels=in_channels, 
#             out_channels=16, 
#             kernel_size=(3, 3, 3), 
#             stride=1, 
#             padding=1
#         )
#         self.relu1 = nn.ReLU()
        
#         self.conv2 = nn.Conv3d(
#             in_channels=16, 
#             out_channels=depth,  # 输出通道数设为时间步数
#             kernel_size=(3, 3, 3), 
#             stride=1, 
#             padding=1
#         )
#         self.relu2 = nn.ReLU()

#     def forward(self, x):
#         # x: [batch, channels, depth, height, width]
#         x = self.relu1(self.conv1(x))  # 卷积1
#         x = self.relu2(self.conv2(x))  # 卷积2
        
#         # 调整维度，将 depth 作为输出通道数
#         x = x.permute(0, 2, 1, 3, 4)  # [batch, depth, channels, height, width] -> [batch, depth, channels, height, width]
        
#         return x

# class MemSAM(nn.Module):
#     mask_threshold: float = 0.0
#     image_format: str = "RGB"

#     def __init__(
#         self,
#         image_encoder: ImageEncoderViT,
#         prompt_encoder: PromptEncoder,
#         mask_decoder: MaskDecoder,
#         memory: Mem,
#         pixel_mean: List[float] = [123.675, 116.28, 103.53],
#         pixel_std: List[float] = [58.395, 57.12, 57.375],
#         num_classes: int = 2, # 二分类
#     ) -> None:
#         """
#         SAM predicts object masks from an image and input prompts.

#         Arguments:
#           image_encoder (ImageEncoderViT): The backbone used to encode the
#             image into image embeddings that allow for efficient mask prediction.
#           prompt_encoder (PromptEncoder): Encodes various types of input prompts.
#           mask_decoder (MaskDecoder): Predicts masks from the image embeddings
#             and encoded prompts.
#           pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
#           pixel_std (list(float)): Std values for normalizing pixels in the input image.
#         """
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.prompt_encoder = prompt_encoder
#         self.mask_decoder = mask_decoder
#         self.memory = memory
#         if self.memory is not None:
#             self.memory = memory
#             self.memory.key_encoder = image_encoder
#         ### 1行 --------------------------------------------- 3D 卷积分割 ---------------------------------------------------
#         self.td = TimeToChannel3DConvNet(in_channels=3, depth=10)
#         self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

#         # ----------------------------------------------  添加 ResNet 分类分支  ----------------------------------------------
#         self.classifier = ResNetClassifier(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)
#         self.agent = AgentAttention(120, 8*8)

#         for param in self.prompt_encoder.parameters():
#             param.requires_grad = False
#         for param in self.mask_decoder.parameters():
#             param.requires_grad = False
#         # for param in self.image_encoder.parameters():
#         #   param.requires_grad = False
#         for n, value in self.image_encoder.named_parameters():
#             if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
#                 value.requires_grad = False
#         pass
        
#     @property
#     def device(self) -> Any:
#         return self.pixel_mean.device

#     @torch.no_grad()
#     def forward_sam(
#         self,
#         batched_input: List[Dict[str, Any]],
#         multimask_output: bool,
#     ) -> List[Dict[str, torch.Tensor]]:
#         """
#         Predicts masks end-to-end from provided images and prompts.
#         If prompts are not known in advance, using SamPredictor is
#         recommended over calling the model directly.

#         Arguments:
#           batched_input (list(dict)): A list over input images, each a
#             dictionary with the following keys. A prompt key can be
#             excluded if it is not present.
#               'image': The image as a torch tensor in 3xHxW format,
#                 already transformed for input to the model.
#               'original_size': (tuple(int, int)) The original size of
#                 the image before transformation, as (H, W).
#               'point_coords': (torch.Tensor) Batched point prompts for
#                 this image, with shape BxNx2. Already transformed to the
#                 input frame of the model.
#               'point_labels': (torch.Tensor) Batched labels for point prompts,
#                 with shape BxN.
#               'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
#                 Already transformed to the input frame of the model.
#               'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
#                 in the form Bx1xHxW.
#           multimask_output (bool): Whether the model should predict multiple
#             disambiguating masks, or return a single mask.

#         Returns:
#           (list(dict)): A list over input images, where each element is
#             as dictionary with the following keys.
#               'masks': (torch.Tensor) Batched binary mask predictions,
#                 with shape BxCxHxW, where B is the number of input prompts,
#                 C is determined by multimask_output, and (H, W) is the
#                 original size of the image.
#               'iou_predictions': (torch.Tensor) The model's predictions
#                 of mask quality, in shape BxC.
#               'low_res_logits': (torch.Tensor) Low resolution logits with
#                 shape BxCxHxW, where H=W=256. Can be passed as mask input
#                 to subsequent iterations of prediction.
#         """
#         input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
#         image_embeddings = self.image_encoder(input_images)

#         outputs = []
#         for image_record, curr_embedding in zip(batched_input, image_embeddings):
#             if "point_coords" in image_record:
#                 points = (image_record["point_coords"], image_record["point_labels"])
#             else:
#                 points = None
#             sparse_embeddings, dense_embeddings = self.prompt_encoder(
#                 points=points,
#                 boxes=image_record.get("boxes", None),
#                 masks=image_record.get("mask_inputs", None),
#             )
#             low_res_masks, iou_predictions = self.mask_decoder(
#                 image_embeddings=curr_embedding.unsqueeze(0),
#                 image_pe=self.prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=multimask_output,
#             )
#             masks = self.postprocess_masks(
#                 low_res_masks,
#                 input_size=image_record["image"].shape[-2:],
#                 original_size=image_record["original_size"],
#             )
#             masks = masks > self.mask_threshold
#             outputs.append(
#                 {
#                     "masks": masks,
#                     "iou_predictions": iou_predictions,
#                     "low_res_logits": low_res_masks,
#                 }
#             )
#         return outputs

#     # 2024年12月6日修改前的代码
#     # def forward(
#     #     self,
#     #     imgs: torch.Tensor, # [b,t,c,h,w]
#     #     pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
#     #     bbox: torch.Tensor=None, # b 4
#     # ) -> torch.Tensor:
#     #     if self.memory is not None:
#     #         #### 4行 -------------------------------------------- CNN -------------------------------------------------------
#     #         input_tensor = imgs.permute(0, 2, 1, 3, 4)
#     #         cnn = self.td(input_tensor)
#     #         output_tensor_reverted = cnn.permute(0, 2, 1, 3, 4)
#     #         print('----------------',(self._forward_with_memory(imgs,pt,bbox)+output_tensor_reverted).shape)
#     #         return self._forward_with_memory(imgs,pt,bbox)
#     #     else:
#     #         return self._forward_without_memory(imgs,pt,bbox)

#     def forward(
#     self,
#     imgs: torch.Tensor,  # [b, t, c, h, w]
#     pt: Tuple[torch.Tensor, torch.Tensor],  # ([b, n, 2], [b, n])
#     bbox: torch.Tensor = None,  # [b, 4]
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if self.memory is not None:
#         #### 使用memory模块的前向传播
#             input_tensor = imgs.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]
#             cnn = self.td(input_tensor)  # [b, t, channels, h, w]
#             output_tensor_reverted = cnn.permute(0, 2, 1, 3, 4)  # [b, channels, t, h, w]
#             mask_pred = self._forward_with_memory(imgs, pt, bbox)
#         else:
#             mask_pred = self._forward_without_memory(imgs, pt, bbox)
    
#         # 提取用于分类的特征
#         b, t, _, h, w = mask_pred.shape
#         # 将mask_pred reshape为 [b*t, 1, h, w]
#         mask_pred_reshaped = mask_pred.view(b * t, 1, h, w)
#         # 提取图像特征
#         imgs_reshaped = rearrange(imgs, "b t c h w -> (b t) c h w")
#         image_features = self.image_encoder(imgs_reshaped)  # [b*t, C, H', W']
#         # 确保image_features的空间尺寸与mask_pred一致
#         image_features = F.interpolate(image_features, size=(h, w), mode="bilinear", align_corners=False)
#         # 拼接mask_pred和image_features
#         combined_features = torch.cat([image_features, mask_pred_reshaped], dim=1)  # [b*t, C+1, h, w]
#         # 将拼接后的特征传递给分类网络

#         class_logits = self.classifier(combined_features)  # [b*t, num_classes]
#         # class_logits = rearrange(class_logits, 'b c h w -> b (h w) c')
#         # class_logits = self.agent(class_logits, 8, 8)

#         # print("class_logits:{}".format(np.shape(class_logits)))
#         class_logits = class_logits.view(b, t, -1)  # [b, t, num_classes]
#         # class_logits = 1
#         return mask_pred, class_logits


#     def _forward_without_memory(
#         self,
#         imgs: torch.Tensor, # [b,t,c,h,w]
#         pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
#         bbox: torch.Tensor=None, # b 4
#     ) -> torch.Tensor:
#         b, t, c, h, w = imgs.shape  # b t c h w
#         imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
#         imgs= self.image_encoder(imgs)
#         imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=b)
#         frames_pred = []
#         for ti in range(0, t):
#             frame = imgs[:,ti,:,:,:]
#             se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
#                         points=(pt[0][:,0],pt[1][:1]),
#                         boxes=None,
#                         masks=None,
#                     )
#             mask, _ = self.mask_decoder( # low_res_mask b 1 128 128
#                         image_embeddings=frame,
#                         image_pe=self.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=se,
#                         dense_prompt_embeddings=de,
#                         multimask_output=False,
#                     ) # b c h w
#             mask = F.interpolate(mask, (h,w), mode="bilinear", align_corners=False) #b 1 256 256
#             frames_pred.append(mask)
#         pred = torch.stack(frames_pred, dim=1) # b t c h w

#         return pred

#     def _forward_with_memory(
#         self,
#         imgs: torch.Tensor, # [b,t,c,h,w]
#         pt: Tuple[torch.Tensor, torch.Tensor],  # ([b n 1 2], [b n])
#         bbox: torch.Tensor=None, # b 4
#     ) -> torch.Tensor:
#         b, t, c, h, w = imgs.shape  # b t c h w
#         # encode imgs to imgs embedding
#         key, shrinkage, selection, imge = self.memory('encode_key', imgs)
#         # init memory
#         hidden = torch.zeros((b, 1, self.memory.hidden_dim, *key.shape[-2:])).to(imge.device)
#         frames_pred = []
#         # first frame
#         if pt is not None:
#             se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
#                         points=(pt[0][:,0],pt[1][:1]),
#                         boxes=None,
#                         masks=None,
#                     )
#         else:
#             se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
#                         points=None,
#                         boxes=None,
#                         masks=None,
#                     )
#         mask, _ = self.mask_decoder(
#                     image_embeddings=imge[:,0],
#                     image_pe=self.prompt_encoder.get_dense_pe(),
#                     sparse_prompt_embeddings=se,
#                     dense_prompt_embeddings=de,
#                     multimask_output=False,
#                 ) # b c h w
#         mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
#         # frames_pred.append(mask)
#         values_0, hidden = self.memory('encode_value', imgs[:,0], imge[:,0], hidden, mask)
#         values = values_0[:,:,:,:0]

#         # process frames
#         for ti in range(0, t):
#             if ti == 0 :
#                 ref_keys = key[:,:,[0]] 
#                 ref_shrinkage = shrinkage[:,:,[0]]
#                 ref_values = values_0 
#             else:
#                 ref_keys = key[:,:,:ti]
#                 ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
#                 ref_values = values

#             # get single frame
#             frame = imge[:,ti]
#             # read memory
#             memory_readout = self.memory(
#                 'read_memory',
#                 key[:, :, ti],
#                 selection[:, :, ti] if selection is not None else None,
#                 ref_keys, ref_shrinkage, ref_values)
#             # generate memory embedding
#             hidden, me = self.memory('decode', frame, hidden, memory_readout)
#             # # featmap
#             # from mmengine.visualization import Visualizer
#             # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
#             #                         save_dir='temp_dir')
#             # drawn_img = visualizer.draw_featmap(featmap=me[0,0]*-1,
#             #                         overlaid_image=imgs[0,ti].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8),
#             #                         channel_reduction='squeeze_mean',
#             #                         alpha=0.3)
#             # if self.memory.reinforce :
#             #     visualizer.add_image(f'featmap_reinforce', drawn_img, step=ti)
#             # else:
#             #     visualizer.add_image(f'featmap_noreinforce', drawn_img, step=ti)

#             mask, _ = self.mask_decoder( 
#                         image_embeddings=frame,
#                         image_pe=self.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=None,
#                         dense_prompt_embeddings=me[:,0], # remove object num dim
#                         multimask_output=False,
#                     ) # b c h w
#             mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
#             frames_pred.append(mask)

#             # last frame no encode
#             if ti < t-1:
#                 # update memory
#                 is_deep_update = np.random.rand() < 0.2
#                 # v16, hidden = self.memory('encode_value', imgs[:,ti], me[:,0], hidden, mask, is_deep_update=is_deep_update)
#                 v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
#                 values = torch.cat([values, v16], 3)

#         pred = torch.stack(frames_pred, dim=1) # b t c h w

#         return pred.sigmoid()
#         # 1129本身是
#         # return pred

#     def postprocess_masks(
#         self,
#         masks: torch.Tensor,
#         input_size: Tuple[int, ...],
#         original_size: Tuple[int, ...],
#     ) -> torch.Tensor:
#         """
#         Remove padding and upscale masks to the original image size.

#         Arguments:
#           masks (torch.Tensor): Batched masks from the mask_decoder,
#             in BxCxHxW format.
#           input_size (tuple(int, int)): The size of the image input to the
#             model, in (H, W) format. Used to remove padding.
#           original_size (tuple(int, int)): The original size of the image
#             before resizing for input to the model, in (H, W) format.

#         Returns:
#           (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
#             is given by original_size.
#         """
#         masks = F.interpolate(
#             masks,
#             (self.image_encoder.img_size, self.image_encoder.img_size),
#             mode="bilinear",
#             align_corners=False,
#         )
#         masks = masks[..., : input_size[0], : input_size[1]]
#         masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
#         return masks

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         # Normalize colors
#         x = (x - self.pixel_mean) / self.pixel_std

#         # Pad
#         h, w = x.shape[-2:]
#         padh = self.image_encoder.img_size - h
#         padw = self.image_encoder.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x



# class AgentAttention(nn.Module):
#     def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#                  proj_drop=0., sr_ratio=1, agent_num=49, **kwargs):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_patches = num_patches
#         window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, 2)  # 默认dim
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)

#         self.agent_num = agent_num
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, H, W):
#         b, n, c = x.shape
#         num_heads = self.num_heads
#         head_dim = c // num_heads
#         q = self.q(x)

#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
#             x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
#         else:
#             kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
#         k, v = kv[0], kv[1]

#         agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
#         q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

#         kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v

#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v

#         x = x.transpose(1, 2).reshape(b, n, c)
#         v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
#         if self.sr_ratio > 1:
#             v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

#         x = x.view(x.size(0), -1)
#         x = self.proj(x)
#         # x = self.proj_drop(x)
#         x = F.softmax(x)

#         return x




# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .mem import Mem
from einops import rearrange
import torchvision.models as models
from timm.models.layers import trunc_normal_

# ------------------------------------------------------------- RESNET分类--------------------------------------------
# class ClassificationHead(nn.Module):
#     def __init__(self, in_features, num_classes=2):
#         super(ClassificationHead, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = torch.flatten(x, 1)  # [batch, features]
#         x = self.fc(x)
#         return x


# class ResNetClassifier(nn.Module):
#     def __init__(self, in_channels: int, num_classes: int):
#         super(ResNetClassifier, self).__init__()
#         self.classifier = models.resnet18(pretrained=True)

#     def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
#         out = self.classifier(combined_features)
#         print('resnet:{}'.format(np.shape(out)))
#         return out

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = "basic"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)


class ResNetClassifier(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes=1000) -> object:
        super(ResNetClassifier, self).__init__()
        self.channels = 180  # out channels from the first convolutional layer
        self.block = block

        self.conv1 = nn.Conv2d(257, 180, 7, stride=4, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=160, blocks=groups[0], strides=2, index=2)
        self.conv3_x = self._make_conv_x(channels=120, blocks=groups[1], strides=2, index=3)
        # self.conv4_x = self._make_conv_x(channels=96, blocks=groups[2], strides=2, index=4)
        # self.conv5_x = self._make_conv_x(channels=720, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 120 if self.block.message == "basic" else 120 * 4

        self.fc = nn.Linear(patches, num_classes)

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)

        out = F.relu(self.bn(out))
        out = self.pool1(out)

        out = self.conv2_x(out)
        out = self.conv3_x(out)

        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out

# -------------------------------------------------------- 3D 卷积分割 ↓ -------------------------------------------------------
class TimeToChannel3DConvNet(nn.Module):
    def __init__(self, in_channels=3, depth=10, height=256, width=256):
        super(TimeToChannel3DConvNet, self).__init__()

        # 3D 卷积层
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=depth,  # 输出通道数设为时间步数
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        x = self.relu1(self.conv1(x))  # 卷积1
        x = self.relu2(self.conv2(x))  # 卷积2

        # 调整维度，将 depth 作为输出通道数
        x = x.permute(0, 2, 1, 3, 4)  # [batch, depth, channels, height, width] -> [batch, depth, channels, height, width]

        return x

class MemSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory: Mem,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        num_classes: int = 2, # 二分类
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory = memory
        if self.memory is not None:
            self.memory = memory
            self.memory.key_encoder = image_encoder
        ### 1行 --------------------------------------------- 3D 卷积分割 ---------------------------------------------------
        self.td = TimeToChannel3DConvNet(in_channels=3, depth=10)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # ----------------------------------------------  添加 ResNet 分类分支  ----------------------------------------------
        self.classifier = ResNetClassifier(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)
        self.agent = AgentAttention(120, 8*8)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
            if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
                value.requires_grad = False
        pass

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(self, batched_input: List[Dict[str, Any]], multimask_output: bool, ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    # 2024年12月6日修改前的代码
    # def forward(
    #     self,
    #     imgs: torch.Tensor, # [b,t,c,h,w]
    #     pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
    #     bbox: torch.Tensor=None, # b 4
    # ) -> torch.Tensor:
    #     if self.memory is not None:
    #         #### 4行 -------------------------------------------- CNN -------------------------------------------------------
    #         input_tensor = imgs.permute(0, 2, 1, 3, 4)
    #         cnn = self.td(input_tensor)
    #         output_tensor_reverted = cnn.permute(0, 2, 1, 3, 4)
    #         print('----------------',(self._forward_with_memory(imgs,pt,bbox)+output_tensor_reverted).shape)
    #         return self._forward_with_memory(imgs,pt,bbox)
    #     else:
    #         return self._forward_without_memory(imgs,pt,bbox)

    def forward(
    self,
    imgs: torch.Tensor,  # [b, t, c, h, w]
    pt: Tuple[torch.Tensor, torch.Tensor],  # ([b, n, 2], [b, n])
    bbox: torch.Tensor = None,  # [b, 4]
) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory is not None:
        #### 使用memory模块的前向传播
            input_tensor = imgs.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]
            cnn = self.td(input_tensor)  # [b, t, channels, h, w]
            output_tensor_reverted = cnn.permute(0, 2, 1, 3, 4)  # [b, channels, t, h, w]
            mask_pred = self._forward_with_memory(imgs, pt, bbox)
        else:
            mask_pred = self._forward_without_memory(imgs, pt, bbox)

        input_tensor = imgs.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]
        cnn = self.td(input_tensor).squeeze(0)  # [b, t, channels, h, w]

        # 提取用于分类的特征
        b, t, _, h, w = mask_pred.shape
        # 将mask_pred reshape为 [b*t, 1, h, w]
        mask_pred_reshaped = mask_pred.view(b * t, 1, h, w)
        # 提取图像特征
        imgs_reshaped = rearrange(imgs, "b t c h w -> (b t) c h w")
        image_features = self.image_encoder(imgs_reshaped)  # [b*t, C, H', W']
        # 确保image_features的空间尺寸与mask_pred一致
        image_features = F.interpolate(image_features, size=(h, w), mode="bilinear", align_corners=False)
        # 拼接mask_pred和image_features
        combined_features = torch.cat([image_features, mask_pred_reshaped], dim=1)  # [b*t, C+1, h, w]
        # 将拼接后的特征传递给分类网络
        class_logits = self.classifier(combined_features)  # [b*t, num_classes]
        # class_logits = rearrange(class_logits, 'b c h w -> b (h w) c')
        # class_logits = self.agent(class_logits, 8, 8)

        # print("class_logits:{}".format(np.shape(class_logits)))
        class_logits = class_logits.view(b, t, -1)  # [b, t, num_classes]
        # class_logits = 1
        return mask_pred, class_logits


    def _forward_without_memory(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        b, t, c, h, w = imgs.shape  # b t c h w
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
        imgs= self.image_encoder(imgs)
        imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=b)
        frames_pred = []
        for ti in range(0, t):
            frame = imgs[:,ti,:,:,:]
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=(pt[0][:,0],pt[1][:1]),
                        boxes=None,
                        masks=None,
                    )
            mask, _ = self.mask_decoder( # low_res_mask b 1 128 128
                        image_embeddings=frame,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    ) # b c h w
            mask = F.interpolate(mask, (h,w), mode="bilinear", align_corners=False) #b 1 256 256
            frames_pred.append(mask)
        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred

    def _forward_with_memory(self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # ([b n 1 2], [b n])
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        b, t, c, h, w = imgs.shape  # b t c h w
        # encode imgs to imgs embedding
        key, shrinkage, selection, imge = self.memory('encode_key', imgs)
        # init memory
        hidden = torch.zeros((b, 1, self.memory.hidden_dim, *key.shape[-2:])).to(imge.device)
        frames_pred = []
        # first frame
        if pt is not None:
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=(pt[0][:,0],pt[1][:1]),
                        boxes=None,
                        masks=None,
                    )
        else:
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=None,
                        boxes=None,
                        masks=None,
                    )
        mask, _ = self.mask_decoder(
                    image_embeddings=imge[:,0],
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                ) # b c h w
        mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
        # frames_pred.append(mask)
        values_0, hidden = self.memory('encode_value', imgs[:,0], imge[:,0], hidden, mask)
        values = values_0[:,:,:,:0]

        # process frames
        for ti in range(0, t):
            if ti == 0 :
                ref_keys = key[:,:,[0]]
                ref_shrinkage = shrinkage[:,:,[0]]
                ref_values = values_0
            else:
                ref_keys = key[:,:,:ti]
                ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                ref_values = values

            # get single frame
            frame = imge[:,ti]
            # read memory
            memory_readout = self.memory(
                'read_memory',
                key[:, :, ti],
                selection[:, :, ti] if selection is not None else None,
                ref_keys, ref_shrinkage, ref_values)
            # generate memory embedding
            hidden, me = self.memory('decode', frame, hidden, memory_readout)
            # # featmap
            # from mmengine.visualization import Visualizer
            # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
            #                         save_dir='temp_dir')
            # drawn_img = visualizer.draw_featmap(featmap=me[0,0]*-1,
            #                         overlaid_image=imgs[0,ti].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8),
            #                         channel_reduction='squeeze_mean',
            #                         alpha=0.3)
            # if self.memory.reinforce :
            #     visualizer.add_image(f'featmap_reinforce', drawn_img, step=ti)
            # else:
            #     visualizer.add_image(f'featmap_noreinforce', drawn_img, step=ti)

            mask, _ = self.mask_decoder(
                        image_embeddings=frame,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=None,
                        dense_prompt_embeddings=me[:,0], # remove object num dim
                        multimask_output=False,
                    ) # b c h w
            mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
            frames_pred.append(mask)

            # last frame no encode
            if ti < t-1:
                # update memory
                is_deep_update = np.random.rand() < 0.2
                # v16, hidden = self.memory('encode_value', imgs[:,ti], me[:,0], hidden, mask, is_deep_update=is_deep_update)
                v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
                values = torch.cat([values, v16], 3)

        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred.sigmoid()
        # 1129本身是
        # return pred

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(120, 2)  # 默认dim
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)
        self.pool2 = nn.AvgPool2d(7)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        v = self.dwc(v)

        x = x.permute(0, 2, 1).reshape(b, c, H, W)
        x = self.pool2(x)
        v = self.pool2(v)

        # print("x shape is:", x.shape)
        # print("v shape is:", v.shape)

        x = x + v#.permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        # x = self.proj_drop(x)
        x = F.softmax(x)

        return x