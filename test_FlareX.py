import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
import math
import torchvision.transforms as transforms
from tqdm import tqdm

from model.RaLiFormer import RaLiFormer
from dataset_all import TestData


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
bz = 1
model_root = "./experiments/model_RaLiFormer/ckpt/model_best.pth"
input_root = "/mnt/zxy/the_next_work/FlareX/dataset/test_dataset"
# Set to a ground-truth folder when metrics are available.
gt_root = None

save_root = "./result/real_FlareX"
save_path_img = os.path.join(save_root, "images")
# save_path_flare = os.path.join(save_root, "flares")

if not os.path.isdir(save_path_img):
    os.makedirs(save_path_img)
# if not os.path.isdir(save_path_flare):
#     os.makedirs(save_path_flare)


def save_tensor_img(tensor_img, save_dir, name):
    # tensor_img: [C, H, W] -> numpy
    temp = tensor_img.squeeze().cpu().detach().numpy()
    if len(temp.shape) == 3:
        temp = np.transpose(temp, (1, 2, 0))
    elif len(temp.shape) == 2:
        pass

    temp = np.clip(temp, 0, 1)
    temp = (temp * 255).astype(np.uint8)
    temp = Image.fromarray(temp)
    temp.save(os.path.join(save_dir, name))


print("Loading Model...")
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz, shuffle=False)

resize = transforms.Resize((512, 512))
model = RaLiFormer().cuda()
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_dict"])
epoch = checkpoint["epoch"]
print("Model Loaded! Start Inference...")

psnr_list, ssim_list = [], []

for data_idx, data_ in tqdm(enumerate(data_load), total=len(data_load)):
    data_input_ori = data_.cuda()
    # Keep the original size for restoration after 512px inference.
    b, c, h_ori, w_ori = data_input_ori.shape
    resize2org = transforms.Resize((h_ori, w_ori))
    data_input = resize(data_input_ori)
    with torch.no_grad():
        result, _ = model(data_input)

        result = data_input_ori - resize2org(data_input - result)
        name = os.path.basename(Mydata_.A_paths[data_idx])

        save_tensor_img(result[0, :], save_path_img, name)

        if gt_root is not None:
            gt_path = os.path.join(gt_root, name)
            if os.path.exists(gt_path):
                img_gt = np.array(Image.open(gt_path).convert("RGB"))

print("Finished!")
