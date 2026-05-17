import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP

from model.RaLiFormer import RaLiFormer
from dataset_all import TestData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bz = 1
# model_root = 'pretrained/model.pth'
model_root = "./experiments/model_RaLiFormer/ckpt/model_best.pth"
input_root = "data/test/real/"

save_root = "result/real"
save_path_img = os.path.join(save_root, "images")
save_path_flare = os.path.join(save_root, "flares")

if not os.path.isdir(save_path_img):
    os.makedirs(save_path_img)
if not os.path.isdir(save_path_flare):
    os.makedirs(save_path_flare)

checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz)

model = RaLiFormer().cuda()
# model = nn.DataParallel(model, device_ids=[0, 1])
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_dict"])
epoch = checkpoint["epoch"]
model.eval()
print("START!")


def save_tensor_img(tensor_img, save_dir, name):
    # tensor_img: [C, H, W]
    temp = np.transpose(tensor_img.cpu().detach().numpy(), (1, 2, 0))
    temp[temp > 1] = 1
    temp[temp < 0] = 0
    temp = (temp * 255).astype(np.uint8)

    # Handle single-channel tensors as grayscale images.
    if temp.shape[2] == 1:
        temp = temp[:, :, 0]

    temp = Image.fromarray(temp)
    temp.save(os.path.join(save_dir, name))


if 1:
    print("Load model successfully!")
    for data_idx, data_ in enumerate(data_load):
        data_input = data_
        data_input = data_input.cuda()

        with torch.no_grad():
            result, flare_predict = model(data_input)

            name = os.path.basename(Mydata_.A_paths[data_idx])
            print(f"Processing: {name}")

            save_tensor_img(result[0, :], save_path_img, name)
            save_tensor_img(flare_predict[0, :], save_path_flare, name)

print("finished!")
