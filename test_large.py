import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageChops
import torchvision.transforms as transforms

from model.RaLiFormer import RaLiFormer
from dataset_all import TestData


class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

    def process_image(self, image):
        to_tensor = transforms.ToTensor()
        original_image = image

        resized_image = self.resize_image(original_image, 512)
        resized_width, resized_height = resized_image.size

        segments = []
        overlaps = []

        if resized_width > 512:
            for end_x in range(512, resized_width + 256, 256):
                end_x = min(end_x, resized_width)
                overlaps.append(end_x)
                cropped_image = resized_image.crop((end_x - 512, 0, end_x, 512))
                processed_segment = self.model(
                    to_tensor(cropped_image).unsqueeze(0).to(self.device)
                ).squeeze(0)
                segments.append(processed_segment)
        else:
            for end_y in range(512, resized_height + 256, 256):
                end_y = min(end_y, resized_height)
                overlaps.append(end_y)
                cropped_image = resized_image.crop((0, end_y - 512, 512, end_y))
                processed_segment = self.model(
                    to_tensor(cropped_image).unsqueeze(0).to(self.device)
                ).squeeze(0)
                segments.append(processed_segment)

        overlaps = [0] + [prev - cur + 512 for prev, cur in zip(overlaps[:-1], overlaps[1:])]

        # Use the model output channels instead of assuming a fixed count.
        img_c = segments[0].shape[0]

        for i in range(1, len(segments)):
            overlap = overlaps[i]
            alpha = torch.linspace(0, 1, steps=overlap).to(self.device)

            if resized_width > 512:
                alpha = alpha.view(1, -1, 1).expand(512, -1, img_c).permute(2, 0, 1)
                segments[i][:, :, :overlap] = (
                    alpha * segments[i][:, :, :overlap]
                    + (1 - alpha) * segments[i - 1][:, :, -overlap:]
                )
            else:
                alpha = alpha.view(-1, 1, 1).expand(-1, 512, img_c).permute(2, 0, 1)
                segments[i][:, :overlap, :] = (
                    alpha * segments[i][:, :overlap, :]
                    + (1 - alpha) * segments[i - 1][:, -overlap:, :]
                )

        if resized_width > 512:
            blended = [
                segment[:, :, :-overlap] for segment, overlap in zip(segments[:-1], overlaps[1:])
            ] + [segments[-1]]
            merged_image = torch.cat(blended, dim=2)
        else:
            blended = [
                segment[:, :-overlap, :] for segment, overlap in zip(segments[:-1], overlaps[1:])
            ] + [segments[-1]]
            merged_image = torch.cat(blended, dim=1)

        return merged_image


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        res, _ = self.model(x)
        return res


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_root = "./experiments/model_RaLiFormer/ckpt/model_best.pth"
input_root = "/mnt/zxy/Flare/FlareReal600/data/"
save_root = "result/real_large"

save_path_img = os.path.join(save_root, "images")
if not os.path.isdir(save_path_img):
    os.makedirs(save_path_img)

print("Loading Model...")
checkpoint = torch.load(model_root)
model = RaLiFormer().cuda()
model.load_state_dict(checkpoint["state_dict"])
model.eval()

wrapper_model = ModelWrapper(model)
processor = ImageProcessor(wrapper_model)

Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=1)

print("START!")

if 1:
    print("Load model successfully!")
    for data_idx, _ in enumerate(data_load):
        img_path = Mydata_.A_paths[data_idx]
        name = os.path.basename(img_path)
        print(f"Processing: {name}")

        merge_img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            output_tensor_lowres = processor.process_image(merge_img)

            deflare_img_np = output_tensor_lowres.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            deflare_img_pil = Image.fromarray((deflare_img_np * 255).astype(np.uint8))

            # Restore high-resolution details with the low-resolution flare residual.
            merge_img_resized = merge_img.resize(deflare_img_pil.size)
            flare_img_lowres = ImageChops.difference(merge_img_resized, deflare_img_pil)
            flare_img_highres = flare_img_lowres.resize(merge_img.size, resample=Image.BICUBIC)
            deflare_img_highres = ImageChops.difference(merge_img, flare_img_highres)

            deflare_img_highres.save(os.path.join(save_path_img, name))

print("finished!")
