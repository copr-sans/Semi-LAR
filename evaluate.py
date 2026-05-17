import argparse
import warnings
from glob import glob
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage import io
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor
from tqdm import tqdm

warnings.filterwarnings("ignore")


def read_rgb(path):
    image = io.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0).cuda()
    img2_tensor = to_tensor(img2).unsqueeze(0).cuda()

    with torch.no_grad():
        output_lpips = loss_fn_alex(img1_tensor, img2_tensor)

    return output_lpips.cpu().numpy()[0, 0, 0, 0]


def extract_mask(img_seg):
    """Return glare, streak, and global masks as 3-channel float masks."""
    height, width = img_seg.shape[:2]
    normalizer = float(height * width)

    streak_mask = (img_seg[:, :, 0] - img_seg[:, :, 1]) / 255
    glare_mask = img_seg[:, :, 1] / 255
    global_mask = (255 - img_seg[:, :, 2]) / 255

    return {
        "glare": [
            np.sum(glare_mask) / normalizer,
            np.expand_dims(glare_mask, 2).repeat(3, axis=2),
        ],
        "streak": [
            np.sum(streak_mask) / normalizer,
            np.expand_dims(streak_mask, 2).repeat(3, axis=2),
        ],
        "global": [
            np.sum(global_mask) / normalizer,
            np.expand_dims(global_mask, 2).repeat(3, axis=2),
        ],
    }


def compare_masked_scores(gt_img, pred_img, mask_img):
    """Return mask-aware PSNR values for glare, streak, and global regions."""
    metric_dict = {}

    for mask_type in ["glare", "streak", "global"]:
        mask_area, img_mask = extract_mask(mask_img)[mask_type]
        if mask_area <= 0:
            continue

        gt_masked = gt_img * img_mask
        pred_masked = pred_img * img_mask
        masked_mse = mean_squared_error(gt_masked, pred_masked) / (255 * 255 * mask_area)

        metric_dict[mask_type] = 10 * np.log10(1.0 / masked_mse)

    return metric_dict


def top_k_lowest(image_metrics, metric_name, k):
    return sorted(image_metrics, key=lambda item: item[metric_name])[:k]


def top_k_highest(image_metrics, metric_name, k):
    return sorted(image_metrics, key=lambda item: item[metric_name], reverse=True)[:k]


def write_per_image_results(output_path, image_metrics):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Image\tPSNR\tSSIM\tLPIPS\n")
        for metrics in image_metrics:
            file.write(
                f"{metrics['image']}\t"
                f"{metrics['psnr']:.10f}\t"
                f"{metrics['ssim']:.10f}\t"
                f"{metrics['lpips']:.10f}\n"
            )


def write_worst_case_results(output_path, image_metrics, top_k):
    worst_psnr = top_k_lowest(image_metrics, "psnr", top_k)
    worst_ssim = top_k_lowest(image_metrics, "ssim", top_k)
    worst_lpips = top_k_highest(image_metrics, "lpips", top_k)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Worst {top_k} images based on PSNR:\n")
        for image in worst_psnr:
            file.write(f"{image['image']}\t{image['psnr']:.10f}\n")

        file.write(f"\nWorst {top_k} images based on SSIM:\n")
        for image in worst_ssim:
            file.write(f"{image['image']}\t{image['ssim']:.10f}\n")

        file.write(f"\nWorst {top_k} images based on LPIPS:\n")
        for image in worst_lpips:
            file.write(f"{image['image']}\t{image['lpips']:.10f}\n")


def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net="alex").cuda()

    gt_list = sorted(glob(str(Path(args.gt) / "*")))
    pred_list = sorted(glob(str(Path(args.input) / "*")))
    mask_list = sorted(glob(str(Path(args.mask) / "*"))) if args.mask else None

    if len(gt_list) != len(pred_list):
        raise ValueError(
            f"GT and prediction counts do not match: {len(gt_list)} vs {len(pred_list)}"
        )
    if mask_list is not None and len(mask_list) != len(gt_list):
        raise ValueError(f"Mask count does not match image count: {len(mask_list)} vs {len(gt_list)}")

    total_images = len(gt_list)
    if total_images == 0:
        raise ValueError("No images found for evaluation.")

    ssim_total = 0.0
    psnr_total = 0.0
    lpips_total = 0.0
    image_metrics = []
    score_dict = {
        "glare": 0.0,
        "streak": 0.0,
        "global": 0.0,
        "glare_num": 0,
        "streak_num": 0,
        "global_num": 0,
    }

    for index in tqdm(range(total_images)):
        gt_img = read_rgb(gt_list[index])
        pred_img = read_rgb(pred_list[index])

        ssim_value = structural_similarity(gt_img, pred_img, channel_axis=2, data_range=255)
        psnr_value = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
        lpips_value = compare_lpips(gt_img, pred_img, loss_fn_alex)

        image_metrics.append(
            {
                "image": Path(pred_list[index]).name,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
            }
        )

        ssim_total += ssim_value
        psnr_total += psnr_value
        lpips_total += lpips_value

        if mask_list is not None:
            mask_img = read_rgb(mask_list[index])
            metric_dict = compare_masked_scores(gt_img, pred_img, mask_img)
            for key, value in metric_dict.items():
                score_dict[key] += value
                score_dict[f"{key}_num"] += 1

    psnr_avg = psnr_total / total_images
    ssim_avg = ssim_total / total_images
    lpips_avg = lpips_total / total_images

    print(f"PSNR: {psnr_avg}, SSIM: {ssim_avg}, LPIPS: {lpips_avg}")

    if mask_list is not None:
        for key in ["glare", "streak", "global"]:
            if score_dict[f"{key}_num"] == 0:
                raise ValueError(f"No valid {key} masks were found.")
            score_dict[key] /= score_dict[f"{key}_num"]

        score_dict["score"] = (score_dict["glare"] + score_dict["global"] + score_dict["streak"]) / 3
        print(
            f"G-PSNR: {score_dict['glare']}, "
            f"S-PSNR: {score_dict['streak']}, "
            f"Score: {score_dict['score']}"
        )

    # write_per_image_results(args.per_image_output, image_metrics)
    # write_worst_case_results(args.worst_output, image_metrics, args.top_k)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./result/real_large/images")
    parser.add_argument("--gt", type=str, default="./data/test/real/gt")
    parser.add_argument("--mask", type=str, default="./data/test/real/mask")
    parser.add_argument("--per_image_output", type=str, default="metrics_per_image.txt")
    parser.add_argument("--worst_output", type=str, default="metrics_worst_cases.txt")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    calculate_metrics(parse_args())
