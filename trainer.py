import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import PIL.Image as Image
from adamp import AdamP
from torch.autograd import Variable
from torchvision.models import vgg16
import pyiqa

from loss.losses import CharbonnierLoss, MyLoss, PerpetualLoss
from utils import AverageMeter, FFTLoss, PatchInfoNCE, compute_psnr_ssim, initialize_weights, to_psnr


class Trainer:
    def __init__(
        self,
        model,
        tmodel,
        args,
        supervised_loader,
        unsupervised_loader,
        val_loader,
        iter_per_epoch,
        writer,
    ):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 5

        self.loss_str = MyLoss().cuda()
        self.consistency = 0.2
        # self.consistency_rampup = 100.0
        self.consistency_rampup = 20.0
        self.iqa_metric = pyiqa.create_metric("musiq", as_loss=False).cuda()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        # Auxiliary losses used by the supervised branch.
        self.loss_fft = FFTLoss().cuda()

        # Unsupervised branch.
        self.loss_unsup = CharbonnierLoss().cuda()
        self.vgg_feat = vgg_model.cuda()
        self.loss_cr = PatchInfoNCE(self.vgg_feat, num_patches=128, temperature=0.1).cuda()

        self.curiter = 0

        self.model.cuda()
        self.tmodel.cuda()

        self.device, _ = self._get_available_devices(self.args.gpus)
        self.model = self.model.to(self.device)
        self.tmodel = self.tmodel.to(self.device)

        self.base_lr = 1e-4
        self.warmup_epochs = 5
        self.optimizer_s = AdamP(
            self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
        )
        # self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[35, 45], gamma=0.5)
        self.lr_scheduler_s = lr_scheduler.CosineAnnealingLR(
            self.optimizer_s, T_max=self.epochs, eta_min=1e-5
        )

        # Track the best validation PSNR for checkpointing.
        self.best_psnr = 0.0

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.9990):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul, _ = self.tmodel(image)

        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    # Dependable Repository update strategy with physical bounding and validity gating.
    def update_dependable_repository(
        self, teacher_predict, student_predict, positive_list, p_name, score_r, raw_input
    ):
        N = teacher_predict.shape[0]

        # Bound the teacher prediction by the input image with a small tolerance.
        constrained_teacher = torch.min(teacher_predict, raw_input + 0.05)

        constrained_teacher = torch.clamp(constrained_teacher, 0, 1)

        score_t = self.iqa_metric(constrained_teacher).detach().cpu().numpy()

        new_positive = positive_list.clone()

        # Approximate a dark-channel validity check with the global minimum.
        img_min_vals = constrained_teacher.amin(dim=(1, 2, 3))  # [B]

        # If the darkest pixel is still too bright, the prediction is likely foggy.
        IS_FOGGY_THRESHOLD = 0.20

        for idx in range(N):
            is_empty_target = positive_list[idx].mean() < 1e-3

            is_clean_enough = img_min_vals[idx] < IS_FOGGY_THRESHOLD
            is_better = score_t[idx] > score_r[idx] + 0.02

            if (is_better and is_clean_enough) or is_empty_target:
                if is_empty_target:
                    new_positive[idx] = constrained_teacher[idx]
                else:
                    new_positive[idx] = 0.8 * positive_list[idx] + 0.2 * constrained_teacher[idx]

                temp = (constrained_teacher[idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                    "uint8"
                )
                Image.fromarray(temp).save(p_name[idx])
                score_r[idx] = score_t[idx]

        return new_positive

    def train(self):
        self.freeze_teachers_parameters()
        if self.args.resume == "True":
            print("Loading a checkpoint from {} ...".format(self.args.resume_path))
            checkpoint = torch.load(self.args.resume_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer_s.load_state_dict(checkpoint["optimizer_dict"])
            if "t_state_dict" in checkpoint:
                self.tmodel.load_state_dict(checkpoint["t_state_dict"])
            # Restore best_psnr when the checkpoint contains it.
            if "best_psnr" in checkpoint:
                self.best_psnr = checkpoint["best_psnr"]
                print(f"Resuming with best PSNR: {self.best_psnr:.4f}")
        else:
            initialize_weights(self.model)

        # if self.start_epoch == 1:
        #     initialize_weights(self.model)
        # else:
        #     checkpoint = torch.load(self.args.resume_path)
        #     self.model.load_state_dict(checkpoint['state_dict'])
        for epoch in range(self.start_epoch, self.epochs + 1):

            if epoch <= self.warmup_epochs:
                # Linear warmup from zero to base_lr.
                lr = self.base_lr * (epoch / self.warmup_epochs)
                for param_group in self.optimizer_s.param_groups:
                    param_group["lr"] = lr
            else:
                pass

            loss_ave, psnr_train, sup_loss, unsup_loss = self._train_epoch(epoch)
            train_psnr = sum(psnr_train) / len(psnr_train)
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            print(
                "[%d] sup_loss: %.6f, unsup_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f"
                % (
                    epoch,
                    sup_loss,
                    unsup_loss,
                    train_psnr,
                    val_psnr,
                    self.lr_scheduler_s.get_last_lr()[0],
                )
            )

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"{name}", param, 0)

            state = {
                "arch": type(self.model).__name__,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "t_state_dict": self.tmodel.state_dict(),
                "optimizer_dict": self.optimizer_s.state_dict(),
                "best_psnr": self.best_psnr,
            }

            # Save the best validation checkpoint.
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                best_ckpt_name = str(self.args.save_path) + "model_best.pth"
                print(
                    f"New best PSNR: {self.best_psnr:.4f}. Saving best model to {best_ckpt_name} ..."
                )
                state["best_psnr"] = self.best_psnr
                torch.save(state, best_ckpt_name)
            # Save checkpoint
            if epoch % self.save_period == 0 and self.args.local_rank <= 0:
                ckpt_name = str(self.args.save_path) + "model_e{}.pth".format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

            # Hand over to cosine decay after warmup.
            if epoch > self.warmup_epochs:
                self.lr_scheduler_s.step()

    def _train_epoch(self, epoch):
        sup_loss_meter = AverageMeter()
        unsup_loss_meter = AverageMeter()
        loss_total_meter = AverageMeter()
        psnr_train = []

        self.model.train()
        self.freeze_teachers_parameters()

        # The unlabeled loader drives the training loop.
        unsupervised_iter = iter(self.unsupervised_loader)
        supervised_iter = iter(self.supervised_loader)

        accumulation_steps = 4

        STAGE_EPOCHS = 10

        for param in self.model.parameters():
            param.requires_grad = True

        tbar = range(len(self.unsupervised_loader))
        tbar = tqdm(tbar, ncols=130, leave=True)

        self.optimizer_s.zero_grad()

        for i in tbar:
            try:
                (unpaired_data_w, unpaired_data_s, p_list, p_name) = next(unsupervised_iter)
            except StopIteration:
                break
            try:
                (img_data, label, img_la) = next(supervised_iter)
            except StopIteration:
                supervised_iter = iter(self.supervised_loader)
                (img_data, label, img_la) = next(supervised_iter)

            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            img_la = Variable(img_la).cuda(non_blocking=True)
            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)

            # Add mild noise for the strongly augmented unlabeled branch.
            noise = torch.randn_like(unpaired_data_s) * 0.01
            unpaired_data_s = unpaired_data_s + noise.cuda(non_blocking=True)

            # Use mixup after the warmup stage to avoid early flare-shape artifacts.
            use_mixup = (np.random.random() < 0.7) and (epoch > STAGE_EPOCHS)
            if use_mixup:
                lam = np.random.beta(1.0, 1.0)
                batch_size = img_data.size(0)
                index = torch.randperm(batch_size).cuda()

                mixed_img = lam * img_data + (1 - lam) * img_data[index, :]
                mixed_label = lam * label + (1 - lam) * label[index, :]
                mixed_la = lam * img_la + (1 - lam) * img_la[index, :]

                img_data = mixed_img
                label = mixed_label
                img_la = mixed_la

            outputs_l, flare_predict = self.model(img_data)

            structure_loss = self.loss_str(outputs_l, label) + self.loss_str(flare_predict, img_la)
            perpetual_loss = self.loss_per(outputs_l, label) + self.loss_per(flare_predict, img_la)
            fft_loss = self.loss_fft(outputs_l, label) + self.loss_fft(flare_predict, img_la) 

            loss_sup = (
                0.5 * structure_loss + 0.5 * perpetual_loss + 0.2 * fft_loss
            )

            (loss_sup / accumulation_steps).backward()

            sup_loss_meter.update(loss_sup.item())

            if outputs_l is not None:
                current_psnr = to_psnr(outputs_l, label)
                psnr_train.extend(current_psnr)
            else:
                psnr_train.append(0.0)

            del img_data, label, img_la, loss_sup
            if "outputs_l" in locals():
                del outputs_l
            if "structure_loss" in locals():
                del structure_loss
            if "perpetual_loss" in locals():
                del perpetual_loss

            # =========================================================
            # PART 2: Unlabeled data
            # =========================================================
            loss_unsup_val = 0.0
            if epoch > STAGE_EPOCHS:
                unpaired_data_w = unpaired_data_w.cuda(non_blocking=True)
                p_list = p_list.cuda(non_blocking=True)

                predict_target_u = self.predict_with_out_grad(unpaired_data_w)

                outputs_ul, _ = self.model(unpaired_data_s)

                score_r = self.iqa_metric(p_list).detach().cpu().numpy()
                p_sample = self.update_dependable_repository(
                    predict_target_u, outputs_ul, p_list, p_name, score_r, unpaired_data_w
                )
                valid_mask = p_sample.mean(dim=(1, 2, 3)) > 1e-3

                if valid_mask.sum().item() > 0:
                    loss_l1 = self.loss_unsup(outputs_ul[valid_mask], p_sample[valid_mask])
                    loss_per = self.loss_per(outputs_ul[valid_mask], p_sample[valid_mask])

                    loss_unsu = loss_l1 + 0.5 * loss_per

                    loss_cr = self.loss_cr(
                        outputs_ul[valid_mask], p_sample[valid_mask], unpaired_data_s[valid_mask]
                    )
                    total_unsup_epochs = self.args.num_epochs - STAGE_EPOCHS
                    current_step = epoch - STAGE_EPOCHS
                    TARGET_CR_WEIGHT = 0.01
                    cr_weight = TARGET_CR_WEIGHT * (current_step / total_unsup_epochs)
                    loss_unsu = loss_unsu + cr_weight * loss_cr

                    effective_epoch = max(0, epoch - STAGE_EPOCHS)
                    consistency_weight = self.get_current_consistency_weight(effective_epoch)
                    scale_factor = 0.3

                    final_unsup_loss = loss_unsu * consistency_weight * scale_factor

                    (final_unsup_loss / accumulation_steps).backward()

                    loss_unsup_val = loss_unsu.item()

                del unpaired_data_w, p_list, predict_target_u, outputs_ul, unpaired_data_s

            unsup_loss_meter.update(loss_unsup_val)

            # Display-only total loss.
            total_loss_display = sup_loss_meter.val + (
                loss_unsup_val * 0.01 if epoch > STAGE_EPOCHS else 0
            )
            loss_total_meter.update(total_loss_display)

            # =========================================================
            # PART 3: Optimizer update with gradient accumulation
            # =========================================================
            if (i + 1) % accumulation_steps == 0:
                self.optimizer_s.step()
                self.optimizer_s.zero_grad()

                # Keep the EMA teacher synchronized with the student.
                with torch.no_grad():
                    self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                    self.curiter += 1

        # Tensorboard Logging
        self.writer.add_scalar("sup_loss", sup_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("unsup_loss", unsup_loss_meter.avg, global_step=epoch)
        self.lr_scheduler_s.step(epoch=epoch - 1)

        return loss_total_meter.avg, psnr_train, sup_loss_meter.avg, unsup_loss_meter.avg

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()

        # tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label) in enumerate(self.val_loader):
                val_data = val_data.cuda()
                val_label = val_label.cuda()
                # forward
                val_output, _ = self.model(val_data)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                psnr_val.extend(to_psnr(val_output, val_label))
                # tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                #     "Eval-Student", epoch, val_psnr.avg, val_ssim.avg))

            self.writer.add_scalar("Val_psnr", val_psnr.avg, global_step=epoch)
            self.writer.add_scalar("Val_ssim", val_ssim.avg, global_step=epoch)
            del val_output, val_label, val_data
            return psnr_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print("No GPUs detected, using the CPU")
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f"Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available")
            n_gpu = sys_gpu
        device = torch.device("cuda:0" if n_gpu > 0 else "cpu")
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
