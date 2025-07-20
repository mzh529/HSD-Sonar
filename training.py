"""
training utils
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta
import kornia
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import diffusers

from eval import evaluate, add_segmentations_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline
import cv2
import lpips
@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    lr_warmup_steps: int = 500
    save_image_epochs: int = 100
    save_model_epochs: int = 1
    mixed_precision: str = 'fp16'  #、 `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume_epoch: int = None

    # EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2 # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3 # w in the paper
    trans_noise_level: float = 0.5 # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    use_cfg_for_eval_conditioning: bool = True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    cfg_maskguidance_condmodel_only: bool = True  # if using mask guidance AND cfg, only give mask to conditional network
    # ^ this is because giving mask to both uncond and cond model make class guidance not work 
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)

def compute_edge_density(batch_images):

    assert batch_images.shape[1] == 1, "Input images must have one channel."
    

    batch_images = batch_images.squeeze(1).cpu().numpy()  # Shape: (batch_size, height, width)
    
    edge_densities = []
    edges_list = []
    count =0
    for image in batch_images:
        count += 1

        # print(f"Image dtype: {image.dtype}, shape: {image.shape}")
        # print(f"Image min: {image.min()}, max: {image.max()}")
        image = (image * 255).astype(np.uint8)
        # print(f"Image dtype: {image.dtype}, shape: {image.shape}")
        # print(f"Image min: {image.min()}, max: {image.max()}")
        # cv2.imwrite(f'/data/mzh/SonarNewSegmentation/segmentation-guided-diffusion/edgs/yuanshi_{count}.png', image)

        edges = cv2.Canny(image, 80, 200)
        # print(f"edges dtype: {edges.dtype}, shape: {edges.shape}")
        # print(f"edges min: {edges.min()}, max: {edges.max()}")

        edges_list.append(edges)
        

        edge_density = np.sum(edges > 0) / edges.size
        edge_densities.append(edge_density)
    
    return edge_densities, edges_list
def save_weight_maps(weight_maps, save_dir):

    if weight_maps.max() > 1:
        weight_maps = weight_maps / weight_maps.max()
    

    batch_size = weight_maps.shape[0]
    for count in range(batch_size):

        weight = weight_maps[count, 0, :, :].cpu().numpy()
        

        weight_uint8 = (weight * 255).astype(np.uint8)
        

        # save_path = f'{save_dir}/weight_{count}.png'
        # cv2.imwrite(save_path, weight_uint8)
def generate_region_weight_map(segmentation_mask, edge_map, importance_map=None):

    # Ensure segmentation_mask is in the range [0, 1]
    if segmentation_mask.max() > 1:
        segmentation_mask = segmentation_mask / segmentation_mask.max()
    
    # Convert edge_map (list of numpy arrays) to a single torch.Tensor
    edge_map_tensor = torch.stack([torch.tensor(em, dtype=torch.float32) for em in edge_map])
    edge_map_tensor = edge_map_tensor.unsqueeze(1)  # Add channel dimension
    
    # Normalize edge_map_tensor to [0, 1]
    edge_map_tensor = edge_map_tensor / edge_map_tensor.max()
    
    # Calculate initial weight map
    weight_map = segmentation_mask * 0.6 + edge_map_tensor * 0.4
    
    # Add semantic importance map if provided
    if importance_map is not None:
        if importance_map.max() > 1:
            importance_map = importance_map / importance_map.max()
        weight_map += importance_map * 0.5
    # save_dir = '/data/mzh/SonarNewSegmentation/segmentation-guided-diffusion/edgs'
    # save_weight_maps(weight_map, save_dir)
    # Normalize final weight map to [0, 1]
    weight_map = weight_map / weight_map.max()
    
    return weight_map


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.cuda().eval()

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.l1_loss(x_features, y_features)

perceptual_loss_fn = VGGPerceptualLoss()
lpips_loss = lpips.LPIPS(net='vgg').cuda()

def compute_loss(predicted_noise, target_noise):

    segmentation_mask = segmentation_mask.expand_as(predicted_noise)  # (B, C, H, W)

    mse_loss = F.mse_loss(predicted_noise * segmentation_mask, target_noise * segmentation_mask)
    # mse_loss = F.mse_loss(predicted_noise , target_noise)

    lpips_loss_val = lpips_loss(predicted_noise, target_noise).mean()


    vgg_loss = perceptual_loss_fn(predicted_noise, target_noise)


    total_loss = mse_loss + 0.1 * lpips_loss_val + 0.05 * vgg_loss
    total_loss = mse_loss + 0.1 * lpips_loss_val
    return total_loss
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    # logging
    run_name = '{}-{}-{}'.format(config.model_type.lower(), config.dataset, config.image_size)
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # for loading segs to condition on:
    # eval_dataloader = iter(eval_dataloader)


    # Now you train the model
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch
    best_loss = float('inf')
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            clean_images = clean_images.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            idx = 0
            loss_tmp = 0

            seg = batch['seg_all']

            print(f"seg dtype: {seg.dtype}, shape: {seg.shape}")


            for i in range(seg.shape[0]): 
                single_image = seg[i]  # Shape: (1, height, width)
                max_val = single_image.max().item() 
                min_val = single_image.min().item()  
                print(f"Image {i}: max={max_val}, min={min_val}")
            seg[seg>0]=1
            print(f"seg dtype: {seg.dtype}, shape: {seg.shape}")


            for i in range(seg.shape[0]):  
                single_image = seg[i]  # Shape: (1, height, width)
                max_val = single_image.max().item()  
                min_val = single_image.min().item()  
                print(f"Image {i}: max={max_val}, min={min_val}")
            edge_density, edge_map = compute_edge_density(seg)
            weight_map = generate_region_weight_map(seg, edge_map)

            weight_map = weight_map / weight_map.max()
            


            num_timesteps = noise_scheduler.config.num_train_timesteps
            timesteps_range = torch.arange(0, num_timesteps, device=clean_images.device)
            

            adjusted_timesteps = (weight_map * (num_timesteps - 1)).long()

            dynamic_timesteps = []
            for i in range(bs):

                sample_timesteps = adjusted_timesteps[i].view(-1).to(device) 
                final_timestep = torch.sum(adjusted_timesteps * weight_map) / torch.sum(weight_map)

                final_timestep = torch.round(final_timestep).long().to(device)
                # final_timestep = final_timestep.view(1).long()
                # print(final_timestep)

                random_timesteps = torch.randint(0, num_timesteps, (1,), device=weight_map.device).long().to(device)
                random_timesteps = random_timesteps.squeeze(0)
                # print(random_timesteps)
                # print(sample_timesteps.device)
                # print(random_timesteps.device)

                # final_timesteps = torch.minimum(random_timesteps, sample_timesteps)
                final_timesteps = torch.round(0.2 * final_timestep + (1 - 0.8) * random_timesteps).long()
                # if final_timesteps > 1000:
                #     final_timesteps = 1000
                final_timesteps = torch.clamp(final_timesteps, max=1000)
                dynamic_timesteps.append(final_timesteps)




            seg = batch['seg_all']
            seg[seg > 0] = 1  


            def compute_sample_edge_density(seg):

                edge_maps = kornia.filters.sobel(seg)  
                edge_maps = (edge_maps > 0.1).float()  
                edge_density = edge_maps.mean(dim=(1,2,3))  
                return edge_density

            edge_density = compute_sample_edge_density(seg)  # [bs]

            save_dir = "/data/mzh/SonarNewSegmentation/segmentation-guided-diffusion/ppt"
            os.makedirs(save_dir, exist_ok=True) 


            # for i in range(seg.shape[0]):
            #     vutils.save_image(seg[i], os.path.join(save_dir, f"seg_{i}.png"))


            # for i in range(edge_maps.shape[0]):
            #     vutils.save_image(edge_maps[i], os.path.join(save_dir, f"edge_map_{i}.png"))

            for i in range(seg.shape[0]):

                seg_np = seg[i].squeeze().cpu().numpy()  
                seg_np = (seg_np * 255).astype(np.uint8) 
                cv2.imwrite(os.path.join(save_dir, f"seg_{i}.png"), seg_np)
            edge_maps = kornia.filters.sobel(seg)  
            edge_maps = (edge_maps > 0.1).float()  

            for i in range(edge_maps.shape[0]):
                edge_np = edge_maps[i].squeeze().cpu().numpy()  
                edge_np = (edge_np * 255).astype(np.uint8)      
                cv2.imwrite(os.path.join(save_dir, f"edge_map_{i}.png"), edge_np)


            num_timesteps = noise_scheduler.config.num_train_timesteps  # 如 1000
            bs = seg.shape[0]


            base_timesteps = (edge_density * (num_timesteps - 1)).long()  # [bs]


            mix_ratio = 0.7  
            random_timesteps = torch.randint(0, num_timesteps, (bs,), device=seg.device)
            dynamic_timesteps = (
                mix_ratio * base_timesteps + 
                (1 - mix_ratio) * random_timesteps
            ).long().clamp(0, num_timesteps-1)  # [bs]

            noise = torch.randn_like(clean_images)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, dynamic_timesteps) 
            timesteps = torch.stack(dynamic_timesteps, dim=0)
            # print(timesteps)
            # # Sample a random timestep for each image
            # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            # print(timesteps)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(noisy_images, batch, config, device)
                
            # Predict the noise residual
            if config.class_conditional:
                class_labels = torch.ones(noisy_images.size(0)).long().to(device)
                # classifier-free guidance
                a = np.random.uniform()
                if a <= config.cfg_p_uncond:
                    class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
            else:
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = compute_loss(noise_pred, noise)
            # loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            loss_tmp += loss.item()
            idx += 1


            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # also train on target domain images if conditional
            # (we don't have masks for this domain, so we can't do segmentation-guided; just use blank masks)
            if config.class_conditional:
                target_domain_images = batch['images_target']
                target_domain_images = target_domain_images.to(device)

                # Sample noise to add to the images
                noise = torch.randn(target_domain_images.shape).to(target_domain_images.device)
                bs = target_domain_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=target_domain_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(target_domain_images, noise, timesteps)

                if config.segmentation_guided:
                    # no masks in target domain so just use blank masks
                    noisy_images = torch.cat((noisy_images, torch.zeros_like(noisy_images)), dim=1)

                # Predict the noise residual
                class_labels = torch.full([noisy_images.size(0)], 2).long().to(device)
                # classifier-free guidance
                a = np.random.uniform()
                if a <= config.cfg_p_uncond:
                    class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
                loss_target_domain = F.mse_loss(noise_pred, noise)
                loss_target_domain.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()



            progress_bar.update(1)
            if config.class_conditional:
                logs = {"loss": loss.detach().item(), "loss_target_domain": loss_target_domain.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                writer.add_scalar("loss_target_domain", loss_target_domain.detach().item(), global_step)
            else: 
                
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            writer.add_scalar("loss", loss.detach().item(), global_step)
            if global_step % 10 == 0:
                writer.flush()

            progress_bar.set_postfix(**logs)
            global_step += 1
        # epoch_loss = loss_tmp / len(train_dataloader)
        # writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
        # writer.flush()
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                    )
            else:
                if config.class_conditional:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDPM")
                else:
                    pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
        elif config.model_type == "DDIM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                    )
            else:
                if config.class_conditional:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDIM")
                else:
                    pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)
        loss_ave = loss_tmp / idx
                    
        if loss_ave < best_loss:
            best_loss = loss_ave

            pipeline.save_pretrained(config.output_dir)
            print(f"Model saved at epoch {epoch} with loss {best_loss}")


        model.eval()

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            if config.segmentation_guided:
                # try:
                # seg_batch = next(eval_dataloader)
                # print(len(eval_dataloader))
                for seg_batch in eval_dataloader:
                    evaluate(config, epoch, pipeline, seg_batch)
                # except StopIteration:
                #     print("No more batches in eval_dataloader, skipping evaluation.")
            else:
                evaluate(config, epoch, pipeline)

        # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        #     pipeline.save_pretrained(config.output_dir)