#pip install grad-cam

from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def denormalize_batch(batch):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose([
        # T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        T.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist()
        ),
        T.Lambda(lambda x: x.permute(0, 2, 3, 1))])

    return inv_transform(batch)


def plot_grad_images(model, target_layers, data_images, pred_labels, target_labels, targets, cifar10_labels_dict):
    targets = [ClassifierOutputTarget(i) for i in targets]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    mod_inp_images = mod_inp_images = torch.stack([i for i in data_images])

    grayscale_cams = cam(input_tensor=mod_inp_images, targets=targets, aug_smooth=True)

    rgb_imgs = denormalize_batch(mod_inp_images).cpu().squeeze().detach().numpy()

    figure = plt.figure(figsize=(12, 5))
    no_images = 10
    for index in range(1, no_images + 1):
        plt.subplot(2, 5, index)
        visualization = show_cam_on_image(rgb_imgs[index - 1], grayscale_cams[index - 1, :], use_rgb=True)
        plt.imshow(visualization)
        title = "Target:" + str(cifar10_labels_dict[target_labels[index - 1]]) + "\nPredicted:" + str(
            cifar10_labels_dict[pred_labels[index - 1]])
        plt.title(title)
