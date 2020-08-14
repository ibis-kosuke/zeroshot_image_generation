import torch
from torch import nn
from torchvision.models import inception_v3

import os
import pickle as pkl
import numpy as np
from scipy import linalg
import argparse
import glob
import cv2

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        print("loading inception_v3...")
        self.inception_network = inception_v3(pretrained=True)
        print("load end.")
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
         # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations

def get_activation(images, model, gpus, batch_size=20):
    image_num = images.shape[0]
    num_batches = (image_num+batch_size-1) // batch_size
    all_activations = np.zeros((image_num, 2048))
    for i in range(num_batches):
        im_batch = images[i*batch_size:(i+1)*batch_size]
        im_batch = im_batch.cuda()
        #activation = nn.parallel.data_parallel(model, (im_batch), gpus)###parallel
        activation = model(im_batch)
        activation = activation.view(im_batch.shape[0], 2048)
        activation = activation.detach().cpu().numpy()
        all_activations[i*batch_size:(i+1)*batch_size] = activation

    return all_activations


def calc_statistics(images, model, gpus):
    activation = get_activation(images, model, gpus)
    mu = np.mean(activation, axis=0)
    cov = np.cov(activation, rowvar=False)
    return mu, cov

def calc_fid_score(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
   


def load_fake_images(args, dir_names):
    epoch_path = os.path.join(args.data_dir, "att_output", args.netG, "valid", args.model_epoch, args.split)
    stage_all_paths = {0:[], 1:[], 2:[]}
    for dir_name in dir_names:
        img_dir_path = os.path.join(epoch_path, dir_name)
        for stage in range(3):
            img_paths = glob.glob(os.path.join(img_dir_path, "stage_%d/*.png" % stage))
            stage_all_paths[stage].extend(img_paths)
    stage_1_imgs = load_np_images(stage_all_paths[0])
    stage_2_imgs = load_np_images(stage_all_paths[1])
    stage_3_imgs = load_np_images(stage_all_paths[2])
    return stage_1_imgs, stage_2_imgs, stage_3_imgs


def load_np_images(all_image_paths):
    ##image: 3x299x299(RGB), -1~1
    all_images = np.zeros((len(all_image_paths), 3, 299, 299))
    for i, path in enumerate(all_image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (299,299))
        img = img[:,:,::-1]
        img = img.astype(np.float32) / 255
        img = 2*img-1
        img = np.transpose(img, (2,0,1))
        all_images[i,:,:,:] = img
    all_images = torch.FloatTensor(all_images)
    return all_images

def load_real_images(args, dir_names):
    images_path = os.path.join(args.data_dir, "CUB_200_2011/images")
    all_image_paths = []
    for dir_name in dir_names:
        im_path = glob.glob(os.path.join(images_path, dir_name, "*.jpg"))
        all_image_paths.extend(im_path)
    all_images = load_np_images(all_image_paths)

    return all_images


def load_dir_names(args):
    if args.split=="test_seen":
        file_path = os.path.join(args.data_dir, "test_seen/class_data.pickle")
    else:
        file_path = os.path.join(args.data_dir, "test_unseen/class_data.pickle")
    if os.path.isfile(file_path):
        with open(file_path,"rb") as f:
            class_data = pkl.load(f)
    dir_names = class_data['classes']
    return dir_names


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/data/unagi0/ktokitake/birds_attngan")
    parser.add_argument("--netG",type=str, default="birds_attn2_2020_08_03_14_39_16")
    parser.add_argument("--m_epoch", type=str, dest="model_epoch", default="netG_epoch_500")
    parser.add_argument("--split", choices=["test_seen", "test_unseen"], default="test_seen")
    parser.add_argument("--gpus", type=str, default="0,1,2")

    args = parser.parse_args()

    args.gpus = list(map(int, args.gpus.split(",")))
    torch.cuda.set_device(args.gpus[0])

    dir_names = load_dir_names(args)
    real_images = load_real_images(args, dir_names)
    fake_images_s1, fake_images_s2, fake_images_s3 = load_fake_images(args, dir_names)
    ### batch x 3 x 299 x 299, -1~1, RGB, tensor
    model = PartialInceptionNetwork()
    model.cuda()
    model.eval()
    
    print("now calcurating real") 
    mu_real, cov_real = calc_statistics(real_images, model, args.gpus)
    print("now calcurating stage1")
    mu_s1, cov_s1 = calc_statistics(fake_images_s1, model, args.gpus)
    print("now calcurating stage2")
    mu_s2, cov_s2 = calc_statistics(fake_images_s2, model, args.gpus)
    print("now calcurating stage3")
    mu_s3, cov_s3 = calc_statistics(fake_images_s3, model, args.gpus)
    fid_s1 = calc_fid_score(mu_real, cov_real, mu_s1, cov_s1)
    fid_s2 = calc_fid_score(mu_real, cov_real, mu_s2, cov_s2)
    fid_s3 = calc_fid_score(mu_real, cov_real, mu_s3, cov_s3)
    print("FID_score stage_1: %.3f, stage_2: %.3f, stage_3: %.3f" % (fid_s1, fid_s2, fid_s3))
    
    
    