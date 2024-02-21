from __future__ import print_function, division
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from glob import glob
import core.stereo_datasets as datasets
from core.aeacv_stereo import AEACVStereo, autocast
from core.utils.utils import InputPadder
from core.utils import frame_utils
import skimage.io
import cv2
DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='KITTI15_testing', 
                            iters=32, mixed_prec=False, inference_size=None,
                            ):
    """ create submission for the KITTI leaderboard """
    model.eval()

    image_set='testing'
    # image_set='training'
    root_15 = '/mnt/bn/depth-data-bn/stereo_data/KITTI/Kitti15'
    image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
    image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
    assert len(image1_list) == len(image2_list)
    torch.backends.cudnn.benchmark = True

    num_samples = len(image1_list)
    print('Number of test samples for KITTI: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for id in tqdm(range(num_samples)):
        image1 = load_image(image1_list[id])
        image2 = load_image(image2_list[id])

        if inference_size is None:
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
        else:
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size = inference_size, mode='bilinear', align_corners=True)
        # warpup to measure inference time
        if id == 0:
            for _ in range(5):
                model(image1, image2, iters=iters, test_mode=True)
        
        with autocast(enabled=mixed_prec):
            start = time.time()
            disp = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()
        # remove padding
        if inference_size is None:
            disp = padder.unpad(disp)[0]  # [H, W]
        else:
            # resize back
            disp = F.interpolate(disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            disp = disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, image1_list[id].split('/')[-1])
        skimage.io.imsave(save_name, (disp.cpu().numpy().squeeze() * 256.).astype(np.uint16))


def test_correct_storage_KITTI(save_path):
    root_15 = '/mnt/bn/depth-data-bn/stereo_data/KITTI/Kitti15'
    disp_list = sorted(glob(os.path.join(root_15, 'training/disp_occ_0/*_10.png')))
    pred_list = sorted(glob(os.path.join(save_path, '*.png')))
    out_list, epe_list, elapsed_list = [], [], []
    for disp_path, pred_path in zip(disp_list, pred_list):
        disp = np.array(Image.open(disp_path)).astype(np.int16)/ 256.
        pred = np.array(Image.open(pred_path)).astype(np.int16)/ 256.
        epe = np.sqrt((disp-pred)**2)
        epe_flattened = epe.flatten()
        val = ((disp > 0).flatten() >= 0.5) & (np.abs(disp).flatten() < 192)
        out = (epe_flattened > 3.0)
        # import pdb;pdb.set_trace()
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val])
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    print(f"Validation KITTI: EPE {epe}, D1 {d1}")


@torch.no_grad()
def create_eth3d_submission(model,
                            output_path='ETH3D_tesing', 
                            iters=32, mixed_prec=False, inference_size=None,
                            ):
    """ create submission for the ETH3Dleaderboard """
    model.eval()

    image_set='testing'
    root = '/mnt/bn/depth-data-bn/stereo_data/ETH3D'
    image1_list = sorted( glob(os.path.join(root, f'two_view_{image_set}/*/im0.png')) )
    image2_list = sorted( glob(os.path.join(root, f'two_view_{image_set}/*/im1.png')) )
    assert len(image1_list) == len(image2_list)
    torch.backends.cudnn.benchmark = True

    num_samples = len(image1_list)
    print('Number of test samples for ETH3D: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for id in range(num_samples):
        image1 = load_image(image1_list[id])
        image2 = load_image(image2_list[id])

        if inference_size is None:
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
        else:
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size = inference_size, mode='bilinear', align_corners=True)
        
        # warpup to measure inference time
        if id == 0:
            for _ in range(5):
                model(image1, image2, iters=iters, test_mode=True)
        
        with autocast(enabled=mixed_prec):
            torch.cuda.synchronize()
            start = time.perf_counter()
            disp = model(image1, image2, iters=iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start
        # remove padding
        if inference_size is None:
            disp = padder.unpad(disp)[0]  # [H, W]
        else:
            # resize back
            disp = F.interpolate(disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            disp = disp * ori_size[-1] / float(inference_size[-1])

        file_name = os.path.basename(os.path.dirname(image1_list[id]))
        # save visualization of result
        os.makedirs(output_path.replace('submission_eth3d', 'vis_of_eth3d'), exist_ok=True)
        vis_save_name = os.path.join(output_path.replace('submission_eth3d', 'vis_of_eth3d'),  file_name + '.png')
        disp = disp.cpu().numpy().squeeze()
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        depth_colormap = cv2.applyColorMap(disp_vis.astype("uint8"), cv2.COLORMAP_JET)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(disp.cpu().numpy().squeeze(), alpha=1/4., beta=0), cv2.COLORMAP_JET)
        cv2.imwrite(vis_save_name, depth_colormap)

        # save disp result
        save_name = os.path.join(output_path, file_name + '.pfm')
        frame_utils.write_pfm(save_name, disp.cpu().numpy().squeeze())

        # save running time
        save_runtime_name = os.path.join(output_path, file_name + '.txt')
        with open(save_runtime_name, 'w') as f:
            f.write('runtime ' + str(inference_time))


@torch.no_grad()
def create_middlebury_submission(model,
                            resolution = 'F',
                            output_path='middlebury_tesing', 
                            iters=32, mixed_prec=False, inference_size=None,
                            ):
    """ create submission for the middlebury leaderboard """
    model.eval()

    image_set = f'test{resolution}'
    root = '/mnt/bn/depth-data-bn/stereo_data/Middlebury/MiddEval3'
    image1_list = sorted(glob(os.path.join(root, f'{image_set}/*/im0.png')) )
    image2_list = sorted(glob(os.path.join(root, f'{image_set}/*/im1.png')) )
    assert len(image1_list) == len(image2_list)
    torch.backends.cudnn.benchmark = True

    num_samples = len(image1_list)
    print('Number of test samples for middlebury: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for id in range(num_samples):
        image1 = load_image(image1_list[id])
        image2 = load_image(image2_list[id])

        if inference_size is None:
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
        else:
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size = inference_size, mode='bilinear', align_corners=True)

        # warpup to measure inference time
        if id == 0:
            for _ in range(5):
                model(image1, image2, iters=iters, test_mode=True)

        with autocast(enabled=mixed_prec):
            torch.cuda.synchronize()
            start = time.perf_counter()
            disp = model(image1, image2, iters=iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start
        # remove padding
        if inference_size is None:
            disp = padder.unpad(disp)[0]  # [H, W]
        else:
            # resize back
            disp = F.interpolate(disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            disp = disp * ori_size[-1] / float(inference_size[-1])

        file_name = os.path.basename(os.path.dirname(image1_list[id]))

        # save viusalization result
        os.makedirs(output_path.replace('submission_middlebury', 'vis_of_middlebury'), exist_ok=True)
        vis_save_name = os.path.join(output_path.replace('submission_middlebury', 'vis_of_middlebury'),  file_name + '.png')
        disp = disp.cpu().numpy().squeeze()
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        depth_colormap = cv2.applyColorMap(disp_vis.astype("uint8"), cv2.COLORMAP_JET)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=1., beta=0), cv2.COLORMAP_JET)
        cv2.imwrite(vis_save_name, depth_colormap)
        # save disp result
        save_disp_dir = os.path.join(output_path, file_name)
        os.makedirs(save_disp_dir, exist_ok=True)
        save_name = os.path.join(save_disp_dir,  'disp0_ACV.pfm')
        frame_utils.write_pfm(save_name, disp)
        # save running time
        save_runtime_name = os.path.join(output_path, file_name, 'time_ACV.txt')
        with open(save_runtime_name, 'w') as f:
            f.write(str(inference_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for submission", default='kitti', choices=["eth3d", "kitti"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--submit_path', default='submit_results', help='the path for the storage of submission results ')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--sigma', type=int, default=32, help="the initial sigma of gaussian")
    parser.add_argument('--error_aware_corr_windows', type=int, default=1, help="number of windows for each disparity plane when compute error aware correlation")
    parser.add_argument('--error_aware_corr_planes', type=int, default=9, help="number of disparity planes when compute error aware correlation")
    parser.add_argument('--error_aware_corr_groups', type=int, default=4, help="number of group when compute error aware correlation")
    args = parser.parse_args()

    model = torch.nn.DataParallel(AEACVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':

        create_eth3d_submission(model, output_path=args.submit_path, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':

        create_kitti_submission(model, output_path=args.submit_path, iters=args.valid_iters, mixed_prec=use_mixed_precision)
        # test_correct_storage_KITTI(args.submit_path)
    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:

        create_middlebury_submission(model, resolution=args.dataset[-1], output_path=args.submit_path, iters=args.valid_iters, mixed_prec=use_mixed_precision)
