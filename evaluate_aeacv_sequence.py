import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

import core.stereo_datasets as datasets
from core.aeacv_stereo import AEACVStereo, autocast
from core.utils.utils import InputPadder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--save_path', help="save path", type=str, default='./inference/aeacv')

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
    os.makedirs(args.save_path, exist_ok=True)

    # Set the model
    model = torch.nn.DataParallel(AEACVStereo(args), device_ids=[0])

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        print("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        print("Done loading checkpoint")
    
    model.cuda()
    model.eval()
    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    with open(args.save_path + "/model.txt", "w") as file:
        file.write("Parameter Count: %d\n" % count_parameters(model))
        file.write(model.__str__())

    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_dict = {}
    epe_dict = {}
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=args.mixed_precision):
            with torch.no_grad():
                flow_prs = model(image1, image2, iters=args.valid_iters, test_mode=True, test_sequence=True)
        
        for idx, flow_pr in enumerate(flow_prs):
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

            epe = torch.abs(flow_pr - flow_gt)

            epe = epe.flatten()
            val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

            if (np.isnan(epe[val].mean().item())):
                continue

            out = (epe > 3.0)
            if val_id == 0:
                epe_dict.update({
                    int(idx): [epe[val].mean().item()]
                })
                out_dict.update({
                    int(idx): [out[val].cpu().numpy()]
                })
            else:
                epe_dict[int(idx)].append(epe[val].mean().item())
                out_dict[int(idx)].append(out[val].cpu().numpy())

    for k, v in epe_dict.items():
        epe_dict[k] = np.array(epe_dict[k])
        epe_dict[k] = np.mean(epe_dict[k])
    for k, v in out_dict.items():
        out_dict[k] = np.concatenate(out_dict[k])
        out_dict[k] = 100 * np.mean(out_dict[k])
    
    with open(args.save_path + "/test.txt", "w") as file:
        for ((k1, v1), (k2, v2)) in zip(epe_dict.items(), out_dict.items()):
            file.write("Validation SceneFlow -> Seq %d: %f, %f\n" % (k1 + 1, v1, v2))
