"""
SMPL Parameter Refinement

This script refines ROMP-estimated SMPL parameters using temporal consistency and 2D keypoint
reprojection losses. It renders the optimized SMPL meshes and saves both parameters and
visualizations.

Arguments:
    --data_dir: output directory for the processed data
    --sparse_dir: path to COLMAP sparse reconstruction
    --gender: SMPL model gender (male/female)
    --smpl_model_path: path to SMPL model directory (default: checkpoints/smpl/)

Output Structure:
    data_dir/
    └── smpl/
        └── refined_ROMP/
            ├── *.pkl          # refined SMPL parameters
            └── *.png          # visualization of refined mesh
"""

import argparse
import glob
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from smplx.body_models import SMPL
from tqdm import tqdm

from utils.align_human_scene_utils import estimate_translation_cv2, read_intrinsics_binary
from utils.refine_romp_loss import get_loss_weights, joints_2d_loss, pose_temporal_loss
from utils.refine_romp_utils import PerspectiveCamera, smpl_to_pose
from utils.render_utils import Renderer, render_trimesh

smpl2op_mapping = torch.tensor(
    smpl_to_pose(
        model_type="smpl",
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        openpose_format="coco25",
    ),
    dtype=torch.long,
).cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--sparse_dir", type=Path, required=True)
    parser.add_argument("--gender", type=str, choices=["male", "female"])
    parser.add_argument("--smpl_model_path", type=Path, default=Path("checkpoints/smpl/"))
    args = parser.parse_args()
    data_dir = args.data_dir
    gender = args.gender
    sparse_dir = args.sparse_dir
    smpl_model_path = args.smpl_model_path

    img_dir = data_dir / "images" / "selected_frames"
    file_dir = data_dir / "smpl" / "ROMP"
    openpose_dir = data_dir / "keypoints" / "openpose"
    img_paths = sorted(glob.glob(f"{img_dir}/*.[pj][np]g"))
    file_paths = sorted(glob.glob(f"{file_dir}/*.npz"))
    openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))

    device = torch.device("cuda:0")
    smpl_model = SMPL(smpl_model_path, gender=gender).to(device)

    input_img = cv2.imread(img_paths[0])
    img_h, img_w = input_img.shape[:2]

    cam_intrinsics = read_intrinsics_binary(sparse_dir)

    cam_extrinsics = np.eye(4)
    render_R = torch.tensor(cam_extrinsics[:3, :3])[None].float()
    render_T = torch.tensor(cam_extrinsics[:3, 3])[None].float()

    renderer = Renderer(img_size=[img_h, img_w], cam_intrinsic=cam_intrinsics)
    cam = PerspectiveCamera(
        focal_length_x=torch.tensor(cam_intrinsics[0, 0], dtype=torch.float32),
        focal_length_y=torch.tensor(cam_intrinsics[1, 1], dtype=torch.float32),
        center=torch.tensor(cam_intrinsics[0:2, 2]).unsqueeze(0),
    ).to(device)
    weight_dict = get_loss_weights()
    overlay = True
    smooth = False
    skip_optim = False
    mean_shape = []
    last_j3d = None
    if not skip_optim:
        output_dir = data_dir / "smpl" / "refined_ROMP"
        output_dir.mkdir(exist_ok=True, parents=False)
        for idx, img_path in enumerate(tqdm(img_paths)):
            input_img = cv2.imread(img_path)
            seq_file = np.load(file_paths[idx], allow_pickle=True)["results"][()]
            actor_id = np.argmax(seq_file["center_confs"])
            if len(seq_file["smpl_thetas"]) >= 2:
                # assert False, "there should be only one person in the video"
                print(f"detect multiple persons in frame {idx}, please check this frame!!!")
            openpose = np.load(openpose_paths[idx])
            openpose[:, -1][openpose[:, -1] < 0.01] = 0.0

            smpl_pose = seq_file["smpl_thetas"][actor_id]
            # smpl_trans = [0.,0.,0.] # seq_file['trans'][0][idx]
            smpl_shape = seq_file["smpl_betas"][actor_id][:10]
            smpl_verts = seq_file["verts"][actor_id]
            pj2d_org = seq_file["pj2d_org"][actor_id]
            joints3d = seq_file["joints"][actor_id]
            last_j3d = joints3d.copy()
            # tranform to perspective projection
            tra_pred = estimate_translation_cv2(joints3d, pj2d_org, proj_mat=cam_intrinsics)

            # cam_extrinsics[:3, 3] = tra_pred # cam_trans
            smpl_trans = tra_pred
            P = cam_intrinsics @ cam_extrinsics[:3, :]

            num_iters = 150

            openpose_j2d = torch.tensor(
                openpose[:, :2][None],
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
            openpose_conf = torch.tensor(
                openpose[:, -1][None],
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )

            opt_betas = torch.tensor(
                smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device
            )
            opt_pose = torch.tensor(
                smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device
            )
            opt_trans = torch.tensor(
                smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device
            )

            opt_params = [
                {"params": opt_betas, "lr": 1e-3},
                {"params": opt_pose, "lr": 1e-3},
                {"params": opt_trans, "lr": 1e-3},
            ]
            optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
            if idx == 0:
                last_pose = [opt_pose.detach().clone()]
            loop = tqdm(range(num_iters))
            for it in loop:
                tmp_img = input_img.copy()
                optimizer.zero_grad()

                smpl_output = smpl_model(
                    betas=opt_betas,
                    body_pose=opt_pose[:, 3:],
                    global_orient=opt_pose[:, :3],
                    transl=opt_trans,
                )
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                smpl_joints_3d = torch.index_select(smpl_output.joints, 1, smpl2op_mapping)
                smpl_joints_2d = cam(smpl_joints_3d)
                # for jth in range(0, smpl_joints_2d.shape[1]):
                #     output_img = cv2.circle(tmp_img, tuple(smpl_joints_2d[0].data.cpu().numpy().astype(np.int32)[jth, :2]), 3, (0,0,255), -1)
                # cv2.imwrite('{DIR}/{seq}/init_refined_smpl/smpl_2d_%04d.png' % it, output_img)

                loss = dict()
                loss["J2D_Loss"] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf)
                # loss["Temporal_Loss"] = pose_temporal_loss(last_pose[0], opt_pose)
                loss["Temporal_Loss"] = pose_temporal_loss(last_pose[0], opt_pose)
                # loss['FOOT_Prior_Loss'] = foot_prior_loss(opt_pose[:, 21:27])
                # loss['Prior_Loss'] = pose_prior_loss(opt_pose[:, 3:], opt_betas)
                w_loss = dict()
                for k in loss:
                    w_loss[k] = weight_dict[k](loss[k], it)

                tot_loss = list(w_loss.values())
                tot_loss = torch.stack(tot_loss).sum()
                tot_loss.backward()
                optimizer.step()

                l_str = "Iter: %d" % it
                for k in loss:
                    l_str += ", %s: %0.4f" % (
                        k,
                        weight_dict[k](loss[k], it).mean().item(),
                    )
                    loop.set_description(l_str)

            smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
            rendered_image = render_trimesh(renderer, smpl_mesh, render_R, render_T, "n")
            crop_start = abs(input_img.shape[0] - input_img.shape[1]) // 2
            crop_end = (input_img.shape[0] + input_img.shape[1]) // 2
            if input_img.shape[0] < input_img.shape[1]:
                rendered_image = rendered_image[crop_start:crop_end, :, :]
            else:
                rendered_image = rendered_image[:, crop_start:crop_end, :]
            valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]
            if overlay:
                output_img = rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)
                output_img = output_img.astype(np.uint8)
                h, w = output_img.shape[:2]
                if h > w:
                    concat_axis = 1
                else:
                    concat_axis = 0
                output_img = np.concatenate([input_img, output_img], axis=concat_axis)
                cv2.imwrite(str(output_dir / f"{idx:04d}.png"), output_img)

            last_pose.pop(0)
            last_pose.append(opt_pose.detach().clone())
            smpl_dict = {}
            smpl_dict["pose"] = opt_pose.data.squeeze().cpu().numpy()
            smpl_dict["trans"] = opt_trans.data.squeeze().cpu().numpy()
            smpl_dict["shape"] = opt_betas.data.squeeze().cpu().numpy()
            smpl_dict["jnts_2d"] = smpl_joints_2d.data.squeeze().cpu().numpy()
            smpl_dict["jnts_3d"] = smpl_joints_3d.data.squeeze().cpu().numpy()
            smpl_dict["verts"] = smpl_verts

            mean_shape.append(smpl_dict["shape"])
            with open(str(output_dir / f"{idx:04d}.pkl"), "wb") as f:
                pkl.dump(smpl_dict, f)
