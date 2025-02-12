import pickle as pkl
import re

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from hloc.utils.read_write_model import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from tqdm import tqdm

from utils.smpl.smpl_server import SMPLServer


def estimate_translation_cv2(
    joints_3d,
    joints_2d,
    focal_length=600,
    img_size=np.array([512.0, 512.0]),
    proj_mat=None,
    cam_dist=None,
):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0, 0], camK[1, 1] = focal_length, focal_length
        camK[:2, 2] = img_size // 2
    else:
        camK = proj_mat
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        joints_3d,
        joints_2d,
        camK,
        cam_dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=20,
        iterationsCount=100,
    )

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:, 0]
        return tra_pred


# from neuman
def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


def dump_romp_estimates(romp_output_dir):
    vibe_estimates = {
        "verts": [],
        "joints3d": [],
        "joints2d_img_coord": [],
        "pose": [],
        "betas": [],
        "trans": [],
    }

    files = sorted(romp_output_dir.glob("*.pkl"))
    for file in files:
        cur_res = pkl.load(open(file, "rb"))
        vibe_estimates["verts"].append(cur_res["verts"])
        vibe_estimates["joints3d"].append(cur_res["jnts_3d"])
        vibe_estimates["joints2d_img_coord"].append(cur_res["jnts_2d"])
        vibe_estimates["pose"].append(cur_res["pose"])
        vibe_estimates["betas"].append(cur_res["shape"])
        vibe_estimates["trans"].append(cur_res["trans"])

    for k, v in vibe_estimates.items():
        vibe_estimates[k] = np.array(v).astype(np.float32)

    return vibe_estimates


def read_smpl(raw_smpl_path):
    assert raw_smpl_path.is_dir()
    vibe_estimates = dump_romp_estimates(raw_smpl_path)
    return vibe_estimates


def solve_translation(p3d, p2d, mvp):
    p3d = torch.from_numpy(p3d.copy()).float()
    p2d = torch.from_numpy(p2d.copy()).float()
    mvp = torch.from_numpy(mvp.copy()).float()
    translation = torch.zeros_like(p3d[0:1, 0:3], requires_grad=True)
    optim_list = [
        {"params": translation, "lr": 1e-3},
    ]
    optim = torch.optim.Adam(optim_list)

    total_iters = 1000
    for _ in tqdm(range(total_iters), total=total_iters):
        xyzw = torch.cat([p3d[:, 0:3] + translation, torch.ones_like(p3d[:, 0:1])], axis=1)
        camera_points = torch.matmul(mvp, xyzw.T).T
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(image_points, p2d)
        loss.backward()
        optim.step()
    print(
        "loss",
        loss.detach().cpu().numpy(),
        "translation",
        translation.detach().cpu().numpy(),
    )
    return translation.clone().detach().cpu().numpy()


def solve_scale(joints_world, plane_model, c2w):
    cam_center = c2w[:3, 3]

    a, b, c, d = plane_model
    scales = []
    for j in joints_world:
        jx, jy, jz = j
        # from open3d plane model is: a*x + b*y + c*z + d = 0
        # and a^2 + b^2 + c^2 = 1
        # We can convert the scale problem into a ray-plane intersection problem:
        # reference: https://education.siggraph.org/static/HyperGraph/raytrace/rayplane_intersection.htm
        # Shoting a ray from camera center, (c_x, c_y, c_z), passing through joint, (j_x, j_y, j_z), and
        # Intersecting the plane, a*x + b*y + c*z + d = 0, at some point (x, y, z)
        # Let R0 = (c_x, c_y, c_z)
        #     Rd = (j_x-c_x, j_y-c_y, j_z-c_z)
        # The ray can be written as: R(s) = R0 + s * Rd
        # with the plane equation:
        # a*(c_x + s*(j_x-c_x)) + b*(c_y + s*(j_y-c_y)) + c*(c_z + s*(j_z-c_z)) + d = 0
        # s = -(a*c_x + b*c_y + c*c_z + d) / (a*(j_x-c_x) + b*(j_y-c_y) + c*(j_z-c_z))
        # let right = -(a*c_x + b*c_y + c*c_z + d)
        #       coe = a*(j_x-c_x) + b*(j_y-c_y) + c*(j_z-c_z)
        right = -(a * cam_center[0] + b * cam_center[1] + c * cam_center[2] + d)
        coe = a * (jx - cam_center[0]) + b * (jy - cam_center[1]) + c * (jz - cam_center[2])
        s = right / coe
        if s > 0:
            scales.append(s)
    return min(scales)


def solve_transformation(verts, plane_model, c2w):
    verts_world_ori = (to_homogeneous(verts) @ c2w.T)[:, :3]
    scale = solve_scale(verts_world_ori, plane_model, c2w)
    transf = np.eye(4) * scale
    transf[3, 3] = 1
    transf = transf @ c2w[:3, :].T
    verts_world = to_homogeneous(verts) @ transf
    return verts_world, scale


def transform_verts_with_scale(verts, c2w, scale):
    transf = np.eye(4) * scale
    transf[3, 3] = 1
    transf = transf @ c2w[:3, :].T
    verts_world = to_homogeneous(verts) @ transf
    return verts_world


# according to https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0
def convert_smpl_to_world(raw_smpl, index, scale, T_c_w, ref_smpl_verts, new_smpls, gender):
    smpl_pose = raw_smpl["pose"][index]
    smpl_betas = raw_smpl["betas"][index]
    smpl_trans = raw_smpl["trans"][index]
    smpl_server = SMPLServer(gender=gender)
    scale = torch.tensor([scale], dtype=torch.float32)
    R_c = cv2.Rodrigues(smpl_pose[:3])[0]
    # t_c = smpl_trans
    R_c_w = T_c_w[:3, :3]
    # t_c_w = T_c_w[:3, 3]
    R_w = R_c_w @ R_c
    R_w = cv2.Rodrigues(R_w)[0].squeeze()
    # p = (
    #     smpl_server.smpl.get_T_hip(betas=torch.tensor(smpl_betas)[None].float())
    #     .squeeze()
    #     .cpu()
    #     .numpy()
    # )
    # TODO: this t_w is not correct when scale is not 1.0. Need to figure out why later
    # right now we just use 0 translation
    # and later get the translation between two point clouds directly
    t_w = [0.0, 0.0, 0.0]
    # t_w = R_c_w @ (p + t_c) + t_c_w - p
    # smpl_pose[:3] = R_w
    new_smpl_pose = np.zeros_like(smpl_pose)
    new_smpl_pose[:3] = R_w
    new_smpl_pose[3:] = smpl_pose[3:]
    smpl_output = smpl_server(
        scale[None],
        torch.tensor(t_w)[None],
        torch.tensor(new_smpl_pose)[None],
        torch.tensor(smpl_betas)[None],
    )
    new_t_w = ref_smpl_verts - smpl_output["smpl_verts"][0].cpu().numpy()
    new_t_w = new_t_w.mean(axis=0) / scale
    new_smpls["scale"].append(scale.numpy())
    new_smpls["pose"].append(new_smpl_pose)
    new_smpls["shape"].append(smpl_betas)
    new_smpls["trans"].append(new_t_w.numpy())
    return scale, new_t_w, new_smpl_pose, smpl_betas


def visualize_different_scales(scene, plane, smpls, gender, c2ws):
    cmap = matplotlib.cm.get_cmap("Spectral")
    smpl_pcds = []
    smpl_server = SMPLServer(gender=gender)
    scales = np.linspace(0.2, 0.6, 5)
    random_idx = np.random.randint(0, len(smpls["pose"]))
    c2w = c2ws[f"{random_idx:04d}.png"]
    for i, scale in enumerate(scales):
        scale = np.array(scale)
        pose = smpls["pose"][random_idx]
        shape = smpls["betas"][random_idx]
        trans = smpls["trans"][random_idx]
        assert isinstance(scale, np.ndarray)
        assert isinstance(pose, np.ndarray)
        assert isinstance(shape, np.ndarray)
        assert isinstance(trans, np.ndarray)
        smpl_output = smpl_server(
            torch.tensor(scale)[None],
            torch.tensor(trans)[None],
            torch.tensor(pose)[None],
            torch.tensor(shape)[None],
        )
        verts = smpl_output["smpl_verts"][0]
        verts_world_ori = (to_homogeneous(verts) @ c2w.T)[:, :3]
        smpl_pcd = o3d.geometry.PointCloud()
        smpl_pcd.points = o3d.utility.Vector3dVector(verts_world_ori.numpy())
        smpl_pcd.paint_uniform_color(cmap(i / len(scales))[:3])
        smpl_pcds.append(smpl_pcd)
    o3d.visualization.draw_geometries([scene, plane] + smpl_pcds)


def visualize_smpl_and_scene(background, smpls, gender, sphere_radius=-1.0):
    cmap = matplotlib.cm.get_cmap("Spectral")
    smpl_pcds = []
    smpl_pcds_np = []
    smpl_server = SMPLServer(gender=gender)
    num_smpls = len(smpls["scale"])
    for i in range(num_smpls):
        scale = smpls["scale"][i]
        pose = smpls["pose"][i]
        shape = smpls["shape"][i]
        trans = smpls["trans"][i]
        assert isinstance(scale, np.ndarray)
        assert isinstance(pose, np.ndarray)
        assert isinstance(shape, np.ndarray)
        assert isinstance(trans, np.ndarray)
        smpl_output = smpl_server(
            torch.tensor(scale)[None],
            torch.tensor(trans)[None],
            torch.tensor(pose)[None],
            torch.tensor(shape)[None],
        )
        smpl_pcd = o3d.geometry.PointCloud()
        smpl_pcd.points = o3d.utility.Vector3dVector(smpl_output["smpl_verts"][0].numpy())
        smpl_pcd.paint_uniform_color(cmap(i / num_smpls)[:3])
        smpl_pcds.append(smpl_pcd)
        smpl_pcds_np.append(smpl_output["smpl_verts"][0].numpy())
    if sphere_radius > 0.0:
        bounding_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        bounding_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        background.append(bounding_sphere)

    o3d.visualization.draw_geometries(background + smpl_pcds)
    return


def read_intrinsics_binary(sparse_dir):
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    assert len(cameras) == 1
    model = cameras[1].model
    params = cameras[1].params
    if model == "SIMPLE_PINHOLE":
        fl_x = params[0]
        fl_y = params[0]
        cx = params[1]
        cy = params[2]
    elif model == "PINHOLE":
        fl_x = params[0]
        fl_y = params[1]
        cx = params[2]
        cy = params[3]
    elif model == "SIMPLE_RADIAL":
        fl_x = params[0]
        fl_y = params[0]
        cx = params[1]
        cy = params[2]
        print("We skip radial distortion for SIMPLE_RADIAL")
    else:
        raise NotImplementedError
    intrinsics = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
    return intrinsics


# assume only one camera
def load_intrinsic(sparse_dir):
    with open(sparse_dir / "cameras.txt", "rt") as f:
        # angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            # only one camera (monocular video) from undistorted images
            # if len(els) == 0:
            #     continue
            assert els[0] == "1"
            assert els[1] == "SIMPLE_PINHOLE" or els[1] == "PINHOLE"
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            # elif els[1] == "SIMPLE_RADIAL":
            #     cx = float(els[5])
            #     cy = float(els[6])
            #     k1 = float(els[7])
            # elif els[1] == "RADIAL":
            #     cx = float(els[5])
            #     cy = float(els[6])
            #     k1 = float(els[7])
            #     k2 = float(els[8])
            # elif els[1] == "OPENCV":
            #     fl_y = float(els[5])
            #     cx = float(els[6])
            #     cy = float(els[7])
            #     k1 = float(els[8])
            #     k2 = float(els[9])
            #     p1 = float(els[10])
            #     p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])

    intrinsic = np.zeros((3, 3), dtype=np.float32)
    intrinsic[0, 0] = fl_x
    intrinsic[1, 1] = fl_y
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    intrinsic[2, 2] = 1
    return intrinsic


def read_extrinsics_binary(sparse_dir):
    images = read_images_binary(sparse_dir / "images.bin")
    c2ws = {}
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    suffix = images[1].name.split(".")[1]
    for i in range(len(images)):
        id = i + 1
        image = images[id]
        assert image.id == id
        if "colmap" not in str(sparse_dir):
            # in colmap image id are not aligned with image name, but the order of registration
            assert image.name == f"{i:04d}.{suffix}"
        qvec = image.qvec
        tvec = image.tvec
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2ws[image.name] = c2w.astype(np.float32)
    return c2ws


def load_extrinsics(sparse_dir):
    c2ws = {}
    w2cs = {}
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    with open(sparse_dir / "images.txt", "rt") as f:
        i = 0
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i += 1
            if i % 2 == 1:
                elems = line.split(" ")
                assert len(elems) == 10
                # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                fname = elems[9]  # assume to be in the format of {idx:04d}.png
                if len(fname) == 9:
                    fname = fname[1:]
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(qvec)
                t = tvec.reshape([3, 1])
                w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(w2c)
                w2cs[fname] = w2c
                c2ws[fname] = c2w

    return c2ws, w2cs


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# from https://github.com/apple/ml-neuman/blob/main/data_io/colmap_helper.py
def read_point_cloud(points_txt_path):
    with open(points_txt_path, "r") as fid:
        line = fid.readline()
        assert line == "# 3D point list with one line of data per point:\n"
        line = fid.readline()
        assert (
            line == "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        line = fid.readline()
        assert re.search("^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$", line)
        num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        num_points = int(num_points)
        mean_track_length = float(mean_track_length)

        xyz = np.zeros((num_points, 3), dtype=np.float32)
        rgb = np.zeros((num_points, 3), dtype=np.float32)

        for i in tqdm(range(num_points), desc="reading point cloud"):
            elems = fid.readline().split()
            xyz[i] = list(map(float, elems[1:4]))
            rgb[i] = list(map(float, elems[4:7]))
        pcd = np.concatenate([xyz, rgb], axis=1)

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    sparse_pcd.colors = o3d.utility.Vector3dVector(pcd[:, 3:6] / 255.0)
    return sparse_pcd


def read_point_cloud_binary(sparse_dir, img_id=None):
    points = read_points3D_binary(sparse_dir / "points3D.bin")
    xyz = []
    rgb = []
    for key in points:
        if img_id is None:
            xyz.append(points[key].xyz)
            rgb.append(points[key].rgb)
        else:
            if img_id in points[key].image_ids:
                xyz.append(points[key].xyz)
                rgb.append(points[key].rgb)
        # rgb.append([255, 0, 0])
    xyz = np.array(xyz).astype(np.float32)
    rgb = np.array(rgb).astype(np.float32)
    sparse_pcd = o3d.t.geometry.PointCloud()
    sparse_pcd.point.positions = xyz
    sparse_pcd.point.colors = rgb / 255.0
    # sparse_pcd.point.positions = o3d.utility.Vector3dVector(xyz)
    # sparse_pcd.point.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    return sparse_pcd
