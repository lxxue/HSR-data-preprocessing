import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SfMPerspectiveCameras,
    SoftPhongShader,
)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes


class Renderer:
    def __init__(self, img_size, cam_intrinsic):

        super().__init__()
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        principal_point = [
            -(self.cam_intrinsic[0, 2] - self.image_size[1] / 2.0) / (self.image_size[1] / 2.0),
            -(self.cam_intrinsic[1, 2] - self.image_size[0] / 2.0) / (self.image_size[0] / 2.0),
        ]
        self.principal_point = torch.tensor(principal_point, device=self.device).unsqueeze(0)

        self.cam_R = (
            torch.from_numpy(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))
            .cuda()
            .float()
            .unsqueeze(0)
        )

        self.cam_T = torch.zeros((1, 3)).cuda().float()

        half_max_length = max(self.cam_intrinsic[0:2, 2])
        self.focal_length = torch.tensor(
            [
                (self.cam_intrinsic[0, 0] / half_max_length).astype(np.float32),
                (self.cam_intrinsic[1, 1] / half_max_length).astype(np.float32),
            ]
        ).unsqueeze(0)

        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            device=self.device,
        )

        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 0.0]],
            ambient_color=((1, 1, 1),),
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
        )
        # self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        self.raster_settings = RasterizationSettings(
            image_size=self.render_img_size,
            faces_per_pixel=10,
            blur_radius=0,
            max_faces_per_bin=30000,
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def set_camera(self, R, T):
        cam_R = R.clone()
        cam_T = T.clone()
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :2] *= -1.0
        cam_R = torch.transpose(self.cam_R, 1, 2)
        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=cam_R,
            T=cam_T,
            device=self.device,
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_recon(self, verts, faces, R=None, T=None, colors=None, mode="npat"):
        """
        mode: normal, phong, texture
        """
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = -torch.tensor([0, 0, -1]).float().to(verts.device)
            shades = (
                (normals * front_light.view(1, 1, 3))
                .sum(-1)
                .clamp(min=0)
                .unsqueeze(-1)
                .expand(-1, -1, 3)
            )
            results = []
            # shading
            if "p" in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)
            # normal
            if "n" in mode:
                normals_vis = normals * 0.5 + 0.5  # -1*normals* 0.5 + 0.5
                normals_vis = normals_vis[:, :, [2, 1, 0]]
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # albedo
            if "a" in mode:
                assert colors is not None
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)

            # albedo*shading
            if "t" in mode:
                assert colors is not None
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return torch.cat(results, dim=1)


def render_trimesh(renderer, mesh, R, T, mode="np"):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, ..., :3] / 255
    renderer.set_camera(R, T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)

    return image
