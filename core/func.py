import torch
import io
import numpy as np
from pathlib import Path
import re
import trimesh
import imageio
from core.colmap import *
from core.graphics import *
import os
from einops import rearrange
import nvdiffrast.torch as dr
import json
import open3d as o3d
from typing import Tuple
from tqdm import tqdm
import torch_scatter
from scipy import stats

def render_mask(glctx, mv: torch.Tensor, mvp: torch.Tensor, vertices: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    V = vertices.shape[0]
    faces = faces.type(torch.int32)
    vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)
    depth_clip = vert_hom @ mv.transpose(-2, -1)
    vertices_clip = vert_hom @ mvp.transpose(-2, -1)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False)
    col, _ = dr.interpolate(colors, rast_out, faces)
    alpha = torch.clamp(rast_out[..., -1:], max=1)
    col = torch.concat((col, alpha), dim=-1)
    col = dr.antialias(col, rast_out, vertices_clip, faces)
    return col

def render(glctx, mv: torch.Tensor, mvp: torch.Tensor, vertices: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    V = vertices.shape[0]
    faces = faces.type(torch.int32)
    vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)
    depth_clip = vert_hom @ mv.transpose(-2, -1)
    vertices_clip = vert_hom @ mvp.transpose(-2, -1)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False)
    verts_depth = -depth_clip[:, :, 2] / depth_clip[:, :, 3]
    verts_depth = verts_depth[0, :, None].contiguous()
    color_all = torch.concat((colors, normals), dim=-1)
    color_all = torch.concat((color_all, verts_depth), dim=-1)
    col, _ = dr.interpolate(color_all, rast_out, faces)
    col = dr.antialias(col, rast_out, vertices_clip, faces)
    return col
import torch_scatter

def render_index(glctx, mvp: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor, image_size: tuple[int, int], fill_value: float=-1.0) -> torch.Tensor:
    B, V, _ = vertices.shape
    H, W = image_size
    vertices_h = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    clip_coords = torch.bmm(vertices_h, mvp.transpose(1, 2))
    ndc = clip_coords[..., :3] / clip_coords[..., 3:4]
    rast_out, _ = dr.rasterize(glctx, ndc, faces, resolution=(H, W))
    vertex_ids = torch.arange(V, device=vertices.device).float()
    vertex_ids_feat = vertex_ids[faces]
    id_feat = vertex_ids_feat[None].expand(B, -1, -1)
    id_img, _ = dr.interpolate(id_feat, rast_out, faces, mode='flat')
    id_img = id_img.squeeze(-1).long()
    mask = rast_out[..., 3] > 0
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=vertices.device), torch.arange(W, device=vertices.device), indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)
    coords = coords[None].expand(B, -1, -1, -1)
    valid_coords = coords[mask]
    valid_ids = id_img[mask]
    batch_ids = torch.arange(B, device=vertices.device).view(-1, 1, 1).expand(-1, H, W)
    valid_batch_ids = batch_ids[mask]
    scatter_index = valid_batch_ids * V + valid_ids
    flat_coords = torch.full((B * V, 2), fill_value, device=vertices.device)
    flat_coords = torch_scatter.scatter(src=valid_coords.float(), index=scatter_index, dim=0, out=flat_coords, reduce='mean')
    image_coords = flat_coords.view(B, V, 2)
    return image_coords
import torch.nn.functional as F

@torch.no_grad()
def compute_image_gradient(imgs: torch.Tensor) -> torch.Tensor:
    assert imgs.dim() == 4, '输入图像应为 (B, C, H, W)'
    B, C, H, W = imgs.shape
    device = imgs.device
    dtype = imgs.dtype
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_x = sobel_x.expand(C, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3)
    imgs_reshaped = imgs.view(B * C, 1, H, W)
    grad_x = F.conv2d(imgs_reshaped, sobel_x, padding=1, groups=1)
    grad_y = F.conv2d(imgs_reshaped, sobel_y, padding=1, groups=1)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-06)
    grad_magnitude = grad_magnitude.view(B, C, H, W)
    return grad_magnitude
import torch.nn.functional as F

def bilinear_sample_avg(image: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = image.shape
    _, V, _ = coords.shape
    visible_mask = (coords != -1).all(dim=-1)
    norm_x = coords[..., 0] / (W - 1) * 2 - 1
    norm_y = coords[..., 1] / (H - 1) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(image, grid, mode='bilinear', align_corners=True, padding_mode='zeros')
    sampled = sampled.squeeze(-1).permute(0, 2, 1)
    sampled[~visible_mask] = 0.0
    visible_count = visible_mask.sum(dim=0).clamp(min=1).unsqueeze(-1)
    values_sum = sampled.sum(dim=0)
    values_avg = values_sum / visible_count
    all_invisible = visible_count.squeeze(-1) == 0
    values_avg[all_invisible] = 0.0
    return values_avg

def compute_vertex_color_gradients(vertices, faces, colors):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.cross(e1, e2, dim=1)
    normal = F.normalize(normal, dim=1)
    tangent = F.normalize(e1, dim=1)
    bitangent = torch.cross(normal, tangent, dim=1)
    T = torch.stack([tangent, bitangent], dim=2)
    c0 = colors[faces[:, 0]]
    c1 = colors[faces[:, 1]]
    c2 = colors[faces[:, 2]]
    dp1 = torch.bmm(e1.unsqueeze(1), T).squeeze(1)
    dp2 = torch.bmm(e2.unsqueeze(1), T).squeeze(1)
    A = torch.stack([dp1, dp2], dim=1)
    dC1 = c1 - c0
    dC2 = c2 - c0
    dC = torch.stack([dC1, dC2], dim=2)
    A_inv = torch.linalg.pinv(A)
    grad2D = torch.bmm(dC, A_inv)
    grad3D = grad2D[..., 0].unsqueeze(-1) * tangent.unsqueeze(1) + grad2D[..., 1].unsqueeze(-1) * bitangent.unsqueeze(1)
    return grad3D

def accumulate_face_to_vertex(faces, face_values, N):
    M, C_dim, _ = face_values.shape
    fv = face_values.repeat_interleave(3, dim=0)
    idx = faces.reshape(-1)
    vertex_grad = torch.zeros((N, C_dim, 3), device=face_values.device)
    counts = torch.zeros((N, 1, 1), device=face_values.device)
    vertex_grad.index_add_(0, idx, fv)
    counts.index_add_(0, idx, torch.ones_like(fv[:, :1, :1]))
    vertex_grad = vertex_grad / (counts + 1e-08)
    return vertex_grad

@torch.no_grad()
def vertex_color_gradient(vertices, faces, colors):
    colors_gray = torch.norm(colors, dim=1, keepdim=True)
    grad_faces = compute_vertex_color_gradients(vertices, faces, colors_gray)
    grad_vertices = accumulate_face_to_vertex(faces, grad_faces, vertices.shape[0])
    grad_dirs = F.normalize(grad_vertices, dim=-1)
    return grad_dirs[:, 0, :]

@torch.no_grad()
def fmap_projection(glctx, mvp: torch.Tensor, fmap: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor, image_size: tuple[int, int], ifrendertest=True):
    V = vertices.shape[0]
    F = faces.shape[0]
    triangles_values = torch.zeros_like(faces[:, 0], dtype=torch.float32).cuda()
    triangles_values_cnt = torch.zeros_like(faces[:, 0], dtype=torch.float32).cuda()
    faces_copy = faces.clone()
    faces = faces.type(torch.int32)
    vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)
    vertices_clip = vert_hom @ mvp.transpose(-2, -1)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False)
    for i in tqdm(range(fmap.shape[0])):
        trig_id = rast_out[i, :1162, :1554, -1] - 1
        indices = trig_id.view(-1).long()
        mask = indices >= 0
        indices = indices[mask].contiguous()
        values = fmap[i].view(-1)[mask].contiguous()
        torch_scatter.scatter_add(values, indices, out=triangles_values)
        torch_scatter.scatter_add(torch.ones_like(values), indices, out=triangles_values_cnt)
        trig_id = None
    tri_mask = triangles_values_cnt > 0
    triangles_values[tri_mask] /= triangles_values_cnt[tri_mask]
    triangles_values = triangles_values[tri_mask]
    triangles_values_cnt = triangles_values_cnt[tri_mask]
    faces_copy = faces_copy[tri_mask]
    F = faces_copy.shape[0]
    vertex_values = torch.zeros((V, 3), dtype=triangles_values.dtype, device=vertices.device)
    vertex_values.scatter_add_(dim=0, index=faces_copy, src=triangles_values[:, None].expand(F, 3))
    vertex_values = vertex_values.sum(dim=1)
    vertex_values_cnt = torch.zeros((V, 3), dtype=triangles_values_cnt.dtype, device=vertices.device)
    vertex_values_cnt.scatter_add_(dim=0, index=faces_copy, src=torch.ones_like(triangles_values)[:, None].expand(F, 3))
    vertex_values_cnt = vertex_values_cnt.sum(dim=1)
    vertex_values[vertex_values_cnt > 0] /= vertex_values_cnt[vertex_values_cnt > 0]
    print(vertex_values[vertex_values_cnt == 0])
    if ifrendertest:
        with torch.no_grad():
            vtx_fft_values = vertex_values[:, None] * 255.0
            fft_col, _ = dr.interpolate(vtx_fft_values, rast_out, faces)
            fft_col = dr.antialias(fft_col, rast_out, vertices_clip, faces)
            fft_col_np = fft_col.detach().cpu().numpy().astype(np.uint8)
            for j in range(fft_col_np.shape[0]):
                img = Image.fromarray(fft_col_np[j, :, :, 0], mode='L')
                img.save(f'fft_col_{i}_{j}.png')
            save_obj(f'fft_col_{i}.obj', vertices, vtx_fft_values.expand_as(vertices) / 255.0, faces)
    return vertex_values

def to_numpy(*args):

    def convert(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        assert a is None or isinstance(a, np.ndarray)
        return a
    return convert(args[0]) if len(args) == 1 else tuple((convert(a) for a in args))

def load_gt_obj(filename: Path, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    filename = Path(filename)
    obj_path = filename.with_suffix('.obj')
    with open(obj_path) as file:
        obj_text = file.read()
    num = '([0-9\\.\\-eE]+)'
    v = re.findall(f'(v {num} {num} {num} {num} {num} {num})', obj_text)
    vertices = np.array(v)[:, 1:].astype(np.float32)
    all_faces = []
    f = re.findall(f'(f {num} {num} {num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 1)[..., :1])
    f = re.findall(f'(f {num}/{num} {num}/{num} {num}/{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 2)[..., :2])
    f = re.findall(f'(f {num}/{num}/{num} {num}/{num}/{num} {num}/{num}/{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 3)[..., :2])
    f = re.findall(f'(f {num}//{num} {num}//{num} {num}//{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 2)[..., :1])
    all_faces = np.concatenate(all_faces, axis=0)
    all_faces -= 1
    faces = all_faces[:, :, 0]
    return (vertices, faces)

def load_obj(filename: Path, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    filename = Path(filename)
    obj_path = filename.with_suffix('.obj')
    with open(obj_path) as file:
        obj_text = file.read()
    num = '([0-9\\.\\-eE]+)'
    v = re.findall(f'(v {num} {num} {num})', obj_text)
    vertices = np.array(v)[:, 1:].astype(np.float32)
    all_faces = []
    f = re.findall(f'(f {num} {num} {num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 1)[..., :1])
    f = re.findall(f'(f {num}/{num} {num}/{num} {num}/{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 2)[..., :2])
    f = re.findall(f'(f {num}/{num}/{num} {num}/{num}/{num} {num}/{num}/{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 3)[..., :2])
    f = re.findall(f'(f {num}//{num} {num}//{num} {num}//{num})', obj_text)
    if f:
        all_faces.append(np.array(f)[:, 1:].astype(np.long).reshape(-1, 3, 2)[..., :1])
    all_faces = np.concatenate(all_faces, axis=0)
    all_faces -= 1
    faces = all_faces[:, :, 0]
    return (vertices, faces)
import open3d as o3d
from PIL import Image

def load_ply_with_open3d(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertices = np.concatenate((vertices, vertex_colors), axis=1)
    return (vertices, faces)

def load_ply_with_open3d_nocolor(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(np.ones_like(vertices))
    vertices = np.concatenate((vertices, vertex_colors), axis=1)
    return (vertices, faces)

def save_obj(filename, vertices, colors, faces):
    with open(filename, 'w') as file:
        for i in range(vertices.shape[0]):
            file.write(f'v {vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n')
        for i in range(faces.shape[0]):
            file.write(f'f {faces[i, 0] + 1} {faces[i, 1] + 1} {faces[i, 2] + 1}\n')

def save_ply(filename: Path, vertices: torch.Tensor, faces: torch.Tensor, vertex_colors: torch.Tensor=None, vertex_normals: torch.Tensor=None):
    filename = Path(filename).with_suffix('.ply')
    vertices, faces, vertex_colors = to_numpy(vertices, faces, vertex_colors)
    assert np.all(np.isfinite(vertices)) and faces.min() == 0 and (faces.max() == vertices.shape[0] - 1)
    header = 'ply\nformat ascii 1.0\n'
    header += 'element vertex ' + str(vertices.shape[0]) + '\n'
    header += 'property double x\n'
    header += 'property double y\n'
    header += 'property double z\n'
    if vertex_normals is not None:
        header += 'property double nx\n'
        header += 'property double ny\n'
        header += 'property double nz\n'
    if vertex_colors is not None:
        assert vertex_colors.shape[0] == vertices.shape[0]
        color = (vertex_colors * 255).astype(np.uint8)
        header += 'property uchar red\n'
        header += 'property uchar green\n'
        header += 'property uchar blue\n'
    header += 'element face ' + str(faces.shape[0]) + '\n'
    header += 'property list int int vertex_indices\n'
    header += 'end_header\n'
    with open(filename, 'w') as file:
        file.write(header)
        for i in range(vertices.shape[0]):
            s = f'{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}'
            if vertex_normals is not None:
                s += f' {vertex_normals[i, 0]} {vertex_normals[i, 1]} {vertex_normals[i, 2]}'
            if vertex_colors is not None:
                s += f' {color[i, 0]:03d} {color[i, 1]:03d} {color[i, 2]:03d}'
            file.write(s + '\n')
        for i in range(faces.shape[0]):
            file.write(f'3 {faces[i, 0]} {faces[i, 1]} {faces[i, 2]}\n')
    full_verts = vertices[faces]

def save_images(images: torch.Tensor, dir: Path):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        imageio.imwrite(dir / f'{i:02d}.png', (images.detach()[i, :, :, :3] * 255).clamp(max=255).type(torch.uint8).cpu().numpy())

def normalize_vertices(vertices: torch.Tensor):
    """shift and resize mesh to fit into a unit sphere"""
    vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    vertices /= torch.norm(vertices, dim=-1).max()
    return vertices

def laplacian(num_verts: int, edges: torch.Tensor) -> torch.Tensor:
    """create sparse Laplacian matrix"""
    V = num_verts
    E = edges.shape[0]
    idx = torch.cat([edges, edges.fliplr()], dim=0).type(torch.long).T
    ones = torch.ones(2 * E, dtype=torch.float32, device=edges.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    deg = torch.sparse.sum(A, dim=1).to_dense()
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    D = torch.sparse.FloatTensor(idx, deg, (V, V))
    return D - A

def _translation(x, y, z, device):
    return torch.tensor([[1.0, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], device=device)

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4, 4], device=device)
    p[0, 0] = 2 * n / (r - l)
    p[0, 2] = (r + l) / (r - l)
    p[1, 1] = 2 * n / (t - b) * (-1 if flip_y else 1)
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2 * f * n) / (f - n)
    p[3, 2] = -1
    return p

def make_star_cameras(az_count, pol_count, distance: float=10.0, r=None, image_size=[512, 512], device='cuda'):
    w, h = (1280, 720)
    f, n = (5.0, 0.1)
    intri_path = os.path.join('/anonymous_data/8d7fb381c44cf673ece0d94c81200ebbff92ac3ac49b9b4bf8d8b5f184b4759f/nvdiff_data', '0', 'cameras.bin')
    extri_path = os.path.join('/anonymous_data/8d7fb381c44cf673ece0d94c81200ebbff92ac3ac49b9b4bf8d8b5f184b4759f/nvdiff_data', '0', 'images.bin')
    cameras = read_intrinsics_binary(intri_path)
    images = read_extrinsics_binary(extri_path)
    test_camera = cameras[1]
    test_image = images[1]
    if r is None:
        r = 1 / distance
    A = az_count
    P = pol_count
    C = A * P
    phi = torch.arange(0, A) * (2 * torch.pi / A)
    phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
    phi_rot[:, 0, 2, 2] = phi.cos()
    phi_rot[:, 0, 2, 0] = -phi.sin()
    phi_rot[:, 0, 0, 2] = phi.sin()
    phi_rot[:, 0, 0, 0] = phi.cos()
    theta = torch.arange(1, P + 1) * (torch.pi / (P + 1)) - torch.pi / 2
    theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
    theta_rot[0, :, 1, 1] = theta.cos()
    theta_rot[0, :, 1, 2] = -theta.sin()
    theta_rot[0, :, 2, 1] = theta.sin()
    theta_rot[0, :, 2, 2] = theta.cos()
    mv = torch.empty((C, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
    mv = _translation(0, 0, -distance, device) @ mv
    return (mv, _projection(r, device))

def make_sphere(level: int=2, radius=1.0, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return (vertices, faces)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w], [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])

def make_colmap_singleview(test_image):
    R = quaternion_to_rotation_matrix(test_image.qvec)
    t = test_image.tvec
    view_matrix = np.hstack((R, t.reshape(3, 1)))
    view_matrix = np.vstack((view_matrix, [0, 0, 0, 1]))
    view_matrix[1:3, :] *= -1
    return view_matrix

def make_costume_cameras(image_size, intr):
    f, n = (100.0, 0.01)
    h, w = image_size
    f_x, f_y, c_x, c_y = intr
    projection_matrix = np.array([[2 * f_x / w, 0, (w - 2 * c_x) / w, 0], [0, -2 * f_y / h, (h - 2 * c_y) / h, 0], [0, 0, (-f - n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]])
    return projection_matrix

def make_colmap_cameras(path='/anonymous_data/8d7fb381c44cf673ece0d94c81200ebbff92ac3ac49b9b4bf8d8b5f184b4759f/data/diffrast_data', image_size=[1168, 1560]):
    intri_path = os.path.join(path, 'sparse', '0', 'cameras.bin')
    extri_path = os.path.join(path, 'sparse', '0', 'images.bin')
    cameras = read_intrinsics_binary(intri_path)
    images = read_extrinsics_binary(extri_path)
    cameras_keys = list(cameras.keys())
    image_keys = list(images.keys())
    test_camera = cameras[cameras_keys[0]]
    images_name = []
    projection_matrix = make_costume_cameras(image_size, test_camera.params)
    view_matrix_allview = []
    for image_id in image_keys:
        test_image = images[image_id]
        images_name.append(test_image.name)
        view_matrix_singleview = make_colmap_singleview(test_image)
        view_matrix_allview.append(view_matrix_singleview)
    return (view_matrix_allview, projection_matrix, images_name)

def make_colmap_cameras_diff_focal(path='/anonymous_data/8d7fb381c44cf673ece0d94c81200ebbff92ac3ac49b9b4bf8d8b5f184b4759f/data/diffrast_data', image_size=[1168, 1560]):
    intri_path = os.path.join(path, 'sparse', '0', 'cameras.bin')
    extri_path = os.path.join(path, 'sparse', '0', 'images.bin')
    cameras = read_intrinsics_binary(intri_path)
    images = read_extrinsics_binary(extri_path)
    cameras_keys = list(cameras.keys())
    image_keys = list(images.keys())
    view_matrix_allview = []
    images_name = []
    for image_id in image_keys:
        test_image = images[image_id]
        images_name.append(test_image.name)
        view_matrix_singleview = make_colmap_singleview(test_image)
        view_matrix_allview.append(view_matrix_singleview)
    projection_matrix_allview = []
    for camera_id in cameras_keys:
        test_camera = cameras[camera_id]
        projection_matrix = make_costume_cameras(image_size, test_camera.params)
        projection_matrix_allview.append(projection_matrix)
    return (view_matrix_allview, projection_matrix_allview, images_name)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def make_blender_cameras(path='/anonymous_data/dataset/nerf_synthetic/hotdog', image_size=[800, 800], extension='.png'):
    with open(os.path.join(path, 'transforms.json')) as json_file:
        contents = json.load(json_file)
        fovx = contents['camera_angle_x']
        frames = contents['frames']
        view_matrix_allview = []
        for idx, frame in enumerate(frames):
            c2w = np.array(frame['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            view_matrix_allview.append(w2c)
        f_x = fov2focal(fovx, 800)
        f_y = f_x
        projection_matrix = make_costume_cameras(image_size, [f_x, f_y, 400, 400])
    return (view_matrix_allview, projection_matrix)

def make_blender_camera_diff_focal(path='/anonymous_data/dataset/nerf_synthetic/hotdog', image_size=[800, 800], extension='.png'):
    with open(os.path.join(path, 'transforms_train.json')) as json_file:
        contents = json.load(json_file)
        fovx = contents['camera_angle_x']
        frames = contents['frames']
        f_x = fov2focal(fovx, image_size[0])
        f_y = f_x
        view_matrix_allview = []
        projection_matrix_allview = []
        images_name = []
        for idx, frame in enumerate(frames):
            c2w = np.array(frame['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            view_matrix_allview.append(w2c)
            images_name.append(str(os.path.basename(frame['file_path'])) + extension)
            projection_matrix = make_costume_cameras(image_size, [f_x, f_y, image_size[0] / 2, image_size[1] / 2])
            projection_matrix_allview.append(projection_matrix)
    return (view_matrix_allview, projection_matrix_allview, images_name)

def _translation(x, y, z, device):
    return torch.tensor([[1.0, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], device=device)

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4, 4], device=device)
    p[0, 0] = 2 * n / (r - l)
    p[0, 2] = (r + l) / (r - l)
    p[1, 1] = 2 * n / (t - b) * (-1 if flip_y else 1)
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2 * f * n) / (f - n)
    p[3, 2] = -1
    return p

def _orthographic(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    o = torch.zeros([4, 4], device=device)
    o[0, 0] = 2 / (r - l)
    o[0, 3] = -(r + l) / (r - l)
    o[1, 1] = 2 / (t - b) * (-1 if flip_y else 1)
    o[1, 3] = -(t + b) / (t - b)
    o[2, 2] = -2 / (f - n)
    o[2, 3] = -(f + n) / (f - n)
    o[3, 3] = 1
    return o

def make_star_cameras(az_count, pol_count, distance: float=10.0, r=None, image_size=[512, 512], device='cuda'):
    if r is None:
        r = 1 / distance
    A = az_count
    P = pol_count
    C = A * P
    phi = torch.tensor([0, 3 / 8, 1 / 4, 1 / 2, 3 / 4, 5 / 8])
    phi = phi * 2 * torch.pi
    phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
    phi_rot[:, 0, 2, 2] = phi.cos()
    phi_rot[:, 0, 2, 0] = -phi.sin()
    phi_rot[:, 0, 0, 2] = phi.sin()
    phi_rot[:, 0, 0, 0] = phi.cos()
    theta = torch.arange(1, P + 1) * (torch.pi / (P + 1)) - torch.pi / 2
    theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
    theta_rot[0, :, 1, 1] = theta.cos()
    theta_rot[0, :, 1, 2] = -theta.sin()
    theta_rot[0, :, 2, 1] = theta.sin()
    theta_rot[0, :, 2, 2] = theta.cos()
    mv = torch.empty((C, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
    mv = _translation(0, 0, -distance, device) @ mv
    flip_matrix = torch.eye(4, device=device)
    flip_matrix[0, 0] = -1
    mv = flip_matrix @ mv
    return (mv, _projection(r, device))

def make_star_cameras_orthographic(az_count, pol_count, distance: float=10.0, r=None, image_size=[256, 256], device='cuda'):
    mv, _ = make_star_cameras(az_count, pol_count, distance, r, image_size, device)
    if r is None:
        r = 1
    image_names = ['rgb_000_back', 'rgb_000_front_left', 'rgb_000_left', 'rgb_000_front', 'rgb_000_right', 'rgb_000_front_right']
    return (mv, _orthographic(r, device), image_names)

def make_sphere(level: int=2, radius=1.0, device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return (vertices, faces)
from pytorch3d.renderer import FoVOrthographicCameras, look_at_view_transform

def get_camera(R, T, focal_length=1 / 2 ** 0.5):
    focal_length = 1 / focal_length
    camera = FoVOrthographicCameras(device=R.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length)
    return camera

def make_star_cameras_orthographic_py3d(azim_list, device, focal=2 / 1.35, dist=1.1):
    R, T = look_at_view_transform(dist, 0, azim_list)
    focal_length = 1 / focal
    return FoVOrthographicCameras(device=R.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length).to(device)

def beta_transform(x: torch.Tensor, alpha: float=0.8, beta: float=0.8) -> torch.Tensor:
    y = torch.clamp(x, 1e-08, 1 - 1e-08).cpu().numpy()
    return torch.tensor(stats.beta.ppf(y, alpha, beta)).float().to(x.device)

def expand_variance_and_clamp(x, var_factor=2, lower=0.0, upper=0.8):
    mean = x.mean()
    std_old = x.std()
    std_new = var_factor ** 0.5 * std_old
    x_rescaled = (x - mean) / (std_old + 1e-08) * std_new + mean
    x_clamped = torch.clamp(x_rescaled, min=lower, max=upper)
    return x_clamped