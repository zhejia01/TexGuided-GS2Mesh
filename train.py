
import argparse
import os
from pytorch3d.loss import chamfer_distance
import sys
import numpy as np
from sympy import im
import torch
from core import func
from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
import nvdiffrast.torch as dr
from PIL import Image
from tqdm import tqdm
from loss_utils import l1_loss, ssim
from image_utils import psnr
import torchvision.transforms.functional as tf
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat((pos, torch.ones(pos.shape[0],1,device='cuda')),axis=-1)
    return posw @ t_mtx.transpose(-2,-1)


def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

def run_remeshgs(max_iter          = 5000,
             out_dir           = 'exp',
             use_opengl        = False,
             obj_interval      = 20,
             input             = None,
             batch_size        = 32,
             depth_w           = 1.,
             normal_w          = 1.,
             rgb_w             = 1.,
             camera            = None):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'save_obj'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'save_img'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'save_n'), exist_ok=True)



    vertices,faces =func.load_ply_with_open3d(os.path.join(input,'fuse_post.ply'))
    pos_idx = faces

    vtxp = vertices[:,:3]
    vtxc = vertices[:,3:]

    vtx_pos = vertices[:,:3]

    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))



    pos_idx = torch.from_numpy(pos_idx.astype(np.int64)).cuda().requires_grad_(False)                             
    vtx_pos_opt = torch.from_numpy(vtxp.astype(np.float32)).cuda().requires_grad_(True)                    
    vtx_col_opt = torch.from_numpy(vtxc.astype(np.float32)).cuda().requires_grad_(True)                 
    image_size = [1168,1560]
    mv,proj,images_name = func.make_colmap_cameras_diff_focal(path=camera,image_size=image_size)
    proj = torch.from_numpy(np.array(proj)).float().cuda()
    mv = np.array(mv)

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    images = []
    gtfile = 'images'
    visfile = 'vis'
    fftfile = 'fft'

    images = []
    for filename in images_name:
        image_name = filename.split('.')[0]+'.png'
        image_path = os.path.join(os.path.join(camera,gtfile), image_name)
        image = Image.open(image_path)
        images.append(image)
    images = torch.stack([torch.tensor(np.array(image)) for image in images])              
    print(images.shape)
    color_gt = images[...,:3] / 255.0                                                                                              
    mask = (images[...,[3]] / 255.0)
    color_gt[mask.expand_as(color_gt)==0] = 0
    color_gt = color_gt.detach().numpy()

    images = []
    for filename in images_name:
        image_name = 'normal_'+ filename.split('.')[0]+'.png'
        image_path = os.path.join(os.path.join(input,visfile), image_name)
        image = Image.open(image_path)
        images.append(image)
    images = torch.stack([torch.tensor(np.array(image)) for image in images])              
    normal_gt = images / 255.0
    normal_gt[mask.expand_as(normal_gt)==0] = 0
    normal_gt = normal_gt.detach().numpy()                                                                                                    


    images = []
    for filename in images_name:
        image_name = 'depth_'+ filename.split('.')[0]+'.tiff'
        image_path = os.path.join(os.path.join(input,visfile), image_name)
        image = Image.open(image_path)
        images.append(image)
    images = torch.stack([torch.tensor(np.array(image)) for image in images]).unsqueeze(-1)              
    depth_gt = images
    depth_gt[mask[:,:,:,0]==0] = 0
    depth_gt = depth_gt.detach().numpy()
    depth_max = depth_gt.max()
    depth_min = depth_gt[depth_gt>1e-10].min()
    depth_gt = (depth_gt - depth_min) / (depth_max - depth_min)
    images = []
    for filename in images_name:
        image_name = filename.split('.')[0] + '_frequency_energy.png.npy'
        image_path = os.path.join(os.path.join(input,fftfile), image_name)
        image = np.load(image_path)
        images.append(image)
    images = torch.stack([torch.tensor(np.array(image)) for image in images]).to('cuda')              
    fft_gt = (images - images.min()) / (images.max() - images.min())                    

    mv = torch.Tensor(mv).float().cuda()
    r_mvp = torch.matmul(proj, mv)
    render_mvp = r_mvp.cuda()
    vtx_fft = func.fmap_projection(glctx, r_mvp, fft_gt, vtx_pos_opt, pos_idx, image_size=image_size ,ifrendertest=False)
    opt = MeshOptimizer(vtx_pos_opt.detach(), vtx_col_opt.detach(), pos_idx.detach(), vtx_fft.detach(), lr=0.0005, col_lr=0.0005)

    loss_list = []
    mse = torch.nn.MSELoss()

    vtx_mesh_opt = (calc_vertex_normals(vtx_pos_opt, pos_idx)+1)/2
    for i in range(len(images_name)):
        with torch.no_grad():
            image_name = images_name[i]
            r_mv = torch.Tensor(mv[i]).cuda()
            r_mv = r_mv.unsqueeze(0)
            proj_i = proj
            r_mvp = torch.matmul(proj_i, r_mv)
            render_mvp = r_mvp.cuda()
            all_opt  = func.render(glctx, r_mv, render_mvp, vtx_pos_opt, vtx_col_opt, vtx_mesh_opt, pos_idx, image_size=image_size)
            all_opt = all_opt[:,:1162,:1554,:]
            color_opt = all_opt[...,:3].permute(0,3,1,2)
            normal_opt = all_opt[...,3:6].permute(0,3,1,2)
            depth_opt = all_opt[...,[6]].permute(0,3,1,2)
            depth_opt = (depth_opt - depth_min) / (depth_max - depth_min)
            color_opt = color_opt[0].permute(1,2,0).cpu().numpy()
            normal_opt = normal_opt[0].permute(1,2,0).cpu().numpy() 
            depth_opt = depth_opt[0].permute(1,2,0).cpu().numpy()
            color_opt = (color_opt * 255).astype(np.uint8)
            normal_opt = (normal_opt * 255).astype(np.uint8)
            depth_opt = (depth_opt * 255).astype(np.uint8)
            Image.fromarray(color_opt).save(os.path.join(out_dir, f'save_img/init_color_{image_name}'))
            Image.fromarray(normal_opt).save(os.path.join(out_dir, f'save_n/init_normal_{image_name}'))
            Image.fromarray(depth_opt.squeeze()).save(os.path.join(out_dir, f'save_img/depth_{image_name}'))
    torch.cuda.empty_cache()
    color_gt = torch.tensor(color_gt, dtype=torch.float32, device='cuda', requires_grad=False).permute(0,3,1,2)                           
    normal_gt = torch.tensor(normal_gt, dtype=torch.float32, device='cuda', requires_grad=False).permute(0,3,1,2)                           
    depth_gt = torch.tensor(depth_gt, dtype=torch.float32, device='cuda', requires_grad=False).permute(0,3,1,2)
    mv = torch.Tensor(mv).float().cuda()
    for it in tqdm(range(max_iter+ 1)):

        opt.zero_grad()

        indices = np.random.choice(normal_gt.shape[0], batch_size, replace=False)
        r_mv = mv[indices]
        proj_i = proj
        r_mvp = torch.matmul(proj_i, r_mv)
        color = color_gt[indices]
        normal = normal_gt[indices]
        depth = depth_gt[indices]

        render_mvp = r_mvp
        vtx_pos_opt = opt.vertices
        vtx_col_opt = opt.colors
        pos_idx = opt.faces

        mesh = Meshes(verts=[vtx_pos_opt], faces=[pos_idx])
        laplacian_loss = mesh_laplacian_smoothing(mesh,method='uniform')
        normal_consist_loss = mesh_normal_consistency(mesh)
        mesh_reg = laplacian_loss + normal_consist_loss
        vtx_mesh_opt = (calc_vertex_normals(vtx_pos_opt, pos_idx)+1)/2
        all_opt  = func.render(glctx, r_mv, render_mvp, vtx_pos_opt, vtx_col_opt, vtx_mesh_opt, pos_idx, image_size=image_size)
        all_opt = all_opt[:,:1162,:1554,:]
        color_opt = all_opt[...,:3].permute(0,3,1,2)
        color_opt = torch.clamp(color_opt, 0, 1)
        normal_opt = all_opt[...,3:6].permute(0,3,1,2)
        depth_opt = all_opt[...,[6]].permute(0,3,1,2)
        depth_opt = (depth_opt - depth_min) / (depth_max - depth_min)

        rgb_loss =   0.8 *  l1_loss(color_opt, color) + 0.2 * (1.0 - ssim(color_opt, color))
        normal_loss = mse(normal, normal_opt)
        depth_loss = mse(depth,depth_opt)
        edge_align_loss = opt.edge_color_alignment_loss()
        loss = rgb_loss*rgb_w  + mesh_reg*0.3  + normal_loss* normal_w + 0.3 * edge_align_loss + depth_loss*depth_w
        loss_dict = {
            'rgb_loss': rgb_loss.item(),
            'normal_loss': normal_loss.item(),
            'total_loss': loss.item(),
            'depth_loss': depth_loss.item()
        }
        loss_list.append(loss_dict)
        loss.backward()

        opt.step()


        vtx_pos_opt, vtx_col_opt, pos_idx = opt.remesh()                    
        vtx_col_opt = torch.clamp(vtx_col_opt, 0, 1)
        with torch.no_grad():
            display_obj = obj_interval and (it % obj_interval == 0)
            if display_obj:
                filename = os.path.join(out_dir, f'save_obj/save-{it:04d}.ply')
                func.save_ply(filename=filename, vertices=vtx_pos_opt, faces=pos_idx, vertex_colors=vtx_col_opt)


    vtx_mesh_opt = (calc_vertex_normals(vtx_pos_opt, pos_idx)+1)/2
    for i in range(len(images_name)):
        with torch.no_grad():
            image_name = images_name[i]
            r_mv = torch.Tensor(mv[i]).cuda()
            r_mv = r_mv.unsqueeze(0)
            proj_i = proj
            r_mvp = torch.matmul(proj_i, r_mv)
            render_mvp = r_mvp.cuda()
            all_opt  = func.render(glctx, r_mv, render_mvp, vtx_pos_opt, vtx_col_opt, vtx_mesh_opt, pos_idx, image_size=image_size)
            all_opt = all_opt[:,:1162,:1554,:]
            color_opt = all_opt[...,:3].permute(0,3,1,2)
            normal_opt = all_opt[...,3:6].permute(0,3,1,2)
            depth_opt = all_opt[...,[6]].permute(0,3,1,2)
            depth_opt = (depth_opt - depth_min) / (depth_max - depth_min)
            color_opt = color_opt[0].permute(1,2,0).cpu().numpy()
            normal_opt = normal_opt[0].permute(1,2,0).cpu().numpy() 
            depth_opt = depth_opt[0].permute(1,2,0).cpu().numpy()
            color_opt = (color_opt * 255).astype(np.uint8)
            normal_opt = (normal_opt * 255).astype(np.uint8)
            depth_opt = (depth_opt * 255).astype(np.uint8)
            Image.fromarray(color_opt).save(os.path.join(out_dir, f'save_img/color_{image_name}'))
            Image.fromarray(normal_opt).save(os.path.join(out_dir, f'save_n/normal_{image_name}'))
            Image.fromarray(depth_opt.squeeze()).save(os.path.join(out_dir, f'save_img/depth_{image_name}'))


    losses = [loss['total_loss'] for loss in loss_list]
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.savefig(os.path.join(out_dir,'loss_plot.png'))
    plt.close()

    rgb_losses = [loss['rgb_loss'] for loss in loss_list]
    plt.plot(rgb_losses)
    plt.xlabel('Iteration')
    plt.ylabel('RGB Loss')
    plt.title('RGB Loss vs Iteration')
    plt.savefig(os.path.join(out_dir,'rgb_loss_plot.png'))
    plt.close()

    normal_losses = [loss['normal_loss'] for loss in loss_list]
    plt.plot(normal_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Normal Loss')
    plt.title('Normal Loss vs Iteration')
    plt.savefig(os.path.join(out_dir,'normal_loss_plot.png'))
    plt.close()






def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--output', help='specify output directory', default='output')
    parser.add_argument('--max-iter', type=int, default=2000)
    parser.add_argument('--obj-interval', type=int, default=500)
    parser.add_argument('--input', type=str, default='./')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--depth_w', type=float, default=0.0)
    parser.add_argument('--normal_w', type=float, default=1.)
    parser.add_argument('--rgb_w', type=float, default=1.)
    parser.add_argument('--camera', type=str)

    args = parser.parse_args()


    run_remeshgs(
        max_iter=args.max_iter,
        out_dir=args.output,
        use_opengl=args.opengl,
        obj_interval=args.obj_interval,
        input=args.input,
        batch_size=args.batch_size,
        depth_w=args.depth_w,
        normal_w=args.normal_w,
        rgb_w=args.rgb_w,
        camera=args.camera
    )


if __name__ == "__main__":
    main()
