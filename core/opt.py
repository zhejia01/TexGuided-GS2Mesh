from copy import deepcopy
import time
import torch
import torch_scatter
from core import func
from core.remesh import calc_edge_length, calc_edges, calc_face_collapses, calc_face_normals, calc_vertex_normals, collapse_edges, flip_edges, pack, prepend_dummies, remove_dummies, split_edges

@torch.no_grad()
def remesh(
        vertices_etc:torch.Tensor,     
        colors:torch.Tensor,     
        ffts:torch.Tensor,    
        faces:torch.Tensor,          
        min_edgelen:torch.Tensor,   
        max_edgelen:torch.Tensor,   
        flip:bool,
        colors_gradient:torch.Tensor,
        max_vertices=1e7,
        ):

    vertices_etc,colors,ffts,faces,colors_gradient = prepend_dummies(vertices_etc,colors,ffts,faces,colors_gradient)
    vertices = vertices_etc[:,:3]     
    nan_tensor = torch.tensor([torch.nan],device=min_edgelen.device)
    min_edgelen = torch.concat((nan_tensor,min_edgelen))
    max_edgelen = torch.concat((nan_tensor,max_edgelen))

    edges,face_to_edge = calc_edges(faces)         
    edge_length = calc_edge_length(vertices,edges)   
    face_normals = calc_face_normals(vertices,faces,normalize=False)     
    vertex_normals = calc_vertex_normals(vertices,faces,face_normals)     
    face_collapse = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edgelen,area_ratio=0.5)
    shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0)                              
    priority = face_collapse.float() + shortness
    vertices_etc,colors,ffts,faces,colors_gradient = collapse_edges(vertices_etc,colors,ffts,faces,edges,priority,colors_gradient)
    if vertices.shape[0]<max_vertices:
        edges,face_to_edge = calc_edges(faces)         
        vertices = vertices_etc[:,:3]     
        edge_length = calc_edge_length(vertices,edges)   
        splits = edge_length > max_edgelen[edges].mean(dim=-1)
        vertices_etc,colors,ffts,faces,colors_gradient= split_edges(vertices_etc,colors,ffts,faces,edges,face_to_edge,splits,colors_gradient,pack_faces=False)
    vertices_etc,colors,ffts,faces,colors_gradient = pack(vertices_etc,colors,ffts,faces,colors_gradient)
    vertices = vertices_etc[:,:3]

    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True)         
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    return remove_dummies(vertices_etc,colors,ffts,faces,colors_gradient)
def lerp_unbiased(a:torch.Tensor,b:torch.Tensor,weight:float,step:int):
    """lerp with adam's bias correction"""
    c_prev = 1-weight**(step-1)
    c = 1-weight**step
    a_weight = weight*c_prev/c
    b_weight = (1-weight)/c
    a.mul_(a_weight).add_(b, alpha=b_weight)


class MeshOptimizer:
    """Use this like a pytorch Optimizer, but after calling opt.step(), do vertices,faces = opt.remesh()."""

    def __init__(self, 
            vertices:torch.Tensor,     
            colors:torch.Tensor,     
            faces:torch.Tensor,     
            ffts:torch.Tensor,     
            lr=0.001,               
            betas=(0.8,0.8,0),                                                                                                  
            gammas=(0,0,0),                                                                                                 
            nu_ref=0.003,                                               
            edge_len_lims=(.0008,.008),                                                                                                                  
            edge_len_tol=.5,                                                 
            gain=.2,                                        
            laplacian_weight=.2,                                        
            ramp=1,                                                            
            grad_lim=10.,                                            
            remesh_interval=1,                                                         
            local_edgelen=True,                                                                   
            col_betas=(0.9,0.99),                              
            col_eps=1e-6,                            
            col_lr=0.001,                                      
            ):
        self._vertices = vertices
        self._colors = colors
        self._faces = faces
        self._lr = lr
        self._betas = betas
        self._gammas = gammas
        self._nu_ref = nu_ref
        self._edge_len_lims = edge_len_lims
        self._edge_len_tol = edge_len_tol
        self._gain = gain
        self._laplacian_weight = laplacian_weight
        self._ramp = ramp
        self._grad_lim = grad_lim
        self._remesh_interval = remesh_interval
        self._local_edgelen = local_edgelen
        self._step = 0
        self._start = time.time()

        V = self._vertices.shape[0]
        self._vertices_etc = torch.zeros([V,9],device=vertices.device)
        self._split_vertices_etc()
        self.vertices.copy_(vertices)                     

        self._colors_gradient = torch.zeros([V,4],device=colors.device)
        self._split_colors_gradient()
        self._vertices.requires_grad_()
        self._colors.requires_grad_()
        self._ref_len.fill_(edge_len_lims[1])

        self._col_betas = col_betas
        self._col_eps = col_eps
        self._col_lr = col_lr

        self._ffts = 1.0 - ffts



    @property
    def vertices(self):
        return self._vertices
    @property
    def colors(self):
        return self._colors

    @property
    def faces(self):
        return self._faces
    @property
    def ffts(self):
        return self._ffts

    def _split_vertices_etc(self):
        self._vertices = self._vertices_etc[:,:3]
        self._m2 = self._vertices_etc[:,3]
        self._nu = self._vertices_etc[:,4]
        self._m1 = self._vertices_etc[:,5:8]
        self._ref_len = self._vertices_etc[:,8]
        with_gammas = any(g!=0 for g in self._gammas)
        self._smooth = self._vertices_etc[:,:8] if with_gammas else self._vertices_etc[:,:3]

    def _split_colors_gradient(self):
        self._col_G = self._colors_gradient[:,0]                
        self._col_M = self._colors_gradient[:,1:4]                  

    def zero_grad(self):
        self._vertices.grad = None
        self._colors.grad = None

    @torch.no_grad()
    def step(self):
        eps = 1e-8

        self._step += 1

        edges,_ = calc_edges(self._faces)     
        E = edges.shape[0]
        edge_smooth = self._smooth[edges]       
        neighbor_smooth = torch.zeros_like(self._smooth)     
        torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(E*2,-1),index=edges.reshape(E*2,1),dim=0,out=neighbor_smooth)
        if self._gammas[0]:
            self._m1.lerp_(neighbor_smooth[:,5:8],self._gammas[0])
        if self._gammas[1]:
            self._m2.lerp_(neighbor_smooth[:,3],self._gammas[1])
        if self._gammas[2]:
            self._nu.lerp_(neighbor_smooth[:,4],self._gammas[2])

        laplace = self._vertices - neighbor_smooth[:,:3]
        grad = torch.addcmul(self._vertices.grad, laplace, self._nu[:,None], value=self._laplacian_weight)

        if self._step>1:
            grad_lim = self._m1.abs().mul_(self._grad_lim)
            grad.clamp_(min=-grad_lim,max=grad_lim)

        lerp_unbiased(self._m1, grad, self._betas[0], self._step)
        lerp_unbiased(self._m2, (grad**2).sum(dim=-1), self._betas[1], self._step)

        velocity = self._m1 / self._m2[:,None].sqrt().add_(eps)     
        speed = velocity.norm(dim=-1)   
        if self._betas[2]:
            lerp_unbiased(self._nu,speed,self._betas[2],self._step)   
        else:
            self._nu.copy_(speed)   

        ramped_lr = self._lr * min(1,self._step * (1-self._betas[0]) / self._ramp)
        self._vertices.add_(velocity * self._ref_len[:,None], alpha=-ramped_lr)

        if self._step % self._remesh_interval == 0:
            if self._local_edgelen:
                len_change = (1 + (self._nu - self._nu_ref) * self._gain)
            else:
                len_change = (1 + (self._nu.mean() - self._nu_ref) * self._gain)
            self._ref_len *= len_change
            self._ref_len.clamp_(*self._edge_len_lims)


        grad_col = self._colors.grad
        if self._step>1:
            grad_lim_col = self._col_M.abs().mul_(self._grad_lim)
            grad_col.clamp_(min=-grad_lim_col,max=grad_lim_col)

        lerp_unbiased(self._col_M, grad_col, self._col_betas[0], self._step)
        lerp_unbiased(self._col_G, (grad_col**2).sum(dim=-1), self._col_betas[1], self._step)

        velocity_col = self._col_M / self._col_G[:,None].sqrt().add_(eps)
        ramped_lr_col = self._col_lr * min(1,self._step * (1-self._col_betas[0]) / self._ramp)
        self._colors.add_(velocity_col, alpha=-ramped_lr_col)                                           


    def remesh(self, flip:bool=True)->tuple[torch.Tensor,torch.Tensor]:
        min_edge_len = self._ref_len * self._ffts * (1 - self._edge_len_tol)
        max_edge_len = self._ref_len * self._ffts * (1 + self._edge_len_tol) 

        self._colors_gradient[:,0] = self._col_G                                                     
        self._colors_gradient[:,1:4] = self._col_M                                                      
        self._vertices_etc,self._colors,self._ffts,self._faces,self._colors_gradient = remesh(self._vertices_etc,self._colors,self._ffts,self._faces,min_edge_len,max_edge_len,flip,self._colors_gradient)
        self._split_vertices_etc()
        self._split_colors_gradient()
        self._vertices.requires_grad_()
        self._colors.requires_grad_()

        return self._vertices, self._colors, self._faces


