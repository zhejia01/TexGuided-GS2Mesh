import torch
import torch.nn.functional as tfunc
import torch_scatter

def prepend_dummies(
        vertices:torch.Tensor,     
        colors:torch.Tensor,     
        ffts:torch.Tensor,    
        faces:torch.Tensor,          
        colors_gradient:torch.Tensor
    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V,D = vertices.shape
    D_col = colors_gradient.shape[1]
    vertices = torch.concat((torch.full((1,D),fill_value=torch.nan,device=vertices.device),vertices),dim=0)

    colors = torch.concat((torch.full((1,3),fill_value=torch.nan,device=vertices.device),colors),dim=0)
    ffts = torch.concat((torch.full((1,),fill_value=torch.nan,device=vertices.device),ffts),dim=0)
    colors_gradient = torch.concat((torch.zeros((1,D_col),dtype=torch.long,device=colors.device),colors_gradient),dim=0) 

    faces = torch.concat((torch.zeros((1,3),dtype=torch.long,device=faces.device),faces+1),dim=0)
    return vertices,colors,ffts,faces,colors_gradient

def remove_dummies(
        vertices:torch.Tensor,                                             
        colors:torch.Tensor,                             
        ffts:torch.Tensor,                         
        faces:torch.Tensor,                                 
        colors_gradient:torch.Tensor
    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """remove dummy elements added with prepend_dummies()"""
    return vertices[1:],colors[1:],ffts[1:],faces[1:]-1,colors_gradient[1:]


def calc_edges(
        faces: torch.Tensor,                                                     
        with_edge_to_face: bool = False
    ) -> tuple[torch.Tensor, ...]:
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |      
    e0---->-o     
    """

    F = faces.shape[0]
    face_edges = torch.stack((faces,faces.roll(-1,1)),dim=-1)                                         
    full_edges = face_edges.reshape(F*3,2)
    sorted_edges,_ = full_edges.sort(dim=-1)                            

    edges,full_to_unique = torch.unique(input=sorted_edges,sorted=True,return_inverse=True,dim=0)             
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F,3)                                                                                 

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:,0]!=sorted_edges[:,0]        
    edge_to_face = torch.zeros((E,2,2),dtype=torch.long,device=faces.device)            
    scatter_src = torch.cartesian_prod(torch.arange(0,F,device=faces.device),torch.arange(0,3,device=faces.device))       
    edge_to_face.reshape(2*E,2).scatter_(dim=0,index=(2*full_to_unique+is_right)[:,None].expand(F*3,2),src=scatter_src)            
    edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face

def calc_edge_length(
        vertices:torch.Tensor,                        
        edges:torch.Tensor,                                                      
        )->torch.Tensor:   

    full_vertices = vertices[edges]       
    a,b = full_vertices.unbind(dim=1)     
    return torch.norm(a-b,p=2,dim=-1)

def calc_face_normals(
        vertices:torch.Tensor,                                      
        faces:torch.Tensor,                                      
        normalize:bool=False,
        )->torch.Tensor:     
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces]         
    v0,v1,v2 = full_vertices.unbind(dim=1)     
    face_normals = torch.cross(v1-v0,v2-v0, dim=1)     
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1)               
    return face_normals     

def calc_vertex_normals(
        vertices:torch.Tensor,                                      
        faces:torch.Tensor,                                      
        face_normals:torch.Tensor=None,                     
        )->torch.Tensor:     

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device)         
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1)     
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)

def calc_face_ref_normals(
        faces:torch.Tensor,                        
        vertex_normals:torch.Tensor,                  
        normalize:bool=False,
        )->torch.Tensor:     
    """calculate reference normals for face flip detection"""
    full_normals = vertex_normals[faces]         
    ref_normals = full_normals.sum(dim=1)     
    if normalize:
        ref_normals = tfunc.normalize(ref_normals, eps=1e-6, dim=1)
    return ref_normals

def pack(
        vertices:torch.Tensor,                          
        colors:torch.Tensor,     
        ffts:torch.Tensor,    
        faces:torch.Tensor,                        
        colors_gradient:torch.Tensor,
        )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:                                             
    """removes unused elements in vertices and faces"""
    V = vertices.shape[0]
    used_faces = faces[:,0]!=0
    used_faces[0] = True
    faces = faces[used_faces]      

    used_vertices = torch.zeros(V,3,dtype=torch.bool,device=vertices.device)
    used_vertices.scatter_(dim=0,index=faces,value=True,reduce='add')                  
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices]      
    colors = colors[used_vertices]      
    ffts = ffts[used_vertices]      
    colors_gradient = colors_gradient[used_vertices]      

    ind = torch.zeros(V,dtype=torch.long,device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] =  torch.arange(0,V1,device=vertices.device)      
    faces = ind[faces]

    return vertices,colors,ffts,faces,colors_gradient

def split_edges(
        vertices:torch.Tensor,                  
        colors:torch.Tensor,     
        ffts:torch.Tensor,    
        faces:torch.Tensor,                        
        edges:torch.Tensor,                                                 
        face_to_edge:torch.Tensor,                       
        splits,        
        colors_gradient:torch.Tensor,
        pack_faces:bool=True,
        )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:                  


    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item()      

    if S==0:
        return vertices,colors,ffts,faces,colors_gradient
    edge_vert = torch.zeros_like(splits, dtype=torch.long)   
    edge_vert[splits] = torch.arange(V,V+S,dtype=torch.long,device=vertices.device)                        
    side_vert = edge_vert[face_to_edge]                          
    split_edges = edges[splits]        

    split_vertices = vertices[split_edges].mean(dim=1)     
    split_colors = colors[split_edges].mean(dim=1)     
    split_ffts = ffts[split_edges].mean(dim=1)    

    split_colors_gradient = colors_gradient[split_edges].mean(dim=1)     
    vertices = torch.concat((vertices,split_vertices),dim=0)
    colors = torch.concat((colors,split_colors),dim=0)
    ffts = torch.concat((ffts,split_ffts),dim=0)
    colors_gradient = torch.concat((colors_gradient,split_colors_gradient),dim=0)

    side_split = side_vert!=0     
    shrunk_faces = torch.where(side_split,side_vert,faces)                          
    new_faces = side_split[:,:,None] * torch.stack((faces,side_vert,shrunk_faces.roll(1,dims=-1)),dim=-1)           
    faces = torch.concat((shrunk_faces,new_faces.reshape(F*3,3)))      
    if pack_faces:
        mask = faces[:,0]!=0
        mask[0] = True
        faces = faces[mask]           

    return vertices,colors,ffts,faces,colors_gradient

def collapse_edges(
        vertices:torch.Tensor,                  
        colors:torch.Tensor,     
        ffts:torch.Tensor,    
        faces:torch.Tensor,                       
        edges:torch.Tensor,                                                 
        priorities:torch.Tensor,         
        colors_gradient:torch.Tensor,
        stable:bool=False,                       
        )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:                  
    V = vertices.shape[0]
    _,order = priorities.sort(stable=stable)   
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    vert_rank = torch.zeros(V,dtype=torch.long,device=vertices.device)   
    edge_rank = rank   
    for i in range(3):
        torch_scatter.scatter_max(src=edge_rank[:,None].expand(-1,2).reshape(-1),index=edges.reshape(-1),dim=0,out=vert_rank)
        edge_rank,_ = vert_rank[edges].max(dim=-1)   
    candidates = edges[(edge_rank==rank).logical_and_(priorities>0)]      

    vert_connections = torch.zeros(V,dtype=torch.long,device=vertices.device)   
    vert_connections[candidates[:,0]] = 1       
    edge_connections = vert_connections[edges].sum(dim=-1)                            
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1))                     
    vert_connections[candidates] = 0                     
    edge_connections = vert_connections[edges].sum(dim=-1)                                
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1))                             
    collapses = candidates[vert_connections[candidates[:,1]] <= 2]                                                           

    vertices[collapses[:,0]] = vertices[collapses].mean(dim=1)           
    colors[collapses[:,0]] = colors[collapses].mean(dim=1)           
    ffts[collapses[:,0]] = ffts[collapses].mean(dim=1)           

    colors_gradient[:,1:4][collapses[:,0]] = colors_gradient[:,1:4][collapses].mean(dim=1)           
    colors_gradient[:,0][collapses[:,0]] = colors_gradient[:,0][collapses][:,0]*0.25 + colors_gradient[:,0][collapses][:,1]*0.25 + torch.sqrt(colors_gradient[:,0][collapses][:,0]*colors_gradient[:,0][collapses][:,1])*0.5
    dest = torch.arange(0,V,dtype=torch.long,device=vertices.device)   
    dest[collapses[:,1]] = dest[collapses[:,0]]
    faces = dest[faces]                    
    c0,c1,c2 = faces.unbind(dim=-1)
    collapsed = (c0==c1).logical_or_(c1==c2).logical_or_(c0==c2)
    faces[collapsed] = 0

    return vertices,colors,ffts,faces,colors_gradient

def calc_face_collapses(
        vertices:torch.Tensor,                  
        faces:torch.Tensor,                        
        edges:torch.Tensor,                                                 
        face_to_edge:torch.Tensor,                       
        edge_length:torch.Tensor,   
        face_normals:torch.Tensor,     
        vertex_normals:torch.Tensor,                  
        min_edge_length:torch.Tensor=None,   
        area_ratio = 0.5,                                                    
        shortest_probability = 0.8
        )->torch.Tensor:                     
    E = edges.shape[0]
    F = faces.shape[0]

    ref_normals = calc_face_ref_normals(faces,vertex_normals,normalize=False)     
    face_collapses = (face_normals*ref_normals).sum(dim=-1)<0   
    if min_edge_length is not None:
        min_face_length = min_edge_length[faces].mean(dim=-1)   
        min_area = min_face_length**2 * area_ratio   
        face_collapses.logical_or_(face_normals.norm(dim=-1) < min_area*2)   
        face_collapses[0] = False

    face_length = edge_length[face_to_edge]     
    if shortest_probability<1:
        randlim = round(2/(1-shortest_probability))
        rand_ind = torch.randint(0,randlim,size=(F,),device=faces.device).clamp_max_(2)                                   
        sort_ind = torch.argsort(face_length,dim=-1,descending=True)     
        local_ind = sort_ind.gather(dim=-1,index=rand_ind[:,None])
    else:
        local_ind = torch.argmin(face_length,dim=-1)[:,None]                                             
    edge_ind = face_to_edge.gather(dim=1,index=local_ind)[:,0]                                    
    edge_collapses = torch.zeros(E,dtype=torch.long,device=vertices.device)
    edge_collapses.scatter_add_(dim=0,index=edge_ind,src=face_collapses.long())                      
    return edge_collapses.bool()

def flip_edges(
        vertices:torch.Tensor,                  
        faces:torch.Tensor,                                         
        edges:torch.Tensor,                                                                   
        edge_to_face:torch.Tensor,                            
        with_border:bool=True,                                          
        with_normal_check:bool=True,                         
        stable:bool=False,                       
        ):
    V = vertices.shape[0]
    E = edges.shape[0]
    device=vertices.device
    vertex_degree = torch.zeros(V,dtype=torch.long,device=device)        
    vertex_degree.scatter_(dim=0,index=edges.reshape(E*2),value=1,reduce='add')
    neighbor_corner = (edge_to_face[:,:,1] + 2) % 3                        
    neighbors = faces[edge_to_face[:,:,0],neighbor_corner]        
    edge_is_inside = neighbors.all(dim=-1)   

    if with_border:
        vertex_is_inside = torch.ones(V,2,dtype=torch.float32,device=vertices.device)           
        src = edge_is_inside.type(torch.float32)[:,None].expand(E,2)           
        vertex_is_inside.scatter_(dim=0,index=edges,src=src,reduce='multiply')
        vertex_is_inside = vertex_is_inside.prod(dim=-1,dtype=torch.long)        
        vertex_degree -= 2 * vertex_is_inside        

    neighbor_degrees = vertex_degree[neighbors]        
    edge_degrees = vertex_degree[edges]     
    loss_change = 2 + neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1)   
    candidates = torch.logical_and(loss_change<0, edge_is_inside)   
    loss_change = loss_change[candidates]    
    if loss_change.shape[0]==0:
        return

    edges_neighbors = torch.concat((edges[candidates],neighbors[candidates]),dim=-1)      
    _,order = loss_change.sort(descending=True, stable=stable)    
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    vertex_rank = torch.zeros((V,4),dtype=torch.long,device=device)     
    torch_scatter.scatter_max(src=rank[:,None].expand(-1,4),index=edges_neighbors,dim=0,out=vertex_rank)
    vertex_rank,_ = vertex_rank.max(dim=-1)   
    neighborhood_rank,_ = vertex_rank[edges_neighbors].max(dim=-1)    
    flip = rank==neighborhood_rank    

    if with_normal_check:
        v = vertices[edges_neighbors]        
        v = v - v[:,0:1]                      
        e1 = v[:,1]
        cl = v[:,2]
        cr = v[:,3]
        n = torch.cross(e1,cl) + torch.cross(cr,e1)                            
        flip.logical_and_(torch.sum(n*torch.cross(cr,cl),dim=-1)>0)                
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1,cr-e1),dim=-1)>0)                 

    flip_edges_neighbors = edges_neighbors[flip]      
    flip_edge_to_face = edge_to_face[candidates,:,0][flip]      
    flip_faces = flip_edges_neighbors[:,[[0,3,2],[1,2,3]]]        
    faces.scatter_(dim=0,index=flip_edge_to_face.reshape(-1,1).expand(-1,3),src=flip_faces.reshape(-1,3))