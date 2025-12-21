# Texture-Guided-Remeshing

## Environment Setup
1. Install CUDA SDK:
we used CUDA TOOLKIT 12.1
2. Create conda environment:
```bash
conda env create -f environment.yaml
conda activate <env_name>
```
3. Install pytorch extensions:
```bash
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
If you meet issues when installing pytorch3d, you could try other installation method following:
[the official PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

4. Install nvdiffrast:
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Refinement
``` bash
python train.py --input ${YOUR_INITIAL_MESH_PATH}/train/ours_30000 --camera ${YOUR_RAW_DATA_PATH} --output ${OUTPUT_PATH} --normal_w 0.3 --rgb_w 3.0 --depth_w 0.3
```
YOUR_INITIAL_MESH_PATH should be organized like 
```bash
data_root
├── train/ours_30000
│   ├── gt 
│   └── renders
├── ...
├─ cfg_args
├─ fuse_post.ply
└─ fuse.ply
```

## Checklist
- [x] Release the refinement code
- [ ] Release the relighting and deformation code