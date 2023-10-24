# triplane_processing
Official code for "Neural Processing of Tri-Plane Hybrid Neural Fields"

[[Paper](https://arxiv.org/abs/2310.01140)]

1) Install conda or mamba (mamba is faster)
2) Create environment with the command  ``` mamba env create -f requirements.yaml ```
3) Activate with ``` conda activate triplane ```
4) Install pycarus ``` pip install pycarus ```

Then, try to import `pycarus` to get the command that you can run to install all the needed Pytorch libraries:
```
$ python3
>>> import pycarus
...
ModuleNotFoundError: PyTorch is not installed. Install it by running: source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
In this example, you can install all the needed Pytorch libraries by running:
```
$ source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
This script downloads and installs the wheels for torch, torchvision, pytorch3d and torch-geometric.
Occasionally, it may fails due to pytorch3d wheel not being available anymore. 

5) create a sym link to a datasets folder: 
    ``` mkdir datasets ```
    ``` ln -s path_to_ModelNet40/  datasets/ModelNet40 ``` 

Fit triplanes with:
```python triplane_fit_pc.py dataset=manifold```
Also modify the cfg/triplane_pcd_fit/main.yaml accordingly (out dir folder for example).
