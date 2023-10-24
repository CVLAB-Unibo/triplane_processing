# triplane_processing
Official code for "Neural Processing of Tri-Plane Hybrid Neural Fields"

[[Paper](https://arxiv.org/abs/2310.01140)]

1) Install conda or mamba (mamba is faster)
2) Create environment with the command  ``` mamba env create -f requirements.yaml ```
3) Activate with ``` conda activate triplane ```
4) Install pycarus ``` pip install pycarus ```
5) create a sym link to a datasets folder: 
    ``` mkdir datasets ```
    ``` ln -s path_to_ModelNet40/  datasets/ModelNet40 ``` 

Fit triplanes with:
```python triplane_fit_pc.py dataset=manifold```
Also modify the cfg/triplane_pcd_fit/main.yaml accordingly (out dir folder for example).
