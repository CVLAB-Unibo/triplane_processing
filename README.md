# triplane_processing
Official code for "Neural Processing of Tri-Plane Hybrid Neural Fields"

[[Paper](https://arxiv.org/abs/2310.01140)]

1) Install conda or mamba (mamba is faster)
2) Create environment with the command  ``` mamba env create -f requirements.yaml ```
3) Activate with ``` conda activate triplane ```
4) create a sym link to a datasets folder: 
    ``` mkdir datasets ```
    ``` ln -s path_to_ModelNet40/  datasets/ModelNet40 ``` 

5) We use Weight&Biases to track our experiments, so you need to create a free acount if you want to run this code.

Fit triplanes from point clouds with:
```python triplane_fit_pc.py dataset=manifold```
Also modify the cfg/triplane_pcd_fit/main.yaml accordingly (out dir folder and weights and biases parameters for example).

In our case, we fit the same pre augmenred data used in [[inr2vec](https://arxiv.org/abs/2310.01140)], so you need to donwload the data from the corresponding web page. Each INR contains the augmented shape used to fit the INR. We use the augmented shapes contained in these files to fit our tri-planes.

If you need pre-computed triplanes on the datasets used in the paper, please contact us. 


Train classifier on triplanes:
```python triplane_cls.py dataset=modelnet```
