import collections
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch
from pycarus.learning.models.siren import SIREN
from pycarus.geometry.pcd import knn, sample_points_around_pcd
from sklearn.neighbors import KDTree
from torch import Tensor
from typing import Callable, Tuple
import torch.nn.functional as F

class CoordsEncoder:
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos),
    ) -> None:
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def compute_partseg_labels(
    pcd: Tensor,
    pcd_labels: Tensor,
    stds: List[float],
    num_points_per_std: List[int],
    coords_range: Tuple[float, float],
    surface_th: float,
    empty_label: int,
) -> Tuple[Tensor, Tensor]:
    coords = sample_points_around_pcd(pcd, stds, num_points_per_std, coords_range, "cuda")
    indices, _, distances = knn(coords, pcd.cuda(), 1)
    indices = indices.squeeze(-1)
    distances = distances.squeeze(-1)

    labels = torch.zeros((coords.shape[0],)).long().cuda()
    labels[distances > surface_th] = empty_label
    labels[distances <= surface_th] = pcd_labels[indices[distances <= surface_th]].cuda()

    return coords, labels

def get_mlps_batched_params(mlps: List[SIREN]) -> List[Tensor]:
    params = []
    for i in range(len(mlps)):
        params.append(list(mlps[i].parameters()))

    batched_params = []
    for i in range(len(params[0])):
        p = torch.stack([p[i] for p in params], dim=0)
        p = torch.clone(p.detach())
        p.requires_grad = True
        batched_params.append(p)

    return batched_params


def flatten_mlp_params(sd: OrderedDict[str, Tensor]) -> Tensor:
    all_params = []
    for k in sd:
        all_params.append(sd[k].view(-1))
    all_params = torch.cat(all_params, dim=-1)
    return all_params


def unflatten_mlp_params(
    params: Tensor,
    sample_sd: OrderedDict[str, Tensor],
) -> OrderedDict[str, Tensor]:
    sd = collections.OrderedDict()

    start = 0
    for k in sample_sd:
        end = start + sample_sd[k].numel()
        layer_params = params[start:end].view(sample_sd[k].shape)
        sd[k] = layer_params
        start = end

    return sd

def get_mlp_params_as_matrix(flattened_params: Tensor, sd: OrderedDict) -> Tensor:
    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]
    start = params_shapes[0].numel() + params_shapes[1].numel()
    end = params_shapes[-1].numel() + params_shapes[-2].numel()
    params = flattened_params[start:-end]
    return params.reshape((-1, feat_dim))


def mlp_batched_forward(batched_params: List[Tensor], coords: Tensor) -> Tensor:
    num_layers = len(batched_params) // 2

    f = coords

    for i in range(num_layers):
        weights = batched_params[i * 2]
        biases = batched_params[i * 2 + 1]

        f = torch.bmm(f, weights.permute(0, 2, 1)) + biases.unsqueeze(1)

        if i < num_layers - 1:
            f = torch.sin(30 * f)

    return f.squeeze(-1)


def get_recalls(gallery: Tensor, labels_gallery: Tensor, kk: List[int]) -> Dict[int, float]:
    """Computes the recall using different nearest neighbors searches.

    Args:
        gallery: the gallery containing all the embeddings for the dataset.
        kk: the number of nearest neighbors to use for each recall.

    Returns:
        The computed recalls with different nearest neighbors searches.
    """
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    tree = KDTree(gallery)

    for query, label_query in zip(gallery, targets):
        with torch.no_grad():
            query = np.expand_dims(query, 0)
            _, indices_matched = tree.query(query, k=max_nn + 1)
            indices_matched = indices_matched[0]

            for k in kk:
                indices_matched_temp = indices_matched[1 : k + 1]
                classes_matched = targets[indices_matched_temp]
                recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls

def focal_loss(pred: Tensor, gt: Tensor, alpha: float = 0.1, gamma: float = 3) -> Tensor:
    alpha_w = torch.tensor([alpha, 1 - alpha]).cuda()

    bce_loss = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction="none")
    bce_loss = bce_loss.view(-1)

    gt = gt.type(torch.long)
    at = alpha_w.gather(0, gt.view(-1))
    pt = torch.exp(-bce_loss)
    f_loss = at * ((1 - pt) ** gamma) * bce_loss

    return f_loss.mean()