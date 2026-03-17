import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DropPath & Initialization
# ============================================================
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


# ============================================================
# Motion-only input D (Torch, GPU)
# Input : x [N, 3, T, V, M]
# Output: D [N, 3, T, V, M]
# D[:, :, t] = x[:, :, t+1] - x[:, :, t], last frame = 0
# ============================================================
class MotionOnlyInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        d = torch.zeros_like(x)
        d[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        return d


# ============================================================
# Temporal Modules
# ============================================================
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=(1, 2, 3, 4),
        residual=False,
        residual_kernel_size=1,
    ):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=d,
                    ),
                )
                for ks, d in zip(kernel_size, dilations)
            ]
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        last_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
        )
        if stride != 1:
            last_branch.add_module("pool", nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1)))
        self.branches.append(last_branch)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        outs = [b(x) for b in self.branches]
        return torch.cat(outs, dim=1) + res


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


# ============================================================
# STGCNEdgeWeightedUnit
# ============================================================
class STGCNEdgeWeightedUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point=25):
        super().__init__()
        k = A.shape[0]
        assert k == 3, "ST-GCN requires 3 partitions"

        self.register_buffer("A", A)  # [3, V, V]
        self.M = nn.Parameter(torch.ones_like(A))

        self.conv_parts = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(k)])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for conv in self.conv_parts:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [N, T, V, C]
        out = 0
        for k in range(self.A.shape[0]):
            a_k = self.A[k] * self.M[k]  # [V, V]
            agg = torch.einsum("vw,ntvc->ntcw", a_k, x_perm)
            agg = agg.permute(0, 2, 1, 3).contiguous()
            out = out + self.conv_parts[k](agg)
        out = self.bn(out)
        out = self.relu(out)
        return out


# ============================================================
# MHSA with Differential Attention + hop-RPE + hyperedge token
# - lambda_init uses layer index
# - attn_drop applied after softmax
# - hops registered as buffer
# ============================================================
def lambda_init_fn(layer_idx: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))


class MHSA(nn.Module):
    def __init__(
        self,
        dim_in,
        dim,
        A,
        num_heads=6,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_point=25,
        layer=0,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        h1 = A.sum(0).detach().cpu().numpy()
        h1[h1 != 0] = 1
        h = [np.eye(num_point), h1]
        hops_np = np.zeros((num_point, num_point), dtype=np.float32)

        for i in range(2, num_point):
            hi = h[i - 1] @ h1.T
            hi[hi != 0] = 1
            h.append(hi)

        for i in range(num_point - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
            hops_np += i * h[i]

        hops = torch.tensor(hops_np).long()
        self.register_buffer("hops", hops)
        self.rpe = nn.Parameter(torch.zeros((int(hops.max().item()) + 1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.outer = nn.Parameter(torch.stack([torch.eye(num_point) for _ in range(num_heads)], dim=0), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.v = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, groups=num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_init = lambda_init_fn(layer)
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim // 2) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim // 2) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim // 2) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim // 2) * 0.1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        # x: [N,C,T,V], e: [N,dim,T,V]
        n, c, t, vtx = x.shape

        v = self.v(x).reshape(n, self.num_heads, -1, t, vtx).permute(0, 3, 1, 4, 2)  # [N,T,H,V,Dh]
        k = self.k(x).reshape(n, 2, self.num_heads, -1, t, vtx).permute(1, 0, 4, 2, 5, 3)
        q = self.q(x).reshape(n, 2, self.num_heads, -1, t, vtx).permute(1, 0, 4, 2, 5, 3)
        k1, k2 = k[0], k[1]
        q1, q2 = q[0], q[1]

        e_k = e.reshape(n, self.num_heads, -1, t, vtx).permute(0, 3, 1, 4, 2)
        k_r = self.rpe[self.hops].view(vtx, vtx, self.num_heads, -1)

        b1 = torch.einsum("bthnc,nmhc->bthnm", q1, k_r)
        b2 = torch.einsum("bthnc,nmhc->bthnm", q2, k_r)
        c1 = torch.einsum("bthnc,bthmc->bthnm", q1, e_k)
        c2 = torch.einsum("bthnc,bthmc->bthnm", q2, e_k)
        d = torch.einsum("hc,bthmc->bthm", self.w1, e_k).unsqueeze(-2)

        a1 = torch.matmul(q1, k1.transpose(-2, -1))
        a2 = torch.matmul(q2, k2.transpose(-2, -1))

        attn1 = ((a1 + b1 + c1 + d) * self.scale).softmax(dim=-1)
        attn2 = ((a2 + b2 + c2 + d) * self.scale).softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(attn1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(attn2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        x1 = torch.matmul(self.alpha * attn1 + self.outer, v)
        x2 = torch.matmul(self.alpha * attn2 + self.outer, v)
        x_spatial = x1 - lambda_full * x2

        x_spatial = x_spatial.transpose(3, 4).reshape(n, t, -1, vtx).transpose(1, 2)  # [N,dim,T,V]
        x_spatial = self.proj_drop(self.proj(x_spatial))
        return x_spatial


# ============================================================
# unit_vit: Transformer + ST-GCN (Full Graph layers 1..10)
# Aligned with kode377_FullGraph:
# - hyperedge token built from raw x by joint_label pooling + pe_proj
# - MHSA consumes x_norm and e
# - GCN consumes raw x
# ============================================================
class unit_vit(nn.Module):
    def __init__(
        self,
        dim_in,
        dim,
        A,
        num_of_heads,
        add_skip_connection=True,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        prev_stride=1,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer=0,
        pe=False,
        num_point=25,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.skip_proj = nn.Conv2d(dim_in, dim, 1, bias=False) if dim_in != dim else nn.Identity()
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False) if pe else None

        self.attn = MHSA(
            dim_in,
            dim,
            A,
            num_heads=num_of_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_point=num_point,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FULL GRAPH: always enable ST-GCN branch
        self.gcn_branch = STGCNEdgeWeightedUnit(dim_in, dim, A, num_point=num_point)

    def _build_hyperedge_token(self, x, joint_label_t):
        if self.pe_proj is None:
            raise ValueError("pe_proj is None. Set pe=True to build hyperedge token.")

        num_groups = int(joint_label_t.max().item()) + 1
        label = F.one_hot(joint_label_t, num_classes=num_groups).float()  # [V,G]
        label = label / (label.sum(dim=0, keepdim=True) + 1e-6)

        z = x @ label                                                    # [N,C,T,G]
        z = self.pe_proj(z)                                              # [N,dim,T,G]
        z_g = z.permute(3, 0, 1, 2).contiguous()                          # [G,N,dim,T]
        e = z_g[joint_label_t].permute(1, 2, 3, 0).contiguous()           # [N,dim,T,V]
        return e

    def forward(self, x, joint_label_t, groups=None):
        x_skip = self.skip_proj(x)

        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        e = self._build_hyperedge_token(x, joint_label_t)

        x_transformer = self.attn(x_norm, e)
        x_gcn = self.gcn_branch(x)

        x_out = self.drop_path(x_transformer + x_gcn)
        x_out = x_skip + x_out
        return x_out


# ============================================================
# TCN_ViT_unit
# ============================================================
class TCN_ViT_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        num_of_heads=6,
        residual=True,
        kernel_size=5,
        dilations=(1, 2),
        pe=False,
        num_point=25,
        layer=0,
        prev_stride=1,
        drop_path=0.0,
    ):
        super().__init__()
        self.vit1 = unit_vit(
            in_channels,
            out_channels,
            A,
            add_skip_connection=residual,
            num_of_heads=num_of_heads,
            prev_stride=prev_stride,
            pe=pe,
            num_point=num_point,
            layer=layer,
            drop_path=drop_path,
        )
        self.tcn1 = MultiScale_TemporalConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
        )
        self.act = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: torch.zeros_like(x)
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label_t, groups=None):
        y = self.act(self.tcn1(self.vit1(x, joint_label_t, groups)) + self.residual(x))
        return y


# ============================================================
# Final Model: kode358D FullGraph (motion-only input)
# ============================================================
class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        num_of_heads=9,
        joint_label=None,
        **kwargs,
    ):
        super().__init__()
        if graph is None:
            raise ValueError("Graph class must be provided.")

        if joint_label is None or len(joint_label) == 0:
            joint_label = [0] * num_point

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32)  # [3, V, V]
        self.register_buffer("A", A)

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person

        self.register_buffer("joint_label_t", torch.tensor(joint_label, dtype=torch.long))

        # motion-only input
        self.motion_only = MotionOnlyInput()

        # BN over (M*V*C)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        dpr = [x.item() for x in torch.linspace(0, 0.2, 10)]
        width = 36 * num_of_heads

        self.l1 = TCN_ViT_unit(3, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=1, drop_path=dpr[0])
        self.l2 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=2, drop_path=dpr[1])
        self.l3 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=3, drop_path=dpr[2])
        self.l4 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=4, drop_path=dpr[3])
        self.l5 = TCN_ViT_unit(width, width, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5, drop_path=dpr[4])
        self.l6 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=6, prev_stride=2, drop_path=dpr[5])
        self.l7 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=7, prev_stride=2, drop_path=dpr[6])
        self.l8 = TCN_ViT_unit(width, width, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8, prev_stride=2, drop_path=dpr[7])
        self.l9 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=9, prev_stride=4, drop_path=dpr[8])
        self.l10 = TCN_ViT_unit(width, width, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=10, prev_stride=4, drop_path=dpr[9])

        self.fc = nn.Linear(width, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()

    def forward(self, x, y):
        """
        x: [N, 3, T, V, M] raw joints
        replaced by D(motion) computed in torch on GPU.
        """
        jl = self.joint_label_t

        # 1) motion-only input
        x = self.motion_only(x)  # [N,3,T,V,M]

        # 2) BN/reshape pipeline
        n, c, t, vtx, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * vtx * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, vtx, c, t).contiguous().view(n * m, vtx, c, t).permute(0, 2, 3, 1).contiguous()  # [N*M,C,T,V]

        # 3) backbone FullGraph
        x = self.l1(x, jl, None)
        x = self.l2(x, jl, None)
        x = self.l3(x, jl, None)
        x = self.l4(x, jl, None)
        x = self.l5(x, jl, None)
        x = self.l6(x, jl, None)
        x = self.l7(x, jl, None)
        x = self.l8(x, jl, None)
        x = self.l9(x, jl, None)
        x = self.l10(x, jl, None)

        # 4) head
        x = x.mean(dim=[-1, -2])           # [N*M, width]
        x = x.view(n, m, -1).mean(dim=1)   # [N, width]
        x = self.drop_out(x)
        x = self.fc(x)
        return x, y
