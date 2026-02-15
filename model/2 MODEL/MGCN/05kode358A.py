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
# A-only input (IDENTICAL to kode377 A-only)
# Input : x [N, 3, T, V, M]
# Output: A [N, 3, T, V, M]
#
# Definition (temporal Laplacian / 2nd diff):
#   For t=1..T-2:
#     A_t = (x_{t-1} - x_t) + (x_{t+1} - x_t)
#         = x_{t-1} + x_{t+1} - 2*x_t
#   A_0 = 0, A_{T-1} = 0
# ============================================================
class AccelerationOnlyInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        A = torch.zeros_like(x)
        if x.size(2) >= 3:
            v1 = x[:, :, 0:-2, :, :] - x[:, :, 1:-1, :, :]
            v2 = x[:, :, 2:,   :, :] - x[:, :, 1:-1, :, :]
            A[:, :, 1:-1, :, :] = v1 + v2
        return A


# ============================================================
# Temporal Modules (same style as kode358)
# ============================================================
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
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
                        branch_channels, branch_channels,
                        kernel_size=ks, stride=stride, dilation=d
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
            last_branch.add_module(
                "pool", nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
            )
        self.branches.append(last_branch)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels, out_channels,
                kernel_size=residual_kernel_size, stride=stride
            )

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
            in_channels, out_channels,
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
# STGCNEdgeWeightedUnit (kode358)
# ============================================================
class STGCNEdgeWeightedUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_point=25):
        super().__init__()
        K = A.shape[0]
        assert K == 3, "ST-GCN requires 3 partitions"
        self.register_buffer("A", A)
        self.M = nn.Parameter(torch.ones_like(A))

        self.conv_parts = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(K)]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for conv in self.conv_parts:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [N,T,V,C]
        out = 0
        for k in range(self.A.shape[0]):
            A_k = self.A[k] * self.M[k]
            agg = torch.einsum("vw,ntvc->ntcw", A_k, x_perm)
            agg = agg.permute(0, 2, 1, 3).contiguous()
            out = out + self.conv_parts[k](agg)
        out = self.bn(out)
        out = self.relu(out)
        return out


# ============================================================
# MHSA with Differential Attention + hop-RPE (kode358)
# ============================================================
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


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
        hops = np.zeros((num_point, num_point), dtype=np.float32)
        for i in range(2, num_point):
            hi = h[i - 1] @ h1.T
            hi[hi != 0] = 1
            h.append(hi)
        for i in range(num_point - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
            hops += i * h[i]

        self.register_buffer("hops", torch.tensor(hops).long())
        self.rpe = nn.Parameter(torch.zeros((int(hops.max()) + 1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.outer = nn.Parameter(
            torch.stack([torch.eye(num_point) for _ in range(num_heads)], dim=0),
            requires_grad=True,
        )
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
        N, C, T, V = x.shape

        v = self.v(x).reshape(N, self.num_heads, -1, T, V).permute(0, 3, 1, 4, 2)
        k = self.k(x).reshape(N, 2, self.num_heads, -1, T, V).permute(1, 0, 4, 2, 5, 3)
        q = self.q(x).reshape(N, 2, self.num_heads, -1, T, V).permute(1, 0, 4, 2, 5, 3)
        k1, k2 = k[0], k[1]
        q1, q2 = q[0], q[1]

        e_k = e.reshape(N, self.num_heads, -1, T, V).permute(0, 3, 1, 4, 2)
        k_r = self.rpe[self.hops].view(V, V, self.num_heads, -1)

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

        x_spatial = x_spatial.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x_spatial = self.proj_drop(self.proj(x_spatial))
        return x_spatial


# ============================================================
# unit_vit (kode358)
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
            dim_in, dim, A,
            num_heads=num_of_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_point=num_point,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_gcn_branch = layer <= 4
        self.gcn_branch = STGCNEdgeWeightedUnit(dim_in, dim, A, num_point=num_point) if self.use_gcn_branch else None

    def forward(self, x, joint_label_t, groups):
        label = F.one_hot(joint_label_t).float()
        denom = label.sum(dim=0, keepdim=True).clamp_min(1.0)
        z = x @ (label / denom)

        if self.pe_proj is not None:
            z = self.pe_proj(z).permute(3, 0, 1, 2)
            e = z[joint_label_t].permute(1, 2, 3, 0)
        else:
            e = z.permute(3, 0, 1, 2)

        x_skip = self.skip_proj(x)
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_transformer = self.attn(x_norm, e)

        x_gcn = self.gcn_branch(x) if self.use_gcn_branch else 0.0
        x_fused = x_transformer + x_gcn
        x_out = self.drop_path(x_fused)
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
            in_channels, out_channels, A,
            add_skip_connection=residual,
            num_of_heads=num_of_heads,
            prev_stride=prev_stride,
            pe=pe,
            num_point=num_point,
            layer=layer,
            drop_path=drop_path,
        )
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels,
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

    def forward(self, x, joint_label_t, groups):
        y = self.act(self.tcn1(self.vit1(x, joint_label_t, groups)) + self.residual(x))
        return y


# ============================================================
# Final Model: kode358 backbone, input replaced by A (same as kode377 A-only)
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
        if joint_label is None:
            joint_label = []

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer("A", A)

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person

        if len(joint_label) > 0:
            self.register_buffer("joint_label_t", torch.tensor(joint_label, dtype=torch.long))
        else:
            self.register_buffer("joint_label_t", torch.zeros(num_point, dtype=torch.long))

        # A-only input (IDENTICAL to kode377 A-only)
        self.acc_only = AccelerationOnlyInput()

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        dpr = [x.item() for x in torch.linspace(0, 0.2, 10)]
        base_dim = 36 * num_of_heads

        self.l1 = TCN_ViT_unit(3, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=1, drop_path=dpr[0])
        self.l2 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=2, drop_path=dpr[1])
        self.l3 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=3, drop_path=dpr[2])
        self.l4 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=4, drop_path=dpr[3])
        self.l5 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=5, drop_path=dpr[4])
        self.l6 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=6, prev_stride=2, drop_path=dpr[5])
        self.l7 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=7, prev_stride=2, drop_path=dpr[6])
        self.l8 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=8, prev_stride=2, drop_path=dpr[7])
        self.l9 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=9, prev_stride=4, drop_path=dpr[8])
        self.l10 = TCN_ViT_unit(base_dim, base_dim, A, residual=True, num_of_heads=num_of_heads, pe=True,
                                num_point=num_point, layer=10, prev_stride=4, drop_path=dpr[9])

        self.fc = nn.Linear(base_dim, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()

    def forward(self, x, y):
        jl = self.joint_label_t
        num_groups = int(jl.max().item()) + 1 if jl.numel() > 0 else 1
        groups = [[int(i) for i in (jl == g).nonzero(as_tuple=False).view(-1).tolist()] for g in range(num_groups)]

        # 1) A-only input (IDENTICAL to kode377 A-only)
        x = self.acc_only(x)  # [N,3,T,V,M]

        # 2) BN/reshape pipeline (kode358)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # 3) backbone kode358
        x = self.l1(x, jl, groups)
        x = self.l2(x, jl, groups)
        x = self.l3(x, jl, groups)
        x = self.l4(x, jl, groups)
        x = self.l5(x, jl, groups)
        x = self.l6(x, jl, groups)
        x = self.l7(x, jl, groups)
        x = self.l8(x, jl, groups)
        x = self.l9(x, jl, groups)
        x = self.l10(x, jl, groups)

        # 4) head
        x = x.mean(dim=[-1, -2])
        x = x.view(N, M, -1).mean(dim=1)
        x = self.drop_out(x)
        x = self.fc(x)
        return x, y
