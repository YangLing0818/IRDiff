import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from models.attn import CrossAttention, RetAugmentationLinearAttention


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        self.cond_dim = config.cond_dim
        self.topk_prompt = config.topk_prompt
        self.emb_mlp_aug = nn.Linear(emb_dim + self.cond_dim, self.cond_dim)
        self.emb_mlp = nn.Linear(emb_dim + self.cond_dim * 2, emb_dim)
        self.prompt_protein_mlp = nn.Linear(self.cond_dim * (self.topk_prompt + 1), self.cond_dim)
        self.prompt_ligand_attn = RetAugmentationLinearAttention(in_dim=self.cond_dim, d=self.cond_dim, context_dim=self.cond_dim)

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand, time_step=None, return_all=False, fix_x=False,
                hbap_protein=None, hbap_ligand=None,
                prompt_hbap_protein=None, prompt_hbap_ligand=None, prompt_batch_ligand=None,
                prompt_hbap_protein_2=None, prompt_hbap_ligand_2=None, prompt_batch_ligand_2=None,
                prompt_hbap_protein_3=None, prompt_hbap_ligand_3=None, prompt_batch_ligand_3=None):

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if hbap_protein is None:
            hbap_protein = torch.zeros([h_protein.shape[0], self.cond_dim]).to(h_protein.device)
        if hbap_ligand is None:
            hbap_ligand = torch.zeros([init_ligand_h.shape[0], self.cond_dim]).to(init_ligand_h.device)

        hbap_protein_aug = self.emb_mlp_aug(torch.cat([h_protein, hbap_protein.detach()], dim=1))
        hbap_ligand_aug = self.emb_mlp_aug(torch.cat([init_ligand_h, hbap_ligand.detach()], dim=1))

        protein_prompt_list = [hbap_protein_aug]
        if prompt_hbap_protein is not None:
            protein_prompt_list.append(prompt_hbap_protein)
        if prompt_hbap_protein_2 is not None:
            protein_prompt_list.append(prompt_hbap_protein_2)
        if prompt_hbap_protein_3 is not None:
            protein_prompt_list.append(prompt_hbap_protein_3)

        hbap_protein_aug = self.prompt_protein_mlp(torch.cat(protein_prompt_list, dim=1))

        ## Todo: to be optimized
        max_lig_len = 128
        hbap_ligand_aug_batch = torch.zeros([batch_size, max_lig_len, hbap_ligand_aug.shape[1]]).to(hbap_ligand_aug.device)

        prompt_hbap_ligand_batch, prompt_hbap_ligand_2_batch, prompt_hbap_ligand_3_batch = None, None, None
        if prompt_hbap_ligand is not None:
            prompt_hbap_ligand_batch = torch.zeros([batch_size, max_lig_len, prompt_hbap_ligand.shape[1]]).to(hbap_ligand_aug.device)
        if prompt_hbap_ligand_2 is not None:
            prompt_hbap_ligand_2_batch = torch.zeros([batch_size, max_lig_len, prompt_hbap_ligand_2.shape[1]]).to(hbap_ligand_aug.device)
        if prompt_hbap_ligand_3 is not None:
            prompt_hbap_ligand_3_batch = torch.zeros([batch_size, max_lig_len, prompt_hbap_ligand_3.shape[1]]).to(hbap_ligand_aug.device)

        valid_num_atom_list = []
        for bi in range(batch_size):
            hbap_ligand_aug_batch_i = hbap_ligand_aug[batch_ligand == bi]
            num_atom = hbap_ligand_aug_batch_i.shape[0]
            assert num_atom <= max_lig_len

            valid_num_atom_list.append(num_atom)
            hbap_ligand_aug_batch[bi, :num_atom] = hbap_ligand_aug_batch_i

            if prompt_hbap_ligand is not None:
                prompt_hbap_ligand_batch_i = prompt_hbap_ligand[prompt_batch_ligand == bi]
                num_prompt_atom = prompt_hbap_ligand_batch_i.shape[0]
                assert num_prompt_atom <= max_lig_len
                prompt_hbap_ligand_batch[bi, :num_prompt_atom] = prompt_hbap_ligand_batch_i

            if prompt_hbap_ligand_2 is not None:
                prompt_hbap_ligand_2_batch_i = prompt_hbap_ligand_2[prompt_batch_ligand_2 == bi]
                num_prompt_atom_2 = prompt_hbap_ligand_2_batch_i.shape[0]
                assert num_prompt_atom_2 <= max_lig_len
                prompt_hbap_ligand_2_batch[bi, :num_prompt_atom_2] = prompt_hbap_ligand_2_batch_i

            if prompt_hbap_ligand_3 is not None:
                prompt_hbap_ligand_3_batch_i = prompt_hbap_ligand_3[prompt_batch_ligand_3 == bi]
                num_prompt_atom_3 = prompt_hbap_ligand_3_batch_i.shape[0]
                assert num_prompt_atom_3 <= max_lig_len
                prompt_hbap_ligand_3_batch[bi, :num_prompt_atom_3] = prompt_hbap_ligand_3_batch_i

        prompt_hbap_ligand_batch_all_list = []
        if prompt_hbap_ligand_batch is not None:
            prompt_hbap_ligand_batch_all_list.append(prompt_hbap_ligand_batch)
        if prompt_hbap_ligand_2_batch is not None:
            prompt_hbap_ligand_batch_all_list.append(prompt_hbap_ligand_2_batch)
        if prompt_hbap_ligand_3_batch is not None:
            prompt_hbap_ligand_batch_all_list.append(prompt_hbap_ligand_3_batch)

        prompt_hbap_ligand_batch_all = torch.cat(prompt_hbap_ligand_batch_all_list, dim=1)

        hbap_ligand_aug_batch = self.prompt_ligand_attn(h=hbap_ligand_aug_batch, h_retrieved=prompt_hbap_ligand_batch_all)

        hbap_ligand_aug_list = []
        for bi in range(batch_size):
            hbap_ligand_aug_i = hbap_ligand_aug_batch[bi][:valid_num_atom_list[bi]]
            hbap_ligand_aug_list.append(hbap_ligand_aug_i)
        hbap_ligand_aug = torch.cat(hbap_ligand_aug_list, dim=0).to(hbap_ligand_aug.device)


        h_protein = self.emb_mlp(torch.cat([h_protein, hbap_protein, hbap_protein_aug], dim=1))
        init_ligand_h = self.emb_mlp(torch.cat([init_ligand_h, hbap_ligand, hbap_ligand_aug], dim=1))

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand, _ =compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, net_cond, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
            prompt_ligand_pos=None, prompt_ligand_v=None, prompt_batch_ligand=None,
            prompt_ligand_pos_2=None, prompt_ligand_v_2=None, prompt_batch_ligand_2=None,
            prompt_ligand_pos_3=None, prompt_ligand_v_3=None, prompt_batch_ligand_3=None,
            time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        if (prompt_ligand_pos is not None) and (prompt_ligand_v is not None) and (prompt_batch_ligand is not None):
            prompt_protein_pos, prompt_ligand_pos, _ = center_pos(
                protein_pos, prompt_ligand_pos, batch_protein, prompt_batch_ligand, mode=self.center_pos_mode)

        if (prompt_ligand_pos_2 is not None) and (prompt_ligand_v_2 is not None) and (prompt_batch_ligand_2 is not None):
            prompt_protein_pos_2, prompt_ligand_pos_2, _ = center_pos(
                protein_pos, prompt_ligand_pos_2, batch_protein, prompt_batch_ligand_2, mode=self.center_pos_mode)

        if (prompt_ligand_pos_3 is not None) and (prompt_ligand_v_3 is not None) and (prompt_batch_ligand_3 is not None):
            prompt_protein_pos_3, prompt_ligand_pos_3, _ = center_pos(
                protein_pos, prompt_ligand_pos_3, batch_protein, prompt_batch_ligand_3, mode=self.center_pos_mode)

        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        ligand_hbap_flag = (time_step[batch_ligand] == self.num_timesteps)
        protein_hbap_flag = time_step[batch_protein] == self.num_timesteps
        ligand_hbap_mask = torch.ones_like(batch_ligand).float()
        ligand_hbap_mask[ligand_hbap_flag] = 0.
        ligand_hbap_mask = ligand_hbap_mask.unsqueeze(-1)
        protein_hbap_mask = torch.ones_like(batch_protein).float()
        protein_hbap_mask[protein_hbap_flag] = 0.
        protein_hbap_mask = protein_hbap_mask.unsqueeze(-1)

        hbap_ligand = None
        hbap_protein = None

        gt_protein_v = protein_v
        gt_protein_pos = protein_pos
        gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)

        if self.model_mean_type == 'noise':
            pass
        elif self.model_mean_type == 'C0':
            gt_ligand_v = ligand_v
            gt_ligand_pos = ligand_pos
            gt_lig_a_h = gt_ligand_v

            hbap_ligand, hbap_protein = net_cond.extract_features(gt_ligand_pos, gt_protein_pos, gt_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
            hbap_ligand = hbap_ligand * ligand_hbap_mask
            hbap_protein = hbap_protein * protein_hbap_mask
        else:
            raise ValueError

        prompt_hbap_ligand = None
        prompt_hbap_protein = None

        prompt_hbap_ligand_2 = None
        prompt_hbap_protein_2 = None

        prompt_hbap_ligand_3 = None
        prompt_hbap_protein_3 = None

        if (prompt_ligand_pos is not None) and (prompt_ligand_v is not None) and (prompt_batch_ligand is not None):
            prompt_lig_a_h = prompt_ligand_v
            prompt_hbap_ligand, prompt_hbap_protein = net_cond.extract_features(prompt_ligand_pos, gt_protein_pos, prompt_lig_a_h, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand, batch_protein)

        if (prompt_ligand_pos_2 is not None) and (prompt_ligand_v_2 is not None) and (prompt_batch_ligand_2 is not None):
            prompt_lig_a_h_2 = prompt_ligand_v_2
            prompt_hbap_ligand_2, prompt_hbap_protein_2 = net_cond.extract_features(prompt_ligand_pos_2, gt_protein_pos, prompt_lig_a_h_2, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand_2, batch_protein)

        if (prompt_ligand_pos_3 is not None) and (prompt_ligand_v_3 is not None) and (prompt_batch_ligand_3 is not None):
            prompt_lig_a_h_3 = prompt_ligand_v_3
            prompt_hbap_ligand_3, prompt_hbap_protein_3 = net_cond.extract_features(prompt_ligand_pos_3, gt_protein_pos, prompt_lig_a_h_3, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand_3, batch_protein)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step,

            hbap_protein=hbap_protein,
            hbap_ligand=hbap_ligand,

            prompt_hbap_protein=prompt_hbap_protein,
            prompt_hbap_ligand=prompt_hbap_ligand,
            prompt_batch_ligand=prompt_batch_ligand,

            prompt_hbap_protein_2=prompt_hbap_protein_2,
            prompt_hbap_ligand_2=prompt_hbap_ligand_2,
            prompt_batch_ligand_2=prompt_batch_ligand_2,

            prompt_hbap_protein_3=prompt_hbap_protein_3,
            prompt_hbap_ligand_3=prompt_hbap_ligand_3,
            prompt_batch_ligand_3=prompt_batch_ligand_3
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         prompt_ligand_pos=None, prompt_ligand_v=None, prompt_batch_ligand=None,
                         prompt_ligand_pos_2=None, prompt_ligand_v_2=None, prompt_batch_ligand_2=None,
                         prompt_ligand_pos_3=None, prompt_ligand_v_3=None, prompt_batch_ligand_3=None,
                         num_steps=None, center_pos_mode=None, pos_only=False, net_cond=None, cond_dim=128):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        gt_protein_v = protein_v.detach()
        gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)

        prompt_hbap_ligand, prompt_hbap_protein = None, None
        prompt_hbap_ligand_2, prompt_hbap_protein_2 = None, None
        prompt_hbap_ligand_3, prompt_hbap_protein_3 = None, None

        prompt_hbap_ligand_, prompt_hbap_protein_ = None, None
        prompt_hbap_ligand_2_, prompt_hbap_protein_2_ = None, None
        prompt_hbap_ligand_3_, prompt_hbap_protein_3_ = None, None

        if (prompt_ligand_pos is not None) and (prompt_ligand_v is not None) and (prompt_batch_ligand is not None):
            prompt_lig_a_h = prompt_ligand_v
            prompt_hbap_ligand, prompt_hbap_protein = net_cond.extract_features(prompt_ligand_pos, protein_pos, prompt_lig_a_h, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand, batch_protein)
            prompt_hbap_ligand_, prompt_hbap_protein_ = torch.zeros_like(prompt_hbap_ligand).to(prompt_hbap_ligand.device).detach(), torch.zeros_like(prompt_hbap_protein).to(prompt_hbap_ligand.device).detach()

        if (prompt_ligand_pos_2 is not None) and (prompt_ligand_v_2 is not None) and (prompt_batch_ligand_2 is not None):
            prompt_lig_a_h_2 = prompt_ligand_v_2
            prompt_hbap_ligand_2, prompt_hbap_protein_2 = net_cond.extract_features(prompt_ligand_pos_2, protein_pos, prompt_lig_a_h_2, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand_2, batch_protein)
            prompt_hbap_ligand_2_, prompt_hbap_protein_2_ = torch.zeros_like(prompt_hbap_ligand_2).to(prompt_hbap_ligand_2.device).detach(), torch.zeros_like(prompt_hbap_protein_2).to(prompt_hbap_ligand_2.device).detach()

        if (prompt_ligand_pos_3 is not None) and (prompt_ligand_v_3 is not None) and (prompt_batch_ligand_3 is not None):
            prompt_lig_a_h_3 = prompt_ligand_v_3
            prompt_hbap_ligand_3, prompt_hbap_protein_3 = net_cond.extract_features(prompt_ligand_pos_3, protein_pos, prompt_lig_a_h_3, gt_protein_a_h, gt_protein_r_h, prompt_batch_ligand_3, batch_protein)
            prompt_hbap_ligand_3_, prompt_hbap_protein_3_ = torch.zeros_like(prompt_hbap_ligand_3).to(prompt_hbap_ligand_3.device).detach(), torch.zeros_like(prompt_hbap_protein_3).to(prompt_hbap_ligand_3.device).detach()

        hbap_protein = None
        hbap_ligand = None

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):

            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t,

                hbap_protein=hbap_protein,
                hbap_ligand=hbap_ligand,

                prompt_hbap_protein=prompt_hbap_protein_,
                prompt_hbap_ligand=prompt_hbap_ligand_,
                prompt_batch_ligand=prompt_batch_ligand,

                prompt_hbap_protein_2=prompt_hbap_protein_2_,
                prompt_hbap_ligand_2=prompt_hbap_ligand_2_,
                prompt_batch_ligand_2=prompt_batch_ligand_2,

                prompt_hbap_protein_3=prompt_hbap_protein_3_,
                prompt_hbap_ligand_3=prompt_hbap_ligand_3_,
                prompt_batch_ligand_3=prompt_batch_ligand_3,
            )
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            pred_ligand_pos = pos0_from_e.detach()
            pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)

            hbap_ligand, hbap_protein = net_cond.extract_features(pred_ligand_pos, protein_pos, pred_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
            hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()

            prompt_hbap_ligand_, prompt_hbap_protein_ = prompt_hbap_ligand, prompt_hbap_protein
            prompt_hbap_ligand_2_, prompt_hbap_protein_2_ = prompt_hbap_ligand_2, prompt_hbap_protein_2
            prompt_hbap_ligand_3_, prompt_hbap_protein_3_ = prompt_hbap_ligand_3, prompt_hbap_protein_3

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }



def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
