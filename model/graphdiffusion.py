import numpy as np
import torch
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import Parameter


class IAP_base(nn.Module):
    def __init__(self, config, low_bound, high_bound) -> None:
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.num_steps = self.config['num_steps']

        self.diffusion_model = SpatialTemporalEncoding(config=config, low_bound=low_bound, high_bound=high_bound)
        if config["schedule"] == "quad":
            self.beta = torch.linspace(
                config["beta_start"] ** 0.5, config["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config["schedule"] == "linear":
            self.beta = torch.linspace(
                config["beta_start"], config["beta_end"], self.num_steps
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat, dim=0)
        self.alpha_prev = F.pad(self.alpha[:-1], (1, 0), value=1.)
        self.alpha_torch = self.alpha.float().to(self.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    def get_randmask(self, observed_mask, sample_ratio):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            # sample_ratio = 0.2 * np.random.rand()
            sample_ratio = sample_ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def trainstep(self, observed_data, observed_mask, adj, node_type, is_train, set_t=-1):

        cond_mask = self.get_randmask(observed_mask, self.config['missing_ratio'])
        cond_mask = cond_mask.to(self.device)
        B = observed_data.shape[0]
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.config['num_steps'], [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1,1,1)
        noise = torch.randn_like(observed_data)

        mean = (observed_data * cond_mask).sum(dim=1,keepdim=True)/(cond_mask.sum(dim=1, keepdim=True)+1e-5)
        mean_ = mean.expand_as(observed_data)
        observed_data_imputed = torch.where(cond_mask.bool(), observed_data, mean_)


        noisy_data = (current_alpha ** 0.5) * observed_data_imputed + (1.0 - current_alpha) ** 0.5 * noise
        noisy_data = noisy_data.to(self.device)

        total_input = torch.stack([observed_data_imputed, (1-cond_mask)*noisy_data], dim=3)
        predicted = self.diffusion_model(total_input, cond_mask, adj, node_type, t)

        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def impute(self, observed_data, observed_mask, adj, node_type, n_samples):
        B, T, K, N = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, T, K, N)
        mean = (observed_data*observed_mask).sum(1, keepdim=True)/(observed_mask.sum(1, keepdim=True)+1e-5)
        mean_ = mean.expand_as(observed_data)

        with torch.no_grad():
            for i in range(n_samples):
                # generate noisy observation for unconditional model
                current_sample = torch.randn_like(observed_data).to(self.device) + mean_
                observed_data_imputed = torch.where(observed_mask.bool(), observed_data, mean.expand_as(observed_data))

                for t in range(self.num_steps - 1, -1, -1):
                    noisy_target =  current_sample
                    total_input = torch.stack([observed_data_imputed ,(1-observed_mask)*noisy_target],dim=3)
                    predicted = self.diffusion_model(total_input, observed_mask, adj, node_type, (torch.ones(B) * t).long().to(self.device))

                    coeff1 = (1-self.alpha_prev[t])*(self.alpha_hat[t])**0.5 / (1 - self.alpha[t])
                    coeff2 = ((1-self.alpha_hat[t])*(self.alpha_prev[t])**0.5) / (1 - self.alpha[t])
                    current_sample = coeff1 *current_sample + coeff2 * predicted
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach().cpu()
        return imputed_samples


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SpatialTemporalEncoding(nn.Module):
    def __init__(self, config, low_bound, high_bound) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.input_projection = Conv1d_with_init(2, config['hidden_channels'], 1)
        self.mid_projection = Conv1d_with_init(config['hidden_channels'], 2*config['hidden_channels'], 1)
        self.output_projection = Conv1d_with_init(config['hidden_channels'], 1, 1)

        self.spatial_encoding = GCN(self.config['hidden_channels']+16, self.config['hidden_channels'], 3)
        self.time_encoding = LinearAttentionTransformer(dim=self.config['hidden_channels'], depth=1, heads=1, max_seq_len=16, n_local_attn_heads=0, local_attn_window_size=0)
        self.diffusion_embedding = DiffusionEmbedding(num_steps=config['num_steps'], embedding_dim=config['diffusion_embedding_size'])
        self.gn = nn.GroupNorm(4, config['hidden_channels'])
        self.learnable_position_embedding = nn.Parameter(torch.randn(4443, 16), requires_grad=True)

        self.is_sea = torch.from_numpy(np.load('/home/mafzhang/data/PRE/8d/is_sea.npy')).to(self.device)
        self.is_sea = self.is_sea.bool()

        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x, mask, adj, node_type, diffusion_step):

        # projection
        B, T, K, C, N = x.shape
        x = rearrange(x, 'b t k c n->(b t k) c n')
        x = self.input_projection(x)
        C = x.shape[1]
        x = F.relu(x)
        x = rearrange(x, '(b t k) c n -> b t k c n', b=B, t=T, k=K)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = diffusion_emb.unsqueeze(1).unsqueeze(1).unsqueeze(-1)

        input = x + diffusion_emb

        #temporal encoding
        x = rearrange(input, 'b t k c n->(b k n) t c')
        x = self.time_encoding(x)
        x = rearrange(x, '(b k n) t c->(b k t) n c', b=B, n=N, k=K)

        #spatial encoding
        x_in = x
        batch_position_embed = self.learnable_position_embedding.unsqueeze(0).expand(x.shape[0],-1,-1)
        x = self.spatial_encoding(x, adj, node_type, batch_position_embed)
        x = rearrange(x, '(b k t) n c-> (b t k) c n', b=B, k=K, t=T)
        x_in = rearrange(x_in, '(b k t) n c-> (b t k) c n', b=B, k=K, t=T)
        x = x + x_in
        x = self.gn(x)

        x = self.mid_projection(x)
        gate, filter = torch.chunk(x, 2, dim=1)
        y = torch.sigmoid(gate)*torch.tanh(filter)
        y = self.output_projection(y)
        y = y.reshape(B, T, K, N)

        low_bound = self.low_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        high_bound = self.high_bound.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(y)
        y = torch.clamp(y, low_bound, high_bound)
        return y

    def get_time_embedding(self):
        pe = torch.zeros(self.config['in_len'], self.config['hidden_channels'], self.config['height'], self.config['width']).to(self.device)
        position = torch.arange(self.config['in_len']).unsqueeze(1).to(self.device)

        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, self.config['hidden_channels'], 2).to(self.device) / self.config['hidden_channels']
        )
        pe[:, 0::2, :, :] = torch.sin(position * div_term).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config['height'], self.config['width'])
        pe[:, 1::2, :, :] = torch.cos(position * div_term).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config['height'], self.config['width'])
        return pe

    def get_position_embeding(self):
        pe = torch.zeros(8, self.config['height'], self.config['width'])

        d_model = 4
        height = self.config['height']
        width = self.config['width']
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.to(self.device)

        return pe




class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class DenseLayer(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias.data)

    def forward(self,
                node_feats, # input node features
                ):
        node_feats = self.linear(node_feats)
        return node_feats


class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                node_type,
                position_embedding
                ):

        # Apply linear layer and sort nodes by head
        node_feats = torch.matmul(adj_matrix, node_feats)
        position_embedding = torch.matmul(adj_matrix, position_embedding)
        node_feats = torch.cat([node_feats, position_embedding], dim=-1)
        node_feats = self.linear(node_feats)
        return node_feats
