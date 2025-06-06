import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
#from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from models.hyper_domain_fusion import HyperDomainFusionModule
from perlin_noise import PerlinNoise
import numpy as np

class Model(torch.nn.Module):

    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super().__init__()
        # 'vit_base_patch8_224_dino'
        # Determine if to output features.
        #self.device = 'cpu'
        self.device = device

        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        ## RGB backbone
        print(checkpoint_path)
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        #self.domain_classifier = DomainLearner(feature_dim = 784*768, num_domain = 2)
        #self.domain_classifier = DomainLearner(feature_dim = 784*768, num_domain = 2)
        #self.ift = IFT_Module(input_dim = 768, output_dim = 768, logit_scale = 1024).to(self.device)

        self.upsample = nn.ConvTranspose2d(
            768, 768, 4, stride=2, padding=1, groups=2
        )

        self.hyper_block = HyperDomainFusionModule(input_dim = 768, output_dim = 768).cuda()

        ## XYZ backbone
        
        if xyz_backbone_name=='Point_MAE':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group)
            #self.xyz_backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        elif xyz_backbone_name=='Point_Bert':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt("checkpoints/Point-BERT.pth")

        #print(self.rgb_backbone.blocks[0])
        self.layer0 = [self.rgb_backbone.patch_embed, self.rgb_backbone._pos_embed, self.rgb_backbone.norm_pre]
        self.layers = self.layer0 + [i for i in self.rgb_backbone.blocks]
        #self.num_disturb = 10
        self.num_disturb = 10
        print('total layers length: ')
        print(len(self.layers))
 

    def forward_rgb_features(self, x, disturb = False):
        #x = self.rgb_backbone.patch_embed(x)
        #x = self.rgb_backbone._pos_embed(x)
        #x = self.rgb_backbone.norm_pre(x)
        #x = self.rgb_backbone.blocks(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            #print(disturb)
            #print(idx)
            if disturb and idx < self.num_disturb:
                x, alpha, beta = self.local_perturb(x)
                #x, alpha, beta = self.inject_noise_with_adain(x)

        x = self.rgb_backbone.norm(x)

        feat = x[:,1:].permute(0, 2, 1).view(-1, 768, 28, 28)
        #print(feat.shape)
        return feat


    def generate_perlin_noise(self, batch_size, height, width, octaves=3, persistence=0.5, lacunarity=2.0):
        noise = torch.zeros((batch_size, height, width)).cuda()
        
        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave
            
            x_coords = torch.arange(height).reshape(-1, 1) / frequency
            y_coords = torch.arange(width).reshape(1, -1) / frequency
            
            # Create a meshgrid for the x and y coordinates
            x_grid, y_grid = torch.meshgrid(x_coords.squeeze(), y_coords.squeeze(), indexing='ij')
            
            # Generate noise based on sine wave for the given frequency and amplitude
            noise_value = torch.sin(2 * torch.pi * (x_grid + y_grid)).cuda() * amplitude
            
            noise += noise_value.unsqueeze(0)  # Add to the batch
            
        # Normalize to range [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise
    
    def local_perturb(self, features, noise_type='perlin', local_noise_std=0.75, alpha_scale=0.1, beta_scale=0.1):
        #print('Disturbing...................................................')
        #print(features.shape)  # Shape: (B, 784, 768)

        # Prepare matrices for noise generation
        zeros_mat = torch.zeros(features.mean(dim=2, keepdim=True).shape).cuda()
        ones_mat = torch.ones(features.mean(dim=2, keepdim=True).shape).cuda()

        if noise_type == 'gaussian':
            alpha = torch.normal(zeros_mat, local_noise_std * ones_mat)  # Size: B, 1, 1, C
            beta =  torch.normal(zeros_mat, local_noise_std * ones_mat)  # Size: B, 1, 1, C
            print(alpha)

            # Compute local features with Gaussian noise
            local_features = ((1 + alpha) * features - alpha * features.mean(dim=2, keepdim=True) +
                              beta * features.mean(dim=2, keepdim=True))

            return local_features, alpha, beta

        elif noise_type == 'perlin':
            batch_size, _, _ = features.shape
            height = features.shape[1]
            width = features.shape[2]

            # Generate Perlin noise using matrix operations
            noise_tensor = self.generate_perlin_noise(batch_size, height, width)

            # Generate alpha and beta for Perlin noise as well
            alpha = torch.normal(zeros_mat, local_noise_std * ones_mat)  # Size: B, 1, 1, C
            beta = torch.normal(zeros_mat, local_noise_std * ones_mat)  # Size: B, 1, 1, C

            # Calculate perturbed mean and apply alpha and beta
            perturbed_mean = features.mean(dim=2, keepdim=True) * (1 + alpha_scale) + (local_noise_std * noise_tensor * beta_scale)

            # Add Perlin noise scaled by the perturbed mean
            local_features = features + perturbed_mean

            return local_features, alpha, beta
        

    def inject_noise_with_adain(self, features, noise_type='perlin', alpha_scale=0.1, beta_scale=0.1, local_noise_std=0.75):
        """
        Inject noise into feature map using Adaptive Instance Normalization (AdaIN) with Gaussian or Perlin noise.

        Args:
            features (torch.Tensor): Input feature map of shape (B, C, H, W).
            noise_type (str): Type of noise to inject ('gaussian' or 'perlin').
            alpha_scale (float): Scaling factor for alpha (mean perturbation).
            beta_scale (float): Scaling factor for beta (variance perturbation).
            local_noise_std (float): Standard deviation for local noise generation.

        Returns:
            torch.Tensor: Feature map with noise injected.
        """
        # Compute the original feature statistics
        features = features.reshape(1, 768, 28, 28)
        batch_size, channels, height, width = features.shape
        mu_o = features.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        sigma_o = features.std(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)

        # Initialize alpha and beta
        zeros_mat = torch.zeros_like(mu_o)
        ones_mat = torch.ones_like(mu_o)
        alpha = torch.normal(zeros_mat, local_noise_std * ones_mat)  # Shape: (B, C, 1, 1)
        beta = torch.normal(zeros_mat, local_noise_std * ones_mat)  # Shape: (B, C, 1, 1)

        # Generate perturbed statistics
        if noise_type == 'gaussian':
            mu_p = mu_o * (1 + alpha * alpha_scale)  # Perturbed mean
            sigma_p = sigma_o * (1 + beta * beta_scale)  # Perturbed variance
        elif noise_type == 'perlin':
            # Generate Perlin noise using matrix operations
            noise_tensor = self.generate_perlin_noise(batch_size, height, width, channels).to(features.device)
            mu_p = mu_o * (1 + alpha * alpha_scale) + noise_tensor.mean(dim=(2, 3), keepdim=True) * beta_scale
            sigma_p = sigma_o * (1 + beta * beta_scale) + noise_tensor.std(dim=(2, 3), keepdim=True) * beta_scale
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        # Normalize features using original statistics
        normalized_features = (features - mu_o) / (sigma_o + 1e-5)

        # Apply perturbed statistics to restore features
        perturbed_features = normalized_features * sigma_p + mu_p

        return perturbed_features, alpha, beta

    def forward(self, rgb, disturb = False):
        
        rgb_features = self.forward_rgb_features(rgb, disturb = disturb)

        #xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

        return rgb_features#, xyz_features, center, ori_idx, center_idx



def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            #if incompatible.missing_keys:
            #    print('missing_keys')
            #    print(
            #            incompatible.missing_keys
            #        )
            #if incompatible.unexpected_keys:
            #    print('unexpected_keys')
            #    print(
            #            incompatible.unexpected_keys

            #        )

            # print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys

                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B,C,N = pts.shape
            pts = pts.transpose(-1, -2) # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood,  center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
            # prepare cls
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
            # add pos embedding
            pos = self.pos_embed(center)
            # final input
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx 
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

            group_input_tokens = self.encoder(neighborhood)  # B G N

            pos = self.pos_embed(center)
            # final input
            x = group_input_tokens
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx
        
