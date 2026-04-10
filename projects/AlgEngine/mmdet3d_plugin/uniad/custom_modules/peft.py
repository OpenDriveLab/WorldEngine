import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 


class LoRALinear(nn.Module):
    """
    LoRA layer: Low-Rank Adaptation.
    This layer consists of a low-rank decomposition of weight updates.
    """
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.1, **kwargs):
        super(LoRALinear, self).__init__()

        self.use_si = False
        self.multi_domain = 0
        if 'use_si' in kwargs.keys():
            self.model = LoRALinearSI(
                in_features, out_features, r, alpha, **kwargs
            )
            self.use_si = True
        elif 'multi_domain' in kwargs.keys():
            self.r = r
            self.alpha = alpha
            self.multi_domain = kwargs['multi_domain']
            a_list, b_list, drop_list = [], [], []
            for i in range(self.multi_domain):
                a_list.append(nn.Linear(in_features, r, bias=False))
                b_list.append(nn.Linear(r, out_features, bias=False))
                drop_list.append(nn.Dropout(dropout))
            self.A = nn.ModuleList(a_list)
            self.B = nn.ModuleList(b_list)
            self.drop =nn.ModuleList(drop_list)
            self.scaling = alpha / r
            self._init_weights()
        else:
            self.r = r
            self.alpha = alpha
            
            # Low-rank decomposition matrices
            self.A = nn.Linear(in_features, r, bias=False)  # Down-projection
            self.drop = nn.Dropout(dropout)
            self.B = nn.Linear(r, out_features, bias=False)  # Up-projection

            nn.init.zeros_(self.B.weight)
            nn.init.normal_(self.A.weight, std=1 / r)
            self.lora_name = "lora_layer"  # Unique name
            
            # Scaling factor for LoRA
            self.scaling = alpha / r
    
    def _init_weights(self):
        for layer in self.A:
            nn.init.normal_(layer.weight, std=1 / self.r)
        for layer in self.B:
            nn.init.zeros_(layer.weight)

    def forward(self, x, task_mask=None, i=None,task_idx=None):
        # Apply low-rank update: scaling * (A(x) * B)
        if self.use_si:
            return self.model(x)
        return self.scaling * self.B(self.drop(self.A(x)))
    
    def update_si_information(self):
        if self.use_si:
            self.model.update_si_information()
    
    def finalize_si_importance(self):
        if self.use_si:
            self.model.finalize_si_importance()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, prior_std=0.1, dropout=0.1, **kwargs):
        """
        Bayesian LoRA Layer: Instead of deterministic weights, 
        it learns a distribution over LoRA parameters using Bayesian inference.
        
        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            rank (int): LoRA rank.
            prior_std (float): Standard deviation of the Gaussian prior.
        """
        super(BayesianLinear, self).__init__()

        # Learnable means and log-variances (for stability)
        self.scaling = 1 / r

        self.A_mu = nn.Parameter(torch.randn(in_features, r) * (1 / r))
        self.A_logvar = nn.Parameter(torch.randn(in_features, r) * (1 / r))

        self.B_mu = nn.Parameter(torch.randn(r, out_features) * (1 / r))
        self.B_logvar = nn.Parameter(torch.randn(r, out_features) * (1 / r))

        self.drop = nn.Dropout(dropout)

        # Gaussian prior (zero mean)
        self.prior_std = prior_std
        

    def sample_weights(self):
        """
        Reparameterization Trick: Sample weights from Gaussian distribution.
        """
        A_std = torch.exp(0.5 * self.A_logvar)
        B_std = torch.exp(0.5 * self.B_logvar)

        # Sample weights using reparameterization
        B_sample = self.B_mu + B_std * torch.randn_like(B_std)
        A_sample = self.A_mu + A_std * torch.randn_like(A_std)

        return A_sample, B_sample

    # def kl_divergence(self):
    #     """
    #     Compute KL divergence between learned weight distributions and the prior.
    #     """
    #     W_std = torch.exp(0.5 * self.W_logvar)
    #     A_std = torch.exp(0.5 * self.A_logvar)

    #     kl_W = (self.W_mu**2 + W_std**2 - 2 * torch.log(W_std) - 1).sum()
    #     kl_A = (self.A_mu**2 + A_std**2 - 2 * torch.log(A_std) - 1).sum()

    #     return 0.5 * (kl_W + kl_A)

    def forward(self, x):
        """
        Forward pass with Bayesian weight sampling.
        """
        if self.training:
            A, B = self.sample_weights()
        else:
            A, B = self.A_mu, self.B_mu  # Use deterministic weights for testing
        
        out = self.drop(x @ A)
        return out @ B  # LoRA forward pass
    
class LoRALinearSI(nn.Module):
    def __init__(self, in_features, out_features, r=8, 
        alpha=1.0, lambda_si=0.1, si_decay=0.99, dropout=0.1,
        plasticity_base=0.5, sparsity_threshold=1e-3):
        super().__init__()
        self.r = r
        self.alpha = alpha  # Base scaling factor for LoRA updates
        self.lambda_si = lambda_si  # Strength of SI regularization
        self.si_decay = si_decay  # Decay factor for importance updates
        self.plasticity_base = plasticity_base  # Base plasticity level
        self.sparsity_threshold = sparsity_threshold  # Threshold for detecting sparse weights
        
        # LoRA trainable parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_features))
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.lora_B)
        nn.init.normal_(self.lora_A, std=1 / r)
        
        # Synaptic Intelligence (SI) buffers
        self.register_buffer("omega_A", torch.zeros_like(self.lora_A))  # Importance of lora_A
        self.register_buffer("omega_B", torch.zeros_like(self.lora_B))  # Importance of lora_B
        self.register_buffer("prev_params_A", self.lora_A.clone().detach())
        self.register_buffer("prev_params_B", self.lora_B.clone().detach())
        self.register_buffer("trajectory_A", torch.zeros_like(self.lora_A))  # Tracks updates for lora_A
        self.register_buffer("trajectory_B", torch.zeros_like(self.lora_B))  # Tracks updates for lora_B
        # self.register_buffer("plasticity", torch.ones_like(self.lora_A) * self.plasticity_base)  # Dynamic plasticity control
        
    def forward(self, x):
        adaptive_alpha = self.alpha #* self.plasticity  # Scale LoRA update based on plasticity
        lora_update = torch.matmul(x, self.lora_A)
        lora_update = self.drop(lora_update)
        lora_update = torch.matmul(lora_update, self.lora_B)
        return adaptive_alpha * lora_update # Dynamic scaling
    
    def update_si_information(self):
        """Update Synaptic Intelligence importance online."""
        if self.lora_A.grad is not None:
            delta_theta_A = self.lora_A - self.prev_params_A
            self.trajectory_A += delta_theta_A * self.lora_A.grad  # Path integral for A
            self.prev_params_A = self.lora_A.detach().clone()
        
        if self.lora_B.grad is not None:
            delta_theta_B = self.lora_B - self.prev_params_B
            self.trajectory_B += delta_theta_B * self.lora_B.grad  # Path integral for B
            self.prev_params_B = self.lora_B.detach().clone()
    
    def compute_sparsity(self, param):
        """Compute the sparsity score: fraction of near-zero values."""
        return torch.mean((torch.abs(param) < self.sparsity_threshold).float())
    
    def finalize_si_importance(self):
        """Compute final importance after training a task and adjust plasticity."""
        self.omega_A = self.si_decay * self.omega_A + (1 - self.si_decay) * (self.trajectory_A / (self.lora_A**2 + 1e-6)).detach()
        self.omega_B = self.si_decay * self.omega_B + (1 - self.si_decay) * (self.trajectory_B / (self.lora_B**2 + 1e-6)).detach()
        self.trajectory_A.zero_()
        self.trajectory_B.zero_()
        
        # Compute sparsity scores
        # sparsity_A = self.compute_sparsity(self.lora_A)
        # sparsity_B = self.compute_sparsity(self.lora_B)
        
        # Adjust plasticity dynamically based on sparsity
        # self.plasticity = torch.exp(-self.omega_A) * (1 - sparsity_A)
    
    def si_loss(self):
        """Compute the SI loss term for both LoRA parameters."""
        loss_A = torch.sum(self.omega_A * (self.lora_A - self.prev_params_A) ** 2)
        loss_B = torch.sum(self.omega_B * (self.lora_B - self.prev_params_B) ** 2)
        return self.lambda_si * (loss_A + loss_B)
    
    def set_plasticity(self, value: float):
        """Manually set a global plasticity value if needed."""
        self.plasticity.fill_(value)



class MOELoRALinear(nn.Module):
    """
    LoRA layer: Low-Rank Adaptation.
    This layer consists of a low-rank decomposition of weight updates.
    """
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.1, num_task=3, **kwargs):
        super(MOELoRALinear, self).__init__()
        
        self.loras = nn.ModuleList([
            LoRALinear(
                in_features, 
                out_features, 
                r, alpha, dropout, **kwargs) for _ in range(num_task)
            ])
        self.num_task=num_task
    
    def forward(self, x, i):
        if isinstance(i, int):
            return self.loras[i](x)
        elif i.dtype == torch.float:
            orig_shape = x.shape
            b = orig_shape[0]
            new_shape = (b//self.num_task, self.num_task) + orig_shape[1:]
            x = x.reshape(new_shape)
            mask_shape = i.shape + (1,)*len(orig_shape[1:])
            i = i.reshape(mask_shape)
            res_list = torch.stack([
                self.loras[t](x[:, t]) for t in range(self.num_task)
            ], dim=1) #[b, task, class, dim]
            res_list = res_list * i
            res_list = res_list.reshape(orig_shape[:-1]+(-1,))
            return res_list

        res_list = torch.stack([
            self.loras[t](x) for t in range(self.num_task)
        ], dim=1) #[b, task, class, dim]
        
        b = res_list.shape[0]
        res =  res_list[torch.arange(b), i]
        # print(res.shape, i.shape)
        return res



class ZeroAdapter(nn.Module):
    """
    LoRA layer: Low-Rank Adaptation.
    This layer consists of multiple LoRA mitigating catastrophic forgetting
    """
    def __init__(self, in_features, out_feature, dropout=0.1, **kwargs):
        super(ZeroAdapter, self).__init__()
        mid_feature = in_features // 2
        self.down_linear = nn.Linear(in_features, mid_feature)
        self.up_linear = nn.Linear(mid_feature, out_feature)

        nn.init.zeros_(self.down_linear.weight)
        nn.init.zeros_(self.down_linear.bias)

        nn.init.zeros_(self.up_linear.weight)
        nn.init.zeros_(self.up_linear.bias)
  
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.lora_name = "lora_layer"  # Unique name
    
    
    def forward(self, x):
        x = self.down_linear(x)
        x = self.drop(self.act(x))
        x = self.up_linear(x)
        return x



class LoRAMoECLAdapter(nn.Module):
    def __init__(self, in_features, mid_feature, out_feature,
        num_task=6, r=8, alpha=1.0, dropout=0.1, **kwargs):
        super(LoRAMoECLAdapter, self).__init__()
        self.r = r
        self.alpha = alpha
        self.num_task = num_task
        
        self.adapters = nn.ModuleList([
            nn.Sequential(
                LoRALinear(in_features, mid_feature, r, alpha, dropout),
                nn.Dropout(dropout),
                nn.ReLU(),
                LoRALinear(mid_feature, out_feature, r, alpha, dropout),
            )
            for _ in range(num_task)
            ])
        
        self.router = nn.Linear(in_features, num_task)
        self.out_drop = nn.Dropout(dropout)

        self.lora_name = "lora_layer"  # Unique name
    
    def forward(self, x, i=None):
        outputs = []
        logits = self.router(x)
        route_prob = logits.softmax(-1)

        for i in range(self.num_task):
            outputs.append(self.adapters[i](x))
        outputs = torch.stack(outputs, dim=-2)
        outputs = torch.sum(outputs * route_prob[..., None], dim=-2)
        outputs = self.out_drop(outputs)

        return outputs
  

class LoRACLAdapter(nn.Module):
    """
    LoRA layer: Low-Rank Adaptation.
    This layer consists of multiple LoRA mitigating catastrophic forgetting
    """
    def __init__(self, in_features, out_feature,
        num_task=6, r=8, alpha=1.0, dropout=0.1, **kwargs):
        super(LoRACLAdapter, self).__init__()
        self.r = r
        self.alpha = alpha
        
        self.loras = nn.ModuleList([
            LoRALinear(in_features, out_feature, r, alpha, dropout) for _ in range(num_task)
            ])
        
        self.attn_weights = nn.ModuleList([nn.Linear(out_feature, 1) for _ in range(num_task)])
        self.attn_drop = nn.Dropout(dropout)
        
        self.num_task = num_task
        
        # Scaling factor for LoRA
        self.scaling = alpha / r
        self.lora_name = "lora_layer"  # Unique name

    def forward(self, x, task_mask=None):
        # Apply low-rank update: scaling * (A(x) * B)
        #x:[b, 1, d]

        assert task_mask is not None

        outputs = []
        output_weights = []

        for i in range(self.num_task):
            out = self.loras[i](x)
            weight_out = self.attn_weights[i](out)
            outputs.append(out)
            output_weights.append(weight_out)

        outputs = torch.cat(outputs, dim=1)
        output_weights = torch.cat(output_weights, dim=1)
        output_weights = output_weights.softmax(1)
        outputs = outputs * self.attn_drop(output_weights)

        # detach invalid outputs:
        task_mask = task_mask[0]
        task_mask = task_mask.unsqueeze(-1).expand(outputs.shape[0], -1, outputs.shape[2])
        # print(task_mask.shape, outputs.shape)
        outputs[task_mask==0] = outputs[task_mask==0].detach()
        outputs = outputs.sum(1)
        return outputs[:, None]


valid_lora_list = (LoRALinear, LoRACLAdapter, ZeroAdapter, LoRAMoECLAdapter, MOELoRALinear)


def lora_wrapper(
    module, 
    LoraLayer=LoRALinear,
    rank=8, alpha=1.0, dropout=0.1,
    num_task=6,
    **kwargs):
    """
    Creates a separate LoRA module that mirrors the Linear layers in the original model.
    """
    if isinstance(module, nn.ModuleList):
        lora_module = nn.ModuleList()
        for m in module:
           lora_module.append(lora_wrapper(
               m, LoraLayer, 
               rank=rank, alpha=alpha, dropout=dropout,num_task=num_task
           )) 
        return lora_module
    
    if isinstance(module, nn.ModuleDict):
        lora_module = nn.ModuleDict()
        for k,v in module.items():
           lora_module[f'lora_{k}'] = lora_wrapper(
               v, LoraLayer, 
               rank=rank, alpha=alpha, dropout=dropout,num_task=num_task
           )
        return lora_module

    if len(list(module.named_modules())) == 1 :
        if not isinstance(module, nn.Linear):
            print(f'Wrap non nn.Linear unit{type(module)}, skipping with Identity')
            return nn.Identity()
        lora_module = LoraLayer(module.in_features, module.out_features, 
                r=rank, alpha=alpha,dropout=dropout, num_task=num_task,**kwargs)
        return lora_module
    
    # sequential case
    
    lora_module = nn.Sequential()

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_layer = LoraLayer(child.in_features, child.out_features, 
                r=rank, alpha=alpha,dropout=dropout, num_task=num_task, **kwargs)
            lora_module.add_module(f'lora_{name}', lora_layer)
        elif isinstance(child, nn.Sequential):
            lora_module.add_module(f'lora_{name}', 
                lora_wrapper(child, 
                    LoraLayer, 
                    rank=rank, alpha=alpha, dropout=dropout,num_task=num_task,
                )
            )
        else:
            lora_module.add_module(f'lora_{name}', nn.Identity())

    return lora_module

def single_peft_forward(x, model, lora_model, lora_only=False, idx=None):
    if lora_only:
        return lora_model(x, i=idx)
    return model(x) + lora_model(x, i=idx)


def peft_wrapper_forward(x, model, lora_model, use_lora=True,
    layer_idx=-1, layer_name="", lora_only=False, task_idx=None):
    """
    Custom forward function to combine original model output with LoRA output.
    layer_idx: can be specified for (nn.ModuleList) model; Default: running sequentially through whole ModuleList
    layer_name: can be specified for (nn.ModuleDict) model; Default:running sequentially through whole ModuleDict
    lora_only: if lora_only=True, forward function will only pass through the lora layer when meet with matched Linear
    """
    if isinstance(model, nn.ModuleList):
        if layer_idx > -1:
            return single_peft_forward(x, model[layer_idx], lora_model[layer_idx], lora_only, task_idx)
    
    if isinstance(model, nn.ModuleDict):
        if layer_name != "":
            return single_peft_forward(x, model[layer_name], lora_model[layer_name], lora_only, task_idx)
    
    if len(list(model.named_modules())) == 1:
        return single_peft_forward(x, model, lora_model, lora_only, task_idx)

    def process_layer(orig_layer, lora_layer, x):
        """ Recursively process nested nn.Sequential layers """
        if isinstance(orig_layer, nn.Sequential) and isinstance(lora_layer, nn.Sequential):
            for o_layer, l_layer in zip(orig_layer.children(), lora_layer.children()):
                x = process_layer(o_layer, l_layer, x)
            return x
        else:
            if use_lora and not isinstance(lora_layer, nn.Identity):
                return single_peft_forward(x, orig_layer, lora_layer, lora_only, task_idx)
            else:
                return orig_layer(x)

    for orig_layer, lora_layer in zip(model.children(), lora_model.children()):
        x = process_layer(orig_layer, lora_layer, x)

    return x

def finetuning_detach(model):
    '''
    work with a detach for customed layer
    ensure if some sublayer inside containing such LoRA layer 
    or adapter with "lora_name" attribute,
    also have this finetuning function and lora_name attr
    '''
    for name, module in model.named_modules():
        if 'lora' in name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False # disable param
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.eval()

def frozen_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        # self.model  = nn.Sequential(
        #     nn.Linear(10, 20),
        #     nn.ReLU(),
        #     nn.Sequential(
        #         nn.Linear(20, 30),
        #         nn.ReLU(),
        #         nn.Linear(30, 40)
        #     )
        # )
        # self.model = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        self.model = nn.ModuleDict()
        for i in range(3):
            self.model[f'{i}'] = nn.Linear(10,10)
        self.lora_layer = lora_wrapper(
            self.model, 
            ZeroAdapter,
            rank=4, alpha=1.0)
    
    def forward(self, x):
        x = peft_wrapper_forward(x, self.model, self.lora_layer)
        return x

def retreive_bayesian_lora_param(module):
    '''
    input, any nn.Module
    searching for all Bayesian Lora param
    return: lora_dict: Dict[sub_name: Dict['A_mu','B_mu','A_logvar','B_logvar']]
    '''
    lora_dict = {}
    lora_list = set(['A_mu','B_mu','A_logvar','B_logvar'])
    if isinstance(module, BayesianLinear):
        lora_dict['.'] = dict()
        for name,m in module.named_parameters():
           lora_dict['.'][name] = m
        return lora_dict 
     
    for name,m in module.named_parameters():
        name_list = name.split('.')
        if name_list[-2] in lora_list:
            m_prefix = ".".join(name_list[:-2])
            if m_prefix not in lora_dict:
                lora_dict[m_prefix] = dict()
            lora_dict[m_prefix][name.split('.')[-1]] = m
    return lora_dict



def test_lora_si():
    from time import time
    import numpy as np

    lora_model = LoRALinearSI(
        256, 256, 16
    )
    t = []
    for _ in range(10):
        s = time()
        x = torch.randn(2, 256)
        y = lora_model(x)
        loss = lora_model.si_loss()
        t.append(time()-s)
        print(loss, np.mean(t))

def test_kl_lora():
    lora_layer = BayesianLinear(
        32, 32, r=8
    )
    inputs = torch.randn(4, 10, 32)
    out = lora_layer(inputs)

    bayesian_params = retreive_bayesian_lora_param(lora_layer)
    loss = 0.
    for v_dict in bayesian_params.values():
        print(v_dict.keys())
        B_std = torch.exp(0.5 * v_dict['B_logvar'])
        A_std = torch.exp(0.5 * v_dict['A_logvar'])

        kl_B = (v_dict['B_mu']**2 + B_std**2 - 2 * torch.log(B_std) - 1).sum()
        kl_A = (v_dict['A_mu']**2 + A_std**2 - 2 * torch.log(A_std) - 1).sum()

        module_loss = 0.5 * (kl_B + kl_A)
        loss += module_loss
    
    print(out.shape, loss)

# Example usage
if __name__ == "__main__":
    # Define a nested Sequential model
    # model = TestModule()
    # finetuning_detach(model)
    # x = torch.randn(4, 10)
    # print(model(x).shape)

    # # Print the model structure after attaching LoRA layers
    # print("Model structure after attaching LoRA layers:\n", model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    # test_lora_si()
    test_kl_lora()
