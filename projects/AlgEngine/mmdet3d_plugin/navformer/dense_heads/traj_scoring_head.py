import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.builder import HEADS

from mmdet3d_plugin.uniad.custom_modules.attn import MemoryEffTransformer
from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


@HEADS.register_module()
class TrajScoringHead(nn.Module):
    def __init__(
        self, 
        num_poses: int, 
        d_ffn: int, 
        d_model: int, 
        num_commands: int, 
        vocab_path: str,
        nhead: int, nlayers: int, 
        normalize_vocab_pos=True, 
        use_nerf=True, 
        transformer_decoder=None,
        reward_shaping=False,
        use_soft_imi=False,
        **kwargs):
        super().__init__()
        self._num_poses = num_poses

        self.transformer = build_transformer_layer_sequence(
            transformer_decoder
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.use_soft_imi = use_soft_imi
        self.reward_shaping = reward_shaping

        if not self.reward_shaping:
            self.heads = nn.ModuleDict({
                'imi': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                )
            })
        else:
            self.heads = nn.ModuleDict({
                'noc': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'da':
                    nn.Sequential(
                        nn.Linear(d_model, d_ffn),
                        nn.ReLU(),
                        nn.Linear(d_ffn, 1),
                    ),
                'ttc': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'comfort': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'progress': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
                'imi': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                )
            })

        self.normalize_vocab_pos = normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                use_lora=False,
                attn_use_lora=False,
            )
        self.use_nerf = use_nerf

        if self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
            # hist status: 5 steps (4hist + 1 curr) * [4(command)+xy+vxvy]
            self.status_embed = nn.Sequential(
                nn.Linear(4+24+2, d_model),
                nn.ReLU(),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
            self.status_embed = nn.Sequential(
                nn.Linear(4+2+2, d_model),
                nn.ReLU(),
            )

    @force_fp32(apply_to=("result","gt_pdm_score","sdc_planning"))
    def loss(
        self,
        result=None,
        gt_pdm_score=None,
        sdc_planning=None,
        sdc_planning_mask=None,
        il_target=None,
        il_target_mask=None,
        ):

        loss_dict = dict()
        # imitation_loss
        vocab = result["trajectory_vocab"][None, :, 4::5, :2]
        dist = torch.linalg.norm(vocab - sdc_planning[..., :2], dim=-1) * sdc_planning_mask[:, :, :, 0]
        imi_label = torch.argmin(dist.sum(-1), dim=-1)

        if self.use_soft_imi:
            soft_imi_label = torch.softmax(-dist.sum(-1),dim=-1)
            loss_dict['loss.imi'] = torch.mean(F.cross_entropy(result['imi'], soft_imi_label, reduction='none'))
        else:
            loss_dict['loss.imi'] = torch.mean(F.cross_entropy(result['imi'], imi_label, label_smoothing=0.2, reduction='none'))

        # Add reward shaping loss
        if self.reward_shaping:
            loss_dict['loss.noc'] = 3 * torch.mean(F.binary_cross_entropy(
                result['noc'], gt_pdm_score['no_at_fault_collisions'], reduction='none'
            ))
            loss_dict['loss.da'] = 3 * torch.mean(F.binary_cross_entropy(
                result['da'], gt_pdm_score['drivable_area_compliance'], reduction='none'
            ))
            loss_dict['loss.ttc'] = 2 * torch.mean(F.binary_cross_entropy(
                result['ttc'], gt_pdm_score['time_to_collision_within_bound'], reduction='none'
            ))
            loss_dict['loss.comfort'] = torch.mean(F.binary_cross_entropy(
                result['comfort'], gt_pdm_score['comfort'], reduction='none'
            ))
            loss_dict['loss.progress'] = torch.mean(F.binary_cross_entropy(
                result['progress'], gt_pdm_score['ego_progress'], reduction='none'
            ))

        return loss_dict

    @auto_fp16(apply_to=("bev_embed"))
    def forward(
        self,
        bev_embed,
        command=None,
        sdc_planning_past=None, # 1 x 4 x 4
        sdc_status=None,
        sdc_planning_mask_past=None,  # 1 x 4 x 4
        gt_pre_command_sdc=None, #1*4
    ):  
        gt_pre_command_sdc = gt_pre_command_sdc[:, 0, :, 0]
        sdc_planning_past = sdc_planning_past[:, 0]

        full_cmd = torch.cat([gt_pre_command_sdc, command[:, None]], dim=1)
        #[b, 5, 4]
        full_cmd = full_cmd.long()
        cmd_one_hot = F.one_hot(full_cmd, num_classes=4)
        cmd_one_hot = cmd_one_hot.float()

        full_ego_status = torch.cat([sdc_planning_past, sdc_status[:, None]], dim=1)

        if self.use_nerf:
            enc_ego_status = torch.cat([
                cmd_one_hot,
                nerf_positional_encoding(full_ego_status[..., :2]),
                torch.cos(full_ego_status[..., -1])[..., None],
                torch.sin(full_ego_status[..., -1])[..., None],
            ], dim=-1
            )
        else:
            enc_ego_status = torch.cat([
                cmd_one_hot,
                full_ego_status[..., :2],
                torch.cos(full_ego_status[..., -1])[..., None],
                torch.sin(full_ego_status[..., -1])[..., None],
            ], dim=-1
            )

        enc_ego_status = enc_ego_status.float()

        status_encoding = self.status_embed(enc_ego_status)

        sdc_planning_mask_past = sdc_planning_mask_past[:, 0, :, 0].float()
        b = sdc_planning_mask_past.shape[0]
        sdc_planning_mask_past = torch.cat([sdc_planning_mask_past, torch.zeros((b, 1)).to(status_encoding.device)],dim=1)
        sdc_planning_mask_past = sdc_planning_mask_past[:, :, None]

        status_encoding = torch.max(status_encoding * sdc_planning_mask_past, dim=1)[0]

        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_embed.shape[1]

        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )

        embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
        if self.normalize_vocab_pos:
            embedded_vocab = self.encoder(embedded_vocab)

        embedded_vocab = embedded_vocab + status_encoding.unsqueeze(1)
        device = embedded_vocab.device

        dist_status = self.transformer(
            embedded_vocab, 
            bev_embed=bev_embed,
            reference_trajs=self.vocab.data.unsqueeze(0).expand(B, -1, -1, -1),
            spatial_shapes=torch.tensor([[200, 200]], device=device),
            level_start_index=torch.tensor([0], device=device)
        )

        result = {}
        # selected_indices: B,
        for k, head in self.heads.items():
            if k == 'imi':
                result[k] = head(dist_status).squeeze(-1)
            else:
                result[k] = head(dist_status).squeeze(-1).sigmoid()

        if not self.reward_shaping:
            scores = result['imi'].log_softmax(-1)
        else:
            scores = (
                    0.05 * result['imi'].log_softmax(-1) +
                    0.5 * result['noc'].log() +
                    0.5 * result['da'].log() +
                    8.0 * (5 * result['ttc'] + 2 * result['comfort'] + 5 * result['progress']).log()
            )

        result["scores"] = scores
        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices

        return result
