import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.builder import HEADS

from mmdet3d_plugin.uniad.custom_modules.attn import MemoryEffTransformer
from mmdet3d_plugin.uniad.custom_modules.peft import (LoRALinear,
    finetuning_detach, frozen_grad, peft_wrapper_forward, lora_wrapper, retreive_bayesian_lora_param)
from mmdet3d_plugin.utils import get_logger

from .traj_scoring_head import nerf_positional_encoding

logger = get_logger(__name__)


@HEADS.register_module()
class TrajScoringHeadRL(nn.Module):
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
        use_lora=False, 
        lora_rank=16, 
        trans_use_lora=False, 
        trans_lora_rank=16, 
        full_finetuning=False,
        rl_finetuning=False, 
        importance_sampling=True, 
        num_task=4,
        orig_IL=False,
        reward_shaping=False,
        use_soft_imi=False,
        rl_loss_weight=dict(
            bce=0.05,
            rank=2.0,
            PG=100.0,
            entropy=1.0
        ),
        hard_case_no_imi=False,
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

        self.full_finetuning = full_finetuning
        self.rl_finetuning = rl_finetuning
        self.rl_loss_weight = rl_loss_weight
        self.use_lora = use_lora
        self.importance_sampling = importance_sampling
        self.orig_IL = orig_IL
        self.hard_case_no_imi = hard_case_no_imi

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
                use_lora=trans_use_lora,
                attn_use_lora=trans_use_lora,
                lora_rank=trans_lora_rank,
                attn_lora_rank=trans_lora_rank,
                num_task=num_task
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

        if self.use_lora:
            self.pos_embed_lora = lora_wrapper(self.pos_embed, LoRALinear, 
                rank=lora_rank, alpha=1.0, dropout=0.1,num_task=num_task)
            self.status_embed_lora = lora_wrapper(self.status_embed, LoRALinear, 
                rank=lora_rank, alpha=1.0, dropout=0.1,num_task=num_task)

            self.heads_lora = lora_wrapper(self.heads, LoRALinear, rank=lora_rank, alpha=1.0, dropout=0.1, num_task=num_task)

            finetuning_detach(self)

            # The reward shaping heads are newly initialized
            if self.orig_IL and self.reward_shaping:
                for head_name in self.heads:
                    if head_name == 'imi':
                        continue
                    for param in self.heads[head_name].parameters():
                        param.requires_grad = True

    def compute_log_pi(self, result, prefix=''):
        '''
        Compute the log of the probability of the trajectory
        using log-sum-exp trick:
        log(\sum_i exp(x_i)) = log(\sum_i exp(x_i - x_max)) + x_max
        '''
        if self.orig_IL and prefix == 'orig_':
            return result[prefix+'imi'].log_softmax(-1)

        if self.reward_shaping:
            log_pis = [
                0.05 * result[prefix+'imi'].log_softmax(-1),
                0.5 * result[prefix+'noc'].clamp(min=1e-6).log(),
                0.5 * result[prefix+'da'].clamp(min=1e-6).log(),
                8.0 * (5 * result[prefix+'ttc'] + 2 * result[prefix+'comfort'] + 5 * result[prefix+'progress']).clamp(min=1e-6).log()
            ]
            all_log_pi = torch.stack(log_pis, dim=-1).sum(-1)
            all_log_pi = all_log_pi - torch.logsumexp(all_log_pi, dim=-1, keepdim=True)
            return all_log_pi
        else:
            return result[prefix+'imi'].log_softmax(-1)


    def compute_RL_loss(self, result, gt_pdm_score, prefix=''):
        '''
        Compute the loss for the RL agent
        '''
        EPS = 1e-6
        log_pi = self.compute_log_pi(result, prefix)
        original_log_pi = self.compute_log_pi(result, 'orig_')

        pi = F.softmax(log_pi, dim=-1).clamp(min=EPS)
        orig_pi = F.softmax(original_log_pi, dim=-1).clamp(min=EPS)
        IS_ratio = (pi / orig_pi).detach()
        IS_ratio[gt_pdm_score['fail_mask']] = 1.
        clipped_IS_ratio = IS_ratio.clamp(max=10)

        reward_mask = (gt_pdm_score['score'] == 1).float()

        probs = pi
        pos_max = (probs * reward_mask).max(dim=-1).values
        neg_max = (probs * (1. - reward_mask)).max(dim=-1).values
        margin = 0.2
        ranking_loss = torch.clamp(neg_max + margin - pos_max, min=0)

        log_probs = F.log_softmax(log_pi, dim=-1)
        RL_loss = -(clipped_IS_ratio * gt_pdm_score['score'] * log_probs).mean()

        entropy = -(pi * log_probs).sum(dim=-1).mean()
        entropy_loss = -entropy

        ret_dict =  {
            'loss.RL.rank': self.rl_loss_weight['rank'] * torch.mean(ranking_loss),
            'loss.RL.PG': self.rl_loss_weight['PG'] * RL_loss,
            'loss.RL.entropy': self.rl_loss_weight['entropy'] * entropy_loss,
        }

        return ret_dict


    def IS_weight(self, result, loss, name='', target=None):
        if not self.importance_sampling:
            return loss

        if self.use_lora or self.full_finetuning:
            if name == 'imi':
                weight =  torch.exp(result[name].log_softmax(-1) - result['orig_'+name].log_softmax(-1)).detach()
            else:
                if self.orig_IL:
                    weight = torch.ones_like(result[name])
                else:
                    weight = torch.exp(result[name].clamp(min=1e-6).log() - result['orig_'+name].clamp(min=1e-6).log()).detach()

            weight[result['fail_mask']] = 1.

            clipped_weight = weight.clamp(0.01, 10)
            if name == 'imi':
                b = clipped_weight.shape[0]
                clipped_weight = clipped_weight[torch.arange(b), target]
                weight = weight[torch.arange(b), target]
            return loss * clipped_weight

        return loss

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

        if 'fail_mask' in gt_pdm_score.keys():
            soft_imi_label = torch.softmax(-dist.sum(-1),dim=-1)
            expert_score = gt_pdm_score['score'][torch.arange(imi_label.shape[0]), imi_label]
            # if the nearest expert label is not the correct one, we don't use the expert label
            imi_mask = (expert_score > 0.8).float()
            if self.hard_case_no_imi:
                # for hard cases (fail_mask == 1) and synthetic cases (fail_mask == -1), we don't use the expert label
                imi_mask[gt_pdm_score['fail_mask'] != 0] = 0
            else:
                # for synthetic cases (fail_mask == -1), we don't use the expert label
                imi_mask[gt_pdm_score['fail_mask'] < 0] = 0
            soft_imi_label = soft_imi_label * imi_mask[:, None]
        else:
            imi_mask = 1.

        if self.use_lora or self.full_finetuning:
            gt_pdm_score['fail_mask'] = gt_pdm_score['fail_mask'] != 0
            result['fail_mask'] = gt_pdm_score['fail_mask'].bool()

        if self.use_soft_imi:
            soft_imi_label = torch.softmax(-dist.sum(-1),dim=-1)
            loss_dict['loss.imi'] = torch.mean(self.IS_weight(result,imi_mask * F.cross_entropy(result['imi'], soft_imi_label,
                reduction='none'), 'imi', target=imi_label))
        else:
            loss_dict['loss.imi'] = torch.mean(self.IS_weight(result,imi_mask * F.cross_entropy(result['imi'], imi_label, label_smoothing=0.2,
                reduction='none'), 'imi', target=imi_label))
 
        # Add reward shaping loss
        if self.reward_shaping:
            loss_dict['loss.noc'] = 3*torch.mean(self.IS_weight(result, F.binary_cross_entropy(
                result['noc'], gt_pdm_score['no_at_fault_collisions'],reduction='none'
            ), 'noc'))

            loss_dict['loss.da'] = 3*torch.mean(self.IS_weight(result , F.binary_cross_entropy(
                result['da'], gt_pdm_score['drivable_area_compliance'],reduction='none'
            ), 'da'))

            loss_dict['loss.ttc'] = 2*torch.mean(self.IS_weight(result, F.binary_cross_entropy(
                result['ttc'], gt_pdm_score['time_to_collision_within_bound'],reduction='none'
            ),'ttc'))

            loss_dict['loss.comfort'] = torch.mean(self.IS_weight(result, F.binary_cross_entropy(
                result['comfort'], gt_pdm_score['comfort'],reduction='none'
            ),'comfort'))

            loss_dict['loss.progress'] = torch.mean(self.IS_weight(result,  F.binary_cross_entropy(
                result['progress'], gt_pdm_score['ego_progress'],reduction='none'
            ),'progress'))

        if self.rl_finetuning:
            loss_dict.update(
                self.compute_RL_loss(result, gt_pdm_score)
            )

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
        # print(gt_pre_command_sdc.shape, command.shape)
        # print(sdc_status.shape, sdc_planning_past.shape, sdc_planning_mask_past.shape)

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

        if self.use_lora:
            status_encoding = peft_wrapper_forward(enc_ego_status, self.status_embed, self.status_embed_lora)
        else:
            status_encoding = self.status_embed(enc_ego_status)

        sdc_planning_mask_past = sdc_planning_mask_past[:, 0, :, 0].float()
        b = sdc_planning_mask_past.shape[0]
        sdc_planning_mask_past = torch.cat([sdc_planning_mask_past, torch.zeros((b, 1)).to(status_encoding.device)],dim=1)
        sdc_planning_mask_past = sdc_planning_mask_past[:, :, None]

        status_encoding = torch.max(status_encoding * sdc_planning_mask_past, dim=1)[0]

        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_embed.shape[1]

        # bev_embed = bev_embed.detach()

        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )

        if self.use_lora:
            embedded_vocab = peft_wrapper_forward(vocab.view(1, L, -1).repeat(B, 1, 1), self.pos_embed, self.pos_embed_lora)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]

        if self.normalize_vocab_pos:
            embedded_vocab = self.encoder(embedded_vocab)

            
        embedded_vocab = embedded_vocab + status_encoding.unsqueeze(1)
        device = embedded_vocab.device

        if self.use_lora or self.full_finetuning:
            bev_embed = bev_embed.detach()

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
                if self.use_lora:
                    result[k] = peft_wrapper_forward(dist_status, head, self.heads_lora['lora_'+k]).squeeze(-1) 
                else:
                    result[k] = head(dist_status).squeeze(-1)
            else:
                if self.use_lora:
                    result[k] = peft_wrapper_forward(dist_status, head, self.heads_lora['lora_'+k]).squeeze(-1).sigmoid()
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

        # if (self.use_lora or self.full_finetuning) and eval==False:
        orig_results = self.forward_origin(
            bev_embed,
            command,
            sdc_planning_past, # 1 x 4 x 4
            sdc_status,
            sdc_planning_mask_past,  # 1 x 4 x 4
            gt_pre_command_sdc, #1*4
        )
        result.update(orig_results)

        return result


    @auto_fp16(apply_to=("bev_embed"))
    def forward_origin(
        self,
        bev_embed,
        command=None,
        sdc_planning_past=None, # 1 x 4 x 4
        sdc_status=None,
        sdc_planning_mask_past=None,  # 1 x 4 x 4
        gt_pre_command_sdc=None, #1*4
    ):  
        
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
            embedded_vocab = self.encoder(embedded_vocab,forward_origin=True).repeat(B, 1, 1)

        embedded_vocab = embedded_vocab + status_encoding.unsqueeze(1)
        device = embedded_vocab.device

        bev_embed = bev_embed.detach()

        dist_status = self.transformer(
            embedded_vocab, 
            bev_embed=bev_embed,
            reference_trajs=self.vocab.data.unsqueeze(0).expand(B, -1, -1, -1),
            spatial_shapes=torch.tensor([[200, 200]], device=device),
            level_start_index=torch.tensor([0], device=device),
            forward_origin=True,
        )

        result = {}
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
        result["selected_indices"] = scores.argmax(1)

        out_res = {}
        for k,v in result.items():
            out_res['orig_'+k] = v
        return out_res
        
