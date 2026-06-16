from typing import Dict, List, Optional, Tuple
import torch
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from mmseg.structures import SegDataSample

from mmengine import Config

from mmengine.structures import PixelData
import os.path as osp

@MODELS.register_module()
class MultiTaskDistillator_PerTask(BaseSegmentor):
    """MMSegmentation标准格式的多任务蒸馏模型"""
    
    def __init__(self,
                 # 学生模型配置
                 backbone: dict,
                 seg_head: dict,
                 edge_head: dict,
                 clf_head: dict,

                 # 教师模型检查点路径
                 teacher_checkpoint: str,
                 #教师模型配置
                 teacher_config_path:str,
                 # 蒸馏损失权重配置
                 distill_cfg: dict = None,
                 #学生模型预训练
                 #模型类型
                 task_type='seg',
                 multi_task=False,
                 # 数据预处理相关
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.multi_task=multi_task
        # ------------------ 构建学生网络 ------------------
        self.backbone = MODELS.build(backbone)
        self.student_seg_head = MODELS.build(seg_head)
        self.student_edge_head = MODELS.build(edge_head)
        self.student_clf_head = MODELS.build(clf_head)
        self.task_type=task_type
        # ------------------ 构建并加载教师网络 ------------------
        # 注意：教师模型结构与配置需在外部定义并通过此处传入
        # 这里假设教师模型配置已全局注册，通过MODELS.build构建
        self.teacher = self._load_teacher(teacher_checkpoint, teacher_config_path)
        
        # ------------------ 蒸馏配置 ------------------
        self.distill_cfg = distill_cfg or {}
        self.feat_weight = self.distill_cfg.get('feat_weight', [0.0,0.0])
        self.output_weight = self.distill_cfg.get('output_weight', 0.0)
        self.temperature = self.distill_cfg.get('temperature', 1.0)
        self.hard_weight = self.distill_cfg.get('hard_weight', 1.0)
        self.seg_decoder_weight = self.distill_cfg.get('seg_weight',(1.0,0.125))
        self.clf_decoder_weight = self.distill_cfg.get('clf_weight',(1.0,0.125))
        self.edge_decoder_weight = self.distill_cfg.get('edge_weight',(1.0,0.125))
        #--------------------冻结非当前任务头-------------
        self._freeze_inactive_heads()
        #-----------------是否执行多任务，只有mode=tensor时可用-------------


    def Load_Stu(self, pretrained, strict=True):
        """简化的加载函数"""
        # 加载checkpoint
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        print(f"加载checkpoint: {pretrained}")
        
        # 定义组件映射（根据你的checkpoint实际键名调整）
        component_configs = [
            # (组件对象, checkpoint前缀, 模型前缀)
            (self.backbone, 'backbone', 'backbone'),
            (self.student_seg_head, 'student_seg_head', 'student_seg_head'),
            (self.student_edge_head, 'student_edge_head', 'student_edge_head'),
            (self.student_clf_head, 'student_clf_head', 'student_clf_head'),
        ]
        return self.Load_Components(component_configs,state_dict,strict)
    
    def Load_Components(self,component_configs,state_dict,strict=True):
        # 用于跟踪进度
        success_count = 0
        total_count = len(component_configs)
        for component, checkpoint_prefix, model_prefix in component_configs:
            # 收集该组件的参数
            component_dict = {}
            for key, value in state_dict.items():
                # 检查是否以checkpoint_prefix开头
                if key.startswith(f"{checkpoint_prefix}."):
                    # 移除前缀
                    new_key = key[len(checkpoint_prefix) + 1:]
                    component_dict[new_key] = value
            if component_dict:
                try:
                    # 尝试加载
                    component.load_state_dict(component_dict, strict=strict)
                    print(f"✓ 成功加载 {model_prefix}: {len(component_dict)} 个参数")
                    success_count += 1
                except Exception as e:
                    print(f"✗ 加载 {model_prefix} 失败: {e}")
            else:
                print(f"⚠ 未找到 {model_prefix} 的参数（前缀: {checkpoint_prefix}）")
        
        print(f"\n加载完成: {success_count}/{total_count} 个组件")
        return success_count == total_count
    
    def Load_Multi_Stu(self,pretrained_dicts,strict=True):
        for key,value in pretrained_dicts.items():
            if key in ('seg','clf','edge','backbone'):
                checkpoint = torch.load(value, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                if key == 'backbone':
                    component_configs = [(self.backbone, 'backbone', 'backbone'),]
                elif key == 'clf':
                    component_configs = [(self.student_clf_head, 'student_clf_head', 'student_clf_head'),]
                elif key == 'seg':
                    component_configs = [(self.student_seg_head, 'student_seg_head', 'student_seg_head'),]
                else:
                    component_configs = [(self.student_edge_head, 'student_edge_head', 'student_edge_head'),]
                self.Load_Components(component_configs,state_dict,strict=strict)

    def Frezze_backbone(self,):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _load_teacher(self, checkpoint_path: str, teacher_config_path):
        """加载并冻结教师模型"""
        # 需要根据你的教师模型配置来构建
        # 这里是一个示例，你需要根据实际情况调整
        if osp.exists(teacher_config_path):
            teacher_cfg = Config.fromfile(teacher_config_path)
            teacher = MODELS.build(teacher_cfg.model)
            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                teacher.load_state_dict(checkpoint['state_dict'])
            else:
                teacher.load_state_dict(checkpoint)
            
            # 冻结所有参数
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            
            return teacher
        else:
            return None
    
    def extract_features(self, img, model):
        """统一特征提取接口，处理不同结构的模型"""
        features = model.extract_feat(img)
        return features
    
    def _freeze_inactive_heads(self):
        """冻结非活动任务头"""
        if not self.multi_task:
            if not self.task_type == 'clf':
                self.freeze_component(self.student_clf_head)
            if not self.task_type=='seg':
                self.freeze_component(self.student_seg_head)
            if not self.task_type == 'edge':
                self.freeze_component(self.student_edge_head)

    def freeze_component(self,model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    def activate_component(self,model):
        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def _activate_heads(self,):
        if self.task_type=='clf':
            self.activate_component(self.student_clf_head)
        elif self.task_type=='seg':
            self.activate_component(self.student_seg_head)
        elif self.task_type == 'edge':
            self.activate_component(self.student_edge_head)

    def change_task(self,task):
        self.task_type=task
        self._freeze_inactive_heads()
        self._activate_heads()

    
    def extract_feat(self, img):
        """统一特征提取接口，处理不同结构的模型"""
        return self.backbone(img)
    
    def encode_decode(self, inputs, batch_data_samples,**kwargs):
        task_type=self.task_type
        feat = self.extract_feat(inputs)
        if self.teacher is not None:
            tea_feat = self.teacher.extract_feat(inputs)
            tea_logits = self.teacher.decode_head(tea_feat)
        else:
            tea_feat = None
            tea_logits = None
        if task_type == 'seg':
            seg_logits = self.student_seg_head(feat)
        elif task_type == 'edge':
            seg_logits = self.student_edge_head(feat)
        elif task_type == 'clf':
            seg_logits = self.student_clf_head(feat,**kwargs)
        return seg_logits,tea_logits,feat,tea_feat
        
    def _forward(self, inputs, data_samples = None):
        feat = self.extract_feat(inputs)
        if not self.multi_task:
            if self.task_type == 'seg':
                return self.student_seg_head(feat)
            elif self.task_type == 'edge':
                return self.student_edge_head(feat)
            elif self.task_type == 'clf':
                return self.student_clf_head(feat)
        else:
            # feat_s,feat_t = feat
            # for s in feat_t:
            #     print(s.shape)
            seg = self.student_seg_head(feat)
            clf = self.student_clf_head(feat)
            edge = self.student_edge_head(feat)
            return seg,clf,edge
        
    def _temperature_scale(self, prob,eps=1e-7):
        """对概率应用温度缩放"""
        if self.temperature == 1.0:
            return prob
        # 避免0和1
        prob = prob.clamp(eps, 1 - eps)
        # 将概率转换为logits
        logits = torch.log(prob / (1 - prob))
        # 应用温度缩放
        scaled_logits = logits / self.temperature
        # scaled_logits_1 = 1-logits/self.temperature
        # 转换回概率
        scaled_prob = torch.sigmoid(scaled_logits)
        return scaled_prob

    def generate_edge_weight(self,pred,threshold=0.3):
        B,C,H,W = pred.shape
        pred_edge = (pred>threshold).float()
        pred_back = 1.0-pred_edge
        sum_edge = torch.sum(pred_edge,dim=(1,2,3))
        sum_back = torch.sum(pred_back,dim=(1,2,3))
        edge_weight = sum_edge/(sum_edge+sum_back+1e-6)
        back_weight = 1-edge_weight
        edge_weight = edge_weight.reshape(B,1,1,1)*pred_edge
        back_weight = back_weight.reshape(B,1,1,1)*pred_back
        return edge_weight+back_weight
    
    def pearson_loss(self, student_logits, teacher_logits,reduction=True):
        # 中心化
        s_centered = student_logits - student_logits.mean(dim=1, keepdim=True)
        t_centered = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
        
        # 计算协方差和标准差
        cov = (s_centered * t_centered).sum(dim=1)
        std_s = torch.sqrt((s_centered ** 2).sum(dim=1) + 1e-8)
        std_t = torch.sqrt((t_centered ** 2).sum(dim=1) + 1e-8)
        
        # 皮尔逊相关系数 (范围[-1, 1])
        pearson = cov / (std_s * std_t)
        
        # 最大化相关系数 -> 最小化 (1 - pearson) 的均值
        if reduction:
            loss = 1 - pearson.mean()
        else:
            loss = 1-pearson
        return loss
    
    def cos_loss(self,stu,tea):
        cos_sim = F.cosine_similarity(stu,tea)
        cos_sim = 1-cos_sim
        return cos_sim.mean()

    def loss(self, img: torch.Tensor, data_samples: List[SegDataSample]) -> dict:
        """重写损失计算，实现多任务蒸馏"""
        # 获取当前任务类型（通过data_samples中的meta信息传递）
        # 你需要在自己的DataSample中存储任务类型信息
        task_type = self.task_type
        
        losses = {}
        
        # 1. 获取教师目标
        with torch.no_grad():
            tea_features = self.teacher.extract_feat(img)
        # 2. 学生前向
        stu_features = self.extract_feat(img)
        # 3. 计算特征蒸馏损失
        if self.feat_weight[0] > 0:
            tea_feat_s,tea_feat_t = tea_features
            stu_feat_s,stu_feat_t = stu_features
            feat_loss_s = 0
            feat_loss_t = 0
            # 对齐特征层（可能需要适配不同模型的特征维度）
            min_layers = min(len(stu_feat_s), len(tea_feat_s))
            for i in range(min_layers):
                feat_loss_s += F.mse_loss(stu_feat_s[i], tea_feat_s[i])
                feat_loss_t += F.mse_loss(stu_feat_t[i], tea_feat_t[i])
            losses['loss_feat_distill'] = self.feat_weight[0] * feat_loss_s / min_layers 
            losses['loss_feat_distill']+=self.feat_weight[1] * feat_loss_t / min_layers
        # 学生任务输出
        if task_type == 'seg':
            stu_output = self.student_seg_head(stu_features)
        elif task_type == 'edge':
            stu_output = self.student_edge_head(stu_features)
        else:  # 'clf'
            stu_output = self.student_clf_head(stu_features)
        # 4. 计算输出蒸馏损失
        if self.output_weight > 0:
            tea_output = self.teacher.decode_head(tea_features)
            if task_type == 'clf':
                # 分类任务：KL散度蒸馏
                stu_log_softmax = F.log_softmax(stu_output / self.temperature, dim=1)
                tea_softmax = F.softmax(tea_output / self.temperature, dim=1)
                output_loss = F.kl_div(stu_log_softmax, tea_softmax, reduction='batchmean') * (self.temperature ** 2) 
                # cosine_sim = F.cosine_similarity(stu_output, tea_output, dim=1)
                # cos_loss = torch.mean(1 - cosine_sim)
                # output_loss += 0.1*cos_loss
                output_loss = self.clf_decoder_weight[0]*output_loss+self.clf_decoder_weight[1]*self.pearson_loss(stu_output,tea_output)
                # output_loss = F.kl_div(stu_log_softmax, tea_softmax, reduction='batchmean') * (self.temperature ** 2) + 0.05*self.pearson_loss(stu_output,tea_output)
                # output_loss = F.mse_loss(stu_output,tea_output)
            else:
                # 分割/边缘任务：MSE或KL散度
                # 软化处理
                if task_type=='edge':
                    stu_out = stu_output[0]
                    tea_out = tea_output[0]
                    #生成权重
                    weight_map = 0.1 + 100 * tea_out
                    # kl = (tea_out-stu_out)**2
                    # weight_map = self.generate_edge_weight(tea_out,threshold=0.3)
                    tea_out_soft = self._temperature_scale(tea_out)
                    stu_out_soft = self._temperature_scale(stu_out)
                    kl = tea_out * torch.log(tea_out_soft / stu_out_soft)+((1 - tea_out_soft) * torch.log((1 - tea_out_soft) / (1 - stu_out_soft)))
                    kl = weight_map*(self.edge_decoder_weight[0]*kl+self.edge_decoder_weight[1]*self.pearson_loss(stu_out,tea_out,False))
                    output_loss = kl.view(kl.size(0), -1).mean(dim=1).mean()
                else:
                    stu_out = stu_output
                    tea_out = tea_output

                    B,C,H,W = stu_out.shape
                    stu_soft = F.log_softmax(stu_out / self.temperature, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, C)
                    tea_soft = F.softmax(tea_out / self.temperature, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, C)
                    output_loss = F.kl_div(stu_soft, tea_soft, reduction='batchmean') * (self.temperature ** 2)
                    output_loss = self.seg_decoder_weight[0]*output_loss+self.seg_decoder_weight[1]*self.pearson_loss(stu_out,tea_out)
            
            losses['loss_output_distill'] = self.output_weight * output_loss
        
        # 5. 可选的辅助任务损失（如果有真实标签）
        if self.hard_weight > 0:
            if task_type == 'seg':
            # 分割头（你的Seg Head）也应有自己的loss_by_feat
                task_losses = self.student_seg_head.loss_by_feat(stu_output, data_samples)
            elif task_type=='edge':
                task_losses = self.student_edge_head.loss_by_feat(stu_output, data_samples)
            else:
                task_losses = self.student_clf_head.loss_by_feat(stu_output, data_samples)
            
            # 按权重合并到总损失
            for loss_name, loss_value in task_losses.items():
                task_losses[loss_name] = self.hard_weight * loss_value
            losses.update(add_prefix(task_losses, 'decode'))
        # 你可以在这里添加与真实标签的交叉熵损失
        
        return losses
    
    def predict(self, inputs, data_samples):
        """预测函数，根据任务类型选择输出头"""
        seg_logits,tea_logits,feat,tea_feat = self.encode_decode(inputs,data_samples)
        #包装特征
        data_samples = self.postprocess_feature_result(feat,tea_feat,data_samples)
        #包装输出
        data_samples = self.postprocess_output_result(seg_logits,tea_logits,data_samples)
        # 包装预测结果
        data_samples = self.postprocess_result(seg_logits,data_samples)
        return data_samples
    
    def reshape_tuple_pytorch(self,tuple_input):
        a_list, b_list = tuple_input
        B = a_list[0].shape[0]
        N = len(a_list)
        
        # 将每个张量沿着批次维度拆分
        a_split = [torch.unbind(a_tensor, dim=0) for a_tensor in a_list]  # [N, B] 每个元素是 [C1, H, W]
        b_split = [torch.unbind(b_tensor, dim=0) for b_tensor in b_list]  # [N, B] 每个元素是 [C2, H, W]
        
        # 重新组织数据结构
        result = []
        for b_idx in range(B):
            a_batch = [a_split[i][b_idx] for i in range(N)]
            b_batch = [b_split[i][b_idx] for i in range(N)]
            result.append((a_batch, b_batch))
        
        return result
    
    def postprocess_feature_result(self,
                           stu_features: Tensor,
                           tea_features:Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        if tea_features is not None:
            tea_features = self.reshape_tuple_pytorch(tea_features)
            stu_features = self.reshape_tuple_pytorch(stu_features)
            for i in range(len(data_samples)):
                data_samples[i].set_data({'teacher_features':tea_features[i]})
                data_samples[i].set_data({'student_features':stu_features[i]})
        return data_samples
    
    def postprocess_output_result(self,
                           stu_output: Tensor,
                           tea_output:Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        if tea_output is not None:
            if isinstance(tea_output,list):
                tea_output = tea_output[0]
            if isinstance(stu_output,list):
                stu_output = stu_output[0]
            for i in range(len(data_samples)):
                data_samples[i].set_data({'teacher_output':tea_output[i]})
                data_samples[i].set_data({'student_output':stu_output[i]})
        return data_samples
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        task_type = self.task_type

        if task_type=='seg':
            return super().postprocess_result(seg_logits,data_samples)
        elif task_type=='clf':
            return self.postprocess_Clf(seg_logits,data_samples)
        else:
            return self.postprocess_Edge(seg_logits,data_samples)

    def postprocess_Clf(self,
                           seg_logits: dict,
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, C, = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :]

            i_seg_logits = i_seg_logits.squeeze(0)
            data_samples[i].set_data({
                'pred_clf_label':i_seg_logits})
        return data_samples
    
    def postprocess_Edge(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        seg_logits = seg_logits[0]
        batch_size, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))
            i_seg_logits = i_seg_logits.squeeze(0)
            data_samples[i].set_data({
                'pred_sem_seg':
                PixelData(**{'data': i_seg_logits})
            })

        return data_samples
