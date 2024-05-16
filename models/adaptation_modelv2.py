import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplabv2 import Deeplab
from models.discriminator import FCDiscriminator
from .utils import freeze_bn, get_scheduler, cross_entropy2d
from data.randaugment import affine_sample
from color import color_transfer


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class CustomModel():

    def __init__(self, opt, logger, isTrain=True):

        self.opt = opt
        self.class_numbers = opt.n_class
        self.logger = logger
        self.best_iou = -100
        self.nets = []
        self.nets_DP = []
        self.default_gpu = 0  ########################## 
        self.num_target = len(opt.tgt_dataset_list)
        self.domain_id = -1
        if opt.bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif opt.bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(opt.bn))
        if self.opt.no_resume:
            restore_from = None
        else:
            restore_from = opt.resume_path
        self.best_iou = 0
        if self.opt.stage == 'stage1' and opt.norepeat == False:
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, num_target=len(opt.tgt_dataset_list) + 2,
                                   freeze_bn=False, restore_from=restore_from, stage=self.opt.stage)
        else:
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, num_target=len(opt.tgt_dataset_list),
                                   freeze_bn=False, restore_from=restore_from, stage=self.opt.stage)
        logger.info('the backbone is {}'.format(opt.model_name))
        self.nets.extend([self.BaseNet])
        self.optimizers = []
        self.schedulers = []
        optimizer_cls = torch.optim.SGD
        optimizer_params = {'lr': opt.lr, 'weight_decay': 1e-4, 'momentum': 0.9}  ##2
        if self.opt.stage == 'warm_up':
            self.net_D_list = []
            self.net_D_DP_list = []
            self.optimizer_D_list = []
            self.DSchedule_list = []
            for i_target in range(self.num_target):
                net_D = FCDiscriminator(inplanes=self.class_numbers)
                net_D_DP = self.init_device(net_D, gpu_id=self.default_gpu, whether_DP=False)
                optimizer_D = torch.optim.Adam(net_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
                self.nets.extend([net_D])
                self.nets_DP.append(net_D_DP)
                self.net_D_list.append(net_D)
                self.net_D_DP_list.append(net_D_DP)
                self.optimizers.extend([optimizer_D])
                self.optimizer_D_list.append(optimizer_D)
                DSchedule = get_scheduler(optimizer_D, opt)
                self.schedulers.extend([DSchedule])
                self.DSchedule_list.append(DSchedule)
        if self.opt.stage == 'warm_up':
            self.BaseOpti = optimizer_cls([{'params': self.BaseNet.get_1x_lr_params(), 'lr': optimizer_params['lr']},
                                           {'params': self.BaseNet.get_10x_lr_params(),
                                            'lr': optimizer_params['lr'] * 10}], **optimizer_params)
        else:
            self.BaseOpti = optimizer_cls(
                [{'params': self.BaseNet.get_1x_lr_params_new(), 'lr': optimizer_params['lr']},
                 {'params': self.BaseNet.get_10x_lr_params_new(), 'lr': optimizer_params['lr'] * 10}],
                **optimizer_params)
        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=False)
        self.nets_DP.append(self.BaseNet_DP)
        self.optimizers.extend([self.BaseOpti])
        self.BaseSchedule = get_scheduler(self.BaseOpti, opt)
        self.schedulers.extend([self.BaseSchedule])
        self.adv_source_label = 0
        self.adv_target_label = 1
        if self.opt.gan == 'Vanilla':
            self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        elif self.opt.gan == 'LS':
            self.bceloss = torch.nn.MSELoss()

    def step_adv(self, source_data, data_target_list, device):

        for net_D in self.net_D_list:
            for param in net_D.parameters():
                param.requires_grad = False
        self.BaseOpti.zero_grad()
        source_x = source_data['img'].to(device)
        source_label = source_data['label'].to(device)
        domain_list = [i for i in range(self.num_target)]
        source_output_list = self.BaseNet_DP(source_x, domain_list, ssl=[])
        source_outputUp_list = []
        loss_GTA = 0
        for i in range(self.num_target):
            source_output = source_output_list[i]
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear',
                                            align_corners=True)
            source_outputUp_list.append(source_outputUp)
            loss_GTA += cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')
        loss_GTA.backward()

        target_outputUp_list = []
        for i_target in range(self.num_target):
            target_x = data_target_list[i_target]['img'].to(device)
            target_output_list = self.BaseNet_DP(target_x, [i_target], ssl=[])
            target_output = target_output_list[0]
            target_outputUp = F.interpolate(target_output['out'], size=target_x.size()[2:], mode='bilinear',
                                            align_corners=True)
            target_outputUp_list.append(target_outputUp)
            # adv
            target_D_out = self.net_D_DP_list[i_target](prob_2_entropy(F.softmax(target_outputUp, dim=1)))
            loss_adv_G = self.bceloss(target_D_out,
                                      torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_source_label).to(
                                          target_D_out.device)) * self.opt.adv
            loss_adv_G.backward()
        self.BaseOpti.step()
        for net_D in self.net_D_list:
            for param in net_D.parameters():
                param.requires_grad = True
        for optimizer_D in self.optimizer_D_list:
            optimizer_D.zero_grad()
        for i_target in range(self.num_target):
            source_D_out = self.net_D_DP_list[i_target](
                prob_2_entropy(F.softmax(source_outputUp_list[i_target].detach(), dim=1)))
            target_D_out = self.net_D_DP_list[i_target](
                prob_2_entropy(F.softmax(target_outputUp_list[i_target].detach(), dim=1)))
            loss_D = self.bceloss(source_D_out,
                                  torch.FloatTensor(source_D_out.data.size()).fill_(self.adv_source_label).to(
                                      source_D_out.device)) + \
                     self.bceloss(target_D_out,
                                  torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_target_label).to(
                                      target_D_out.device))
            loss_D.backward()
        for optimizer_D in self.optimizer_D_list:
            optimizer_D.step()
        return loss_GTA.item(), loss_adv_G.item(), loss_D.item()

    def step(self, source_data, data_target_list, device):

        source_x = source_data['img'].to(device)
        source_label = source_data['label'].to(device)
        source_outputUp_list = []

        ############# train for source
        transfered_source_x_list = []
        for i in range(self.num_target):
            data_i = data_target_list[i]
            target_image = data_i['img'].to(device)
            ### source image with i domain style
            transfered_source_x = self.image_style(source_x, target_image).to(device)
            transfered_source_x_list.append(transfered_source_x)
        transfered_source_x_list.append(source_x)
        transfered_source_x_all = torch.concat(transfered_source_x_list[:-1], dim=0)

        loss_seg = 0
        feat_list = []
        for i in range(self.num_target + 2):
            if i != (self.num_target + 1):
                source_output_i = self.BaseNet_DP(transfered_source_x_list[i], [i], feat_list)
            else:
                source_output_i = self.BaseNet_DP(transfered_source_x_all, [self.num_target + 1], feat_list,
                                                  ensembel=True)
            source_output_i = source_output_i[0]
            source_outputUp = F.interpolate(source_output_i['out'], size=source_x.size()[2:], mode='bilinear',
                                            align_corners=True)
            source_outputUp_list.append(source_outputUp)
            loss_seg += cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')
        loss_seg.backward()

        ###################################################### train for target
        for i in range(self.num_target):
            self.domain_id = i
            data_i = data_target_list[i]
            target_imageS = data_i['img_strong'].to(device)
            target_params = data_i['params']
            target_lpsoft = data_i['lpsoft'].to(device) if 'lpsoft' in data_i.keys() else None
            threshold_arg = F.interpolate(target_lpsoft, scale_factor=0.25, mode='bilinear', align_corners=True)

            # ### Pseudo-label assignment---one
            # rectified = threshold_arg
            # threshold_arg = rectified.max(1, keepdim=True)[1]  ## pseudo-label
            # rectified = rectified / rectified.sum(1, keepdim=True)
            # argmax = rectified.max(1, keepdim=True)[0]  ## confidence score
            # threshold_arg[argmax < self.opt.train_thred] = 250
            # ## threshold_arg[argmax < 0.5] = 250  ## -1

            ### Pseudo-label assignment:  entropy-guided pseudo-label -- sencond
            _, class_num, _, _ = threshold_arg.size()
            predicted_entropy = prob_2_entropy(threshold_arg).sum(dim=1)
            rectified = threshold_arg
            threshold_arg = rectified.max(1, keepdim=True)[1]  ## pseudo-label
            thres = []
            for cs in range(class_num):
                x = predicted_entropy[threshold_arg.squeeze() == cs]
                if len(x) == 0:
                    thres.append(0)
                    continue
                x = np.sort(x.cpu().numpy())
                thres.append(x[np.int(np.round(len(x) * 0.5))])
            thres = np.array(thres)
            masks = torch.zeros(threshold_arg.size()).to(device).bool()
            for cl in range(class_num):
                mask1 = (threshold_arg == cl)
                mask2 = (predicted_entropy < np.maximum(0.25,thres[cl])).unsqueeze(dim=1) ## 0.2, 0.1，0.3，0.15
                # mask2 = (predicted_entropy < thres[cl]).unsqueeze(dim=1)  ## 0.2, 0.1，0.3，0.15
                mask = mask1*mask2
                masks = masks | mask
            threshold_arg[~masks] = 250
            #####################
            batch, _, w, h = threshold_arg.shape
            ### forward
            feat_list_t = []
            target_out_list = self.BaseNet_DP(target_imageS, [i, self.num_target], feat_list_t)
            target_out = target_out_list[0]
            targetS_out_agnostic = target_out_list[1]
            target_out['out'] = F.interpolate(target_out['out'], size=threshold_arg.shape[2:], mode='bilinear',
                                              align_corners=True)
            targetS_out_agnostic['out'] = F.interpolate(targetS_out_agnostic['out'], size=threshold_arg.shape[2:],
                                                        mode='bilinear', align_corners=True)
            threshold_argS = self.label_strong_T(threshold_arg.clone().float(),
                                                 target_params, padding=250, scale=4).to(torch.int64)
            threshold_arg = threshold_argS  ## strong pseudo-label
            maskS = (threshold_arg != 250).float()

            ### style transfer
            loss = torch.Tensor([0]).to(self.default_gpu)
            loss_CTS_transfered_all = 0
            weights_all = 0
            JS_Divergence = 0
            variance_all = 0
            i_domain_transfer_list = []
            i_domain_transfer_list.append(target_imageS)
            for ii in range(self.num_target):
                if ii == i:
                    continue
                transfered_domain = ii
                data_transfered_domain = data_target_list[transfered_domain]
                data_transfered_domain_imageS = data_transfered_domain['img_strong'].to(device)
                transfered = self.image_style(target_imageS, data_transfered_domain_imageS).to(device)
                i_domain_transfer_list.append(transfered)
                ## ii domain classifier
                target_out_transfer_list = self.BaseNet_DP(transfered, [transfered_domain], feat_list_t)
                target_out_transfer = target_out_transfer_list[0]  ## ii domain classifier output
                target_out_transfer['out'] = F.interpolate(target_out_transfer['out'], size=threshold_arg.shape[2:],
                                                           mode='bilinear', align_corners=True)
                ## calculate predictive variance
                # JS_Divergence1 = torch.sum(self.JS_Divergence_With_Temperature(target_out_transfer['out'].detach(),
                #                                                                target_out['out'].detach(), 2.5), dim=1)
                # JS_Divergence += JS_Divergence1
                # exp_JS_Divergence = torch.exp(-JS_Divergence)
                # weights_all += exp_JS_Divergence.detach()

                ## calculate predictive variance
                variance1 = torch.sum(F.kl_div(F.log_softmax(target_out_transfer['out'], dim=1),
                                               F.softmax(target_out['out'].detach(), dim=1), reduction='none'), dim=1)
                variance_all += variance1
                exp_variance1 = torch.exp(-self.opt.gamma * variance1)
                weights_all += exp_variance1.detach()

                ## 2、Cooperative-self training
                loss_CTS_transfered = cross_entropy2d(input=target_out_transfer['out'],
                                                      target=threshold_arg.reshape([batch, w, h]).detach(),
                                                      reduction='none')
                loss_CTS_transfered_all += loss_CTS_transfered
            weights_all /= self.num_target - 1

            #### 3、Ensemble-self training
            transfered_target_x_all = torch.concat(i_domain_transfer_list, dim=0)
            transfered_target_x_all_list = self.BaseNet_DP(transfered_target_x_all, [self.num_target + 1], feat_list_t,
                                                           target_ensembel=True, ensembel=True)
            ensemble_target_output = transfered_target_x_all_list[0]
            ensemble_target_output['out'] = F.interpolate(ensemble_target_output['out'], size=threshold_arg.shape[2:],
                                                          mode='bilinear', align_corners=True)
            loss_ensemble_all = cross_entropy2d(input=ensemble_target_output['out'],
                                                target=threshold_arg.reshape([batch, w, h]).detach(), reduction='none')

            #### 4、naive self-training
            loss_CTS_all = cross_entropy2d(input=target_out['out'],
                                           target=threshold_arg.reshape([batch, w, h]).detach(), reduction='none')

            #### re-weighting slef-training
            if self.opt.rectify:
                loss_ensemble_all = (loss_ensemble_all * weights_all * maskS).sum() / maskS.sum()
                loss_CTS_transfered_all = ((loss_CTS_transfered_all * weights_all * maskS).sum() / maskS.sum()) / (
                        self.num_target - 1)
                loss_CTS_all = (loss_CTS_all * weights_all * maskS).sum() / maskS.sum()
            else:
                loss_ensemble_all = (loss_ensemble_all * maskS).sum() / maskS.sum()
                loss_CTS_transfered_all = ((loss_CTS_transfered_all * maskS).sum() / maskS.sum()) / (
                    self.num_target - 1)
                loss_CTS_all = (loss_CTS_all * maskS).sum() / maskS.sum()

            ### 5、Hierarchical online knowledge distillation
            student = F.log_softmax(targetS_out_agnostic['out'], dim=1)
            teacher = F.softmax(target_out['out'].detach(), dim=1)
            teacher_copy = F.log_softmax(target_out['out'], dim=1)
            ensemble_teacher = F.softmax(ensemble_target_output['out'].detach(), dim=1)
            loss_kd = F.kl_div(student, teacher, reduction='none') + F.kl_div(teacher_copy, ensemble_teacher,reduction='none')
            # loss_kd = F.kl_div(student, teacher, reduction='none') + F.kl_div(student, ensemble_teacher, reduction='none')

            mask = (teacher != 250).float()
            loss_kd = (loss_kd * mask).sum() / mask.sum()
            loss_kd /= self.num_target
            #### total loss
            # loss += loss_kd + loss_CTS_all + self.opt.ratio * loss_CTS_transfered_all #+ 0.1 * loss_ensemble_all)
            # loss += loss_kd + loss_CTS_all + (self.opt.ratio * loss_CTS_transfered_all + 0.1 * loss_ensemble_all)
            loss += loss_kd + loss_CTS_all  + (self.opt.ratio * loss_CTS_transfered_all + 0.05*loss_ensemble_all) ## 0.01

            loss.backward()

        self.BaseOpti.step()
        self.BaseOpti.zero_grad()
        return loss_seg.item(), loss.item(), loss_CTS_all.item(), loss_kd.item(), loss_CTS_transfered_all.item(), loss_ensemble_all.item()

    def entropy(self, pred, reduction='none'):

        pred = F.softmax(pred, dim=1)
        epsilon = 1e-5
        H = -pred * torch.log(pred + epsilon)
        H = H.sum(dim=1)
        if reduction == 'mean':
            return H.mean()
        else:
            return H

    def JS_Divergence_With_Temperature(self, p, q, temp_factor, get_softmax=True):
        KLDivLoss = nn.KLDivLoss(reduction='none')
        if get_softmax:
            p_softmax_output = F.softmax(p / temp_factor)
            q_softmax_output = F.softmax(q / temp_factor)
        log_mean_softmax_output = ((p_softmax_output + q_softmax_output) / 2).log()
        return (KLDivLoss(log_mean_softmax_output, p_softmax_output) + KLDivLoss(log_mean_softmax_output, q_softmax_output)) / 2

    def image_style(self, source_x, target_imageS):
        img_transfer = []
        for img_x, img_t in zip(source_x, target_imageS):
            img1 = np.array(img_x.cpu(), dtype=np.float64).transpose(1, 2, 0)[:, :, ::-1] * 255.0  ## BGR
            img2 = np.array(img_t.cpu(), dtype=np.float64).transpose(1, 2, 0)[:, :, ::-1] * 255.0  ## BGR
            transfer = color_transfer(img2.astype(np.uint8), img1.astype(np.uint8))
            transfer_RGB = transfer[:, :, ::-1].transpose(2, 0, 1).astype(float) / 255.0
            img_transfer.append(torch.from_numpy(transfer_RGB).float())
        return torch.stack(img_transfer)

    def label_strong_T(self, label, params, padding, scale=1):
        label = label + 1
        for i in range(label.shape[0]):
            for (Tform, param) in params.items():
                if Tform == 'Hflip' and param[i].item() == 1:
                    label[i] = label[i].clone().flip(-1)
                elif (
                        Tform == 'ShearX' or Tform == 'ShearY' or Tform == 'TranslateX' or Tform == 'TranslateY' or Tform == 'Rotate') and \
                        param[i].item() != 1e4:
                    v = int(param[i].item() // scale) if Tform == 'TranslateX' or Tform == 'TranslateY' else param[
                        i].item()
                    label[i:i + 1] = affine_sample(label[i:i + 1].clone(), v, Tform)
                elif Tform == 'CutoutAbs' and isinstance(param, list):
                    x0 = int(param[0][i].item() // scale)
                    y0 = int(param[1][i].item() // scale)
                    x1 = int(param[2][i].item() // scale)
                    y1 = int(param[3][i].item() // scale)
                    label[i, :, y0:y1, x0:x1] = 0
        label[label == 0] = padding + 1  # for strong augmentation, constant padding
        label = label - 1
        return label

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            # net = DataParallelWithCallback(net, device_ids=[0])
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net

    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger != None:
                logger.info("Successfully set the model eval mode")
        else:
            net.eval()
            if logger != None:
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net == None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
        else:
            net.train()
        return