#!/usr/bin/env bashclear

#### Synthetic to Real
### 7-class setting warmup
#python train.py --name warmup_G2CI --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

### real to Real
#python train.py --name warmup_C2MI --src_dataset cityscapes --src_rootpath Dataset/Cityscapes --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_I2CM --src_dataset idd --src_rootpath Dataset/IDD_Segmentation --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_M2CI --src_dataset mapillary --src_rootpath Dataset/Mapillary --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

 
#####real to Real
#python train.py --name stage1_C2MI --src_dataset cityscapes --src_rootpath Dataset/Cityscapes --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_C2MI/from_cityscapes_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_I2CM --src_dataset idd --src_rootpath Dataset/IDD_Segmentation --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --used_save_pseudo --rectify --resume_path ./logs/warmup_I2CM/from_idd_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_M2CI --src_dataset mapillary --src_rootpath Dataset/Mapillary --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_M2CI/from_mapillary_to_2_on_deeplabv2_best_model.pkl

#### 19-class setting warmup
#python train.py --name warmup_G2CI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2CM_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2MI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name warmup_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#

### 7-class setting stage1: self-training
#python train.py --name stage1_G2CI --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CI/from_gta5_to_2_on_deeplabv2_best_model.pkl

#python train.py --name stage1_G2CM_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CM_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2MI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2MI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CM/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2MI/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2CI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI/from_gta5_to_3_on_deeplabv2_best_model.pkl
### re-training
#python train.py --name warmup_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
#python train.py --name stage1_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl
#python train.py --name stage1_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl




#python train.py --name stage1_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/stage1_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl
##'/disks/disk0/feifei/paper/paper4/MST/Code/MST/logs/stage1_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl'
#






#### evaluate 7 class
#python test.py --bs 1 --stage stage1 --resume_path ./logs/stage1_G2CI/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --resume_path ./logs/stage1_G2CM/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2MI/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2CMI/from_gta5_to_3_on_deeplabv2_best_model.pkl
#
#### evaluate 19 class
#python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --resume_path ./logs/stage1_G2CI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --resume_path ./logs/stage1_G2CM_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2MI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
#python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl
python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/warmup_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl

#



