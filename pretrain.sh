#!/usr/bin/env bashclear

#### Synthetic to Real
## 7-class setting warmup
python train.py --name warmup_G2CI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

## 19-class setting warmup
python train.py --name warmup_G2CI_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

#### Real to Real
python train.py --name warmup_C2MI --src_dataset cityscapes --src_rootpath Dataset/Cityscapes --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_I2CM --src_dataset idd --src_rootpath Dataset/IDD_Segmentation --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_M2CI --src_dataset mapillary --src_rootpath Dataset/Mapillary --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

python train.py --name warmup_C2MI_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset cityscapes --src_rootpath Dataset/Cityscapes --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_I2CM_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_M2CI_19 --n_class 19 --img_size "1024,512" --resize 1024 --rcrop "512,256" --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume








#



