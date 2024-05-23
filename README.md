# Multi-level Collaborative Learning for Multi-Target Domain Adaptive Semantic Segmentation

```

### Datasets

By default, the datasets are put in ``<root_dir>/Dataset``. Simialr to [here](https://github.com/Mael-zys/CoaST), i.e. ``ln -s path_GTA5 <root_dir>/Dataset/GTA5``

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:

```bash
<root_dir>/Dataset/GTA5/                               % GTA dataset root
<root_dir>/Dataset/GTA5/images/                        % GTA images
<root_dir>/Dataset/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and ground-truths. The Cityscapes dataset directory should have this basic structure:

```bash
<root_dir>/Dataset/Cityscapes/                         % Cityscapes dataset root
<root_dir>/Dataset/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/Dataset/Cityscapes/leftImg8bit/train
<root_dir>/Dataset/Cityscapes/leftImg8bit/val
<root_dir>/Dataset/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/Dataset/Cityscapes/gtFine/train
<root_dir>/Dataset/Cityscapes/gtFine/val
...
```

* **Mapillary**: Please follow the instructions in [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) to download the images and validation ground-truths. The Mapillary Vistas dataset directory should have this basic structure:

```bash
<root_dir>/Dataset/Mapillary/                          % Mapillary dataset root
<root_dir>/Dataset/Mapillary/train                     % Mapillary train set
<root_dir>/Dataset/Mapillary/train/images
<root_dir>/Dataset/Mapillary/validation                % Mapillary validation set
<root_dir>/Dataset/Mapillary/validation/images
<root_dir>/Dataset/Mapillary/validation/labels
...
```

* **IDD**: Please follow the instructions in [IDD](https://idd.insaan.iiit.ac.in/) to download the images and validation ground-truths. The IDD Segmentation dataset directory should have this basic structure:

```bash
<root_dir>/Dataset/IDD_Segmentation/                         % IDD dataset root
<root_dir>/Dataset/IDD_Segmentation/leftImg8bit              % IDD images
<root_dir>/Dataset/IDD_Segmentation/leftImg8bit/train
<root_dir>/Dataset/IDD_Segmentation/leftImg8bit/val
<root_dir>/Dataset/IDD_Segmentation/gtFine                   % Semantic segmentation labels
<root_dir>/Dataset/IDD_Segmentation/gtFine/val
...
```

## step 1: Pre-trained models
## 7-class setting warmup
 ```bash
python train.py --name warmup_G2CI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
 ```
## Real to Real
 ```bash
python train.py --name warmup_C2MI --src_dataset cityscapes --src_rootpath Dataset/Cityscapes --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_I2CM --src_dataset idd --src_rootpath Dataset/IDD_Segmentation --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_M2CI --src_dataset mapillary --src_rootpath Dataset/Mapillary --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
 ```
## 19-class setting warmup
 ```
python train.py --name warmup_G2CI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes idd --tgt_rootpath Dataset/Cityscapes Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
 ```

## step 2: Pseudo-labels initialization and train models
















Due to my google drive limits, I can only upload the [checkpoints](https://drive.google.com/drive/folders/193DynhYYHxMCX7iAOY1Z5DU4b_4T4AHA?usp=sharing) for G2CM 7-classes and 19-classes settings.

## Evaluation

### 7-class setting

<details>
  <summary>
    <b>1. Synthetic to Real</b>
  </summary>

- **GTA5 $\rightarrow$ Cityscapes + IDD.**

  ```bash
  python test.py --bs 1 --stage stage1 --resume_path ./logs/stage1_G2CI/from_gta5_to_2_on_deeplabv2_best_model.pkl
  ```
- **GTA5 $\rightarrow$ Cityscapes + Mapillary.**

  ```bash
  python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --resume_path ./logs/stage1_G2CM/from_gta5_to_2_on_deeplabv2_best_model.pkl
  ```
- **GTA5 $\rightarrow$ Mapillary + IDD.**

  ```bash
  python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2MI/from_gta5_to_2_on_deeplabv2_best_model.pkl
  ```


## Acknowledgements

This codebase is heavily borrowed from [MTAF](https://github.com/valeoai/MTAF) and [ProDA](https://github.com/microsoft/ProDA).

