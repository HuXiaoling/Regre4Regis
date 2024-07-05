# Regression for registration

    python3 main.py --params params/regress_train.json

    python3 inference.py --params params/regress_test.json

Pre-training experimental results:

| Setting | regress loss weight (l1) | seg loss weight | uncer loss weight | seg|  x | y | z |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   8 channels  | 1 (0.0147)    | 0.5 (-0.7352) | ---   | &check;   | &check;   | &check;   | &check;   |


Uncertainty training with only uncertainty loss and without dropout:

    Setting1: no dropout

        Step1: regress coordinates and masks only 

            python3 main_uncer.py --params params/regress_train.json ("mode": "pre")
        
        Step2: regress sigma

            python3 main_uncer.py --params params/regress_train.json ("mode": "finetune")

            python3 inference_uncer.py --params params/regress_test.json


    Setting2: no dropout with single sigma

        Step1: regress coordinates and masks only

            python3 main_uncer_single_sigma.py --params params/regress_train.json ("mode": "pre")
        
        Step2: regress sigma

            python3 main_uncer_single_sigma.py --params params/regress_train.json ("mode": "finetune")

            python3 inference_uncer_single_sigma.py --params params/regress_test.json

Pre-training experimental results:

| Setting | regress loss weight (l2) | seg loss weight | uncer loss weight | seg|  x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0363)    | 0.01 (-0.6640)    | --- (3.7952)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 8 channels    | 1 (0.0410)    | 0.05 (-0.7070)    | --- (5.1415)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 8 channels    | 1 (0.0254)    | 0.1  (-0.7179)    | --- (2.0328)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 8 channels    | 1 (0.0421)    | 0.5  (-0.7223)    | --- (6.4939)  | &check;   | &check;   | &check;   | &check;   | Done*     |
| 6 channels    | 1 (0.0443)    | 0.01 (-0.6363)    | --- (5.3125)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 6 channels    | 1 (0.0493)    | 0.05 (-0.6992)    | --- (5.7710)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 6 channels    | 1 (0.0235)    | 0.1  (-0.7237)    | --- (1.1944)  | &check;   | &check;   | &check;   | &check;   | Done      |
| 6 channels    | 1 (0.0298)    | 0.5  (-0.7256)    | --- (1.4838)  | &check;   | &check;   | &check;   | &check;   | Done*     |

Finetune experimental results (loss weight = 0.5):

| Setting | regress loss weight (l2) | seg loss weight | uncer loss weight | seg|  x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.1105)    | 0.5 (-0.7242) | Gaussian  0.1 (0.2017)    | &check;   | &check;   | &check;   | &cross;   | Done (Fail)   |
| 8 channels    | 1 (0.1093)    | 0.5 (-0.7235) | Gaussian  0.5 (0.1888)    | &check;   | &check;   | &check;   | &cross;   | Done (Fail)   |
| 8 channels    | 1 (0.0283)    | 0.5 (-0.7141) | Laplacian 0.1 (1.1182)    | &check;   | &check;   | &check;   | &check;   | Done          |
| 8 channels    | 1 (0.0291)    | 0.5 (-0.7119) | Laplacian 0.5 (1.4062)    | &check;   | &check;   | &check;   | &check;   | Done          |
| 6 channels    | 1 (0.0382)    | 0.5 (-0.7295) | Gaussian  0.1 (0.0940)    | &check;   | &check;   | &check;   | &check;   | Done          |
| 6 channels    | 1 (0.0344)    | 0.5 (-0.7228) | Gaussian  0.5 (0.0740)    | &check;   | &check;   | &check;   | &check;   | Done          |
| 6 channels    | 1 (0.0322)    | 0.5 (-0.7291) | Laplacian 0.1 (0.6011)    | &check;   | &check;   | &check;   | &check;   | Done          |
| 6 channels    | 1 (0.0348)    | 0.5 (-0.7239) | Laplacian 0.5 (0.6633)    | &check;   | &check;   | &check;   | &check;   | Done          |

Finetune experimental results (loss weight = 0.5) by increasing uncertainty loss weight:

| Setting | regress loss weight (l2) | seg loss weight | uncer loss weight | seg|  x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.1088)    | 0.5 (-0.7209) | Gaussian   0.1  (2.9452)  | &check; | &check; | &check; | &cross; | Done (Fail)   |
| 8 channels    | 1 (0.1081)    | 0.5 (-0.7045) | Gaussian   0.5  (2.9571)  | &check; | &check; | &check; | &cross; | Done (Fail)   |
| 8 channels    | 1             | 0.5           | Laplacian  0.01           |         |         |         |         | Running       |
| 8 channels    | 1 (0.0244)    | 0.5 (-0.7107) | Laplacian  0.05 (1.2930)  | &check; | &check; | &check; | &check; | Done          |
| 8 channels    | 1 (0.0248)    | 0.5 (-0.6961) | Laplacian  0.1  (1.1233)  | &check; | &check; | &check; | &check; | Done          |
| 8 channels    | 1 (0.0227)    | 0.5 (-0.5428) | Laplacian  0.5  (0.8866)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1             | 0.5           | Gaussian   0.01           |         |         |         |         | Running       |
| 6 channels    | 1 (0.0279)    | 0.5 (-0.7230) | Gaussian   0.05 (0.9441)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1 (0.0262)    | 0.5 (-0.7177) | Gaussian   0.1  (0.8739)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1 (0.0262)    | 0.5 (-0.6996) | Gaussian   0.5  (0.8722)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1             | 0.5           | Laplacian  0.01           |         |         |         |         | Running       |
| 6 channels    | 1 (0.0243)    | 0.5 (-0.7212) | Laplacian  0.05 (0.9008)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1 (0.0246)    | 0.5 (-0.7174) | Laplacian  0.1  (0.9362)  | &check; | &check; | &check; | &check; | Done          |
| 6 channels    | 1 (0.0243)    | 0.5 (-0.6252) | Laplacian  0.5  (0.9320)  | &check; | &check; | &check; | &check; | Done          |

Finetune experimental results (loss weight = 0.1) by increasing uncertainty loss weight:

| Setting | regress loss weight (l2) | seg loss weight | uncer loss weight | seg|  x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1             | 0.1           | Gaussian   0.01           |  |  |  |  |           |
| 8 channels    | 1             | 0.1           | Gaussian   0.05           |  |  |  |  | Running   |
| 8 channels    | 1 (0.0387)    | 0.1 (-0.6493) | Gaussian   0.1  (2.3576)  |  |  |  |  | Running   |
| 8 channels    | 1             | 0.1           | Gaussian   0.5            |  |  |  |  |           |
| 8 channels    | 1             | 0.1           | Laplacian  0.01           |  |  |  |  |           |
| 8 channels    | 1             | 0.1           | Laplacian  0.05           |  |  |  |  |           |
| 8 channels    | 1 (0.0321)    | 0.1 (-0.6321) | Laplacian  0.1  (2.2276)  |  |  |  |  | Running   |
| 8 channels    | 1             | 0.1           | Laplacian  0.5            |  |  |  |  |           |
| 6 channels    | 1             | 0.1           | Gaussian   0.01           |  |  |  |  |           |
| 6 channels    | 1             | 0.1           | Gaussian   0.05           |  |  |  |  |           |
| 6 channels    | 1 (0.0387)    | 0.1 (-0.6905) | Gaussian   0.1  (1.4279)  |  |  |  |  | Running   |
| 6 channels    | 1             | 0.1           | Gaussian   0.5            |  |  |  |  |           |
| 6 channels    | 1             | 0.1           | Laplacian  0.01           |  |  |  |  |           |
| 6 channels    | 1             | 0.1           | Laplacian  0.05           |  |  |  |  |           |
| 6 channels    | 1 (0.1246)    | 0.1 (-0.7171) | Laplacian  0.1 (5.2980)   |  |  |  |  | Running   |
| 6 channels    | 1             | 0.1           | Laplacian  0.5            |  |  |  |  |           |

Uncertainty training with dropout:

    Setting1: line 10, line 30: nn.Dropout(p=0.2).

        python3 main_uncer.py --params params/regress_train_uncer_dropout.json

    Setting2: line 10, line 30: nn.Dropout(p=0.3).

    Setting3: line 10, line 30: nn.Dropout(p=0.4).

    Setting4: line 30: nn.Dropout(p=0.5).

    Setting5: line 30: nn.Dropout(p=0.2).

Stage2: 
    
    CNN part: 13.53 seconds;

    python ./scripts/test_inference_only_reg_many_models_ours.py 
        --input $INP 
        --input_seg $SUB 
        --mni_img ../proj_keymorph_atlas/mni.nii.gz 
        --mni_seg ../proj_keymorph_atlas/mni.seg.nii.gz 
        --grid_seg_x ./mni.grid.x.nii.gz 
        --grid_seg_y ./mni.grid.y.nii.gz 
        --grid_seg_z ./mni.grid.z.nii.gz 
        --output_dir FOL/ 
        --checkpoint_restore experiments/regress/train_outputs_full_aug_yogurt_2/model_best.pth
