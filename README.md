# Regression for registration and model the uncertainty

## Two stages setting

### Stage 1: Regress the coordinates (and their uncertainty/distributions)

#### Option 1: Only regress the coordinates directly
```
python3 main.py --params params/regress_train.json

python3 inference.py --params params/regress_test.json
```
<!-- ### Option 2: Uncertainty training with only uncertainty loss and without dropout -->
#### Option 2: Regress the coordinates as well the uncertainty

##### Setting1: regress sigma for each channel separately (three sigmas)

```
python3 main_uncer.py --params params/regress_train.json ("mode": "pre")

python3 main_uncer.py --params params/regress_train.json ("mode": "training/finetune")

python3 inference_uncer.py --params params/regress_test.json
```

##### Setting2:  regress single sigma for three channel

```
python3 main_uncer_single_sigma.py --params params/regress_train.json ("mode": "pre")

python3 main_uncer_single_sigma.py --params params/regress_train.json ("mode": "training/finetune")

python3 inference_uncer_single_sigma.py --params params/regress_test.json
```
<!-- ### Option 3: Uncertainty training with dropout

    Setting1: line 10, line 30: nn.Dropout(p=0.2).

        python3 main_uncer.py --params params/regress_train_uncer_dropout.json

    Setting2: line 10, line 30: nn.Dropout(p=0.3).

    Setting3: line 10, line 30: nn.Dropout(p=0.4).

    Setting4: line 30: nn.Dropout(p=0.5).

    Setting5: line 30: nn.Dropout(p=0.2). -->

### Stage 2: Test-time fitting
```
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
```
## End2end

### Setting1: regress sigma for each channel separately (three sigmas)


```
python3 main_end2end.py --params params/regress_train.json ("mode": "pre")

python3 main_end2end.py --params params/regress_train.json ("mode": "training/finetune")
```

### Setting2: regress single sigma for three channel

```
python3 main_end2end_single_sigma.py --params params/regress_train.json ("mode": "pre")

python3 main_end2end_single_sigma.py --params params/regress_train.json ("mode": "training/finetune")
```