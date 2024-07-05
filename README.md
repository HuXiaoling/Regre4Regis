# Regression for registration

## Stage 1: regress the coordinates (and their uncertainty/distributions)

### Pre-training without uncertainty branch:

    python3 main.py --params params/regress_train.json

    python3 inference.py --params params/regress_test.json

### Uncertainty training with only uncertainty loss and without dropout:

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

### Uncertainty training with dropout:

    Setting1: line 10, line 30: nn.Dropout(p=0.2).

        python3 main_uncer.py --params params/regress_train_uncer_dropout.json

    Setting2: line 10, line 30: nn.Dropout(p=0.3).

    Setting3: line 10, line 30: nn.Dropout(p=0.4).

    Setting4: line 30: nn.Dropout(p=0.5).

    Setting5: line 30: nn.Dropout(p=0.2).

## Stage 2: test-time fitting

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
