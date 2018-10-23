# RecurrentGaze

This is the implementation of the ["Recurrent CNN for 3D Gaze Estimation using Appearance and Shape Cues"](http://bmvc2018.org/contents/papers/0871.pdf) paper, accepted for BMVC 2018. 

Trained weights using EYEDIAP dataset are available for the following models: NFEL5836, NFEL5836_2918 and NFEL5836GRU. Models used in the paper, divided into folds, are available on demand due to their size. 

## Main files

All the information is included inside the files, but here is a brief summary:

### *generate_training_CS* and *generate_training_FT*:

Scripts to generate data files from EYEDIAP *Continuous Screen target* and *Floating Target* subsets, respectively, so that the input data is compatible with *Train* and *Test* scripts. The [EYEDIAP dataset](https://www.idiap.ch/dataset) is required. If using another dataset, these scripts have to be changed accordingly. The scripts also load the 3D landmarks computed using [Bulat et al. 2017 code](https://github.com/1adrianb/face-alignment). Examples of the format of the input 3D landmarks and generated output files can be found in the *Examples* folder. 

### *Train*:

Main script to train (and evaluate) the gaze estimation models included in the paper, using the files generated from *generate_training_X* script.

Usage example for NFEL5836GRU model:
```
python3 Train.py -t FT_SM_NFEL5836GRU_fold3 -exp NFEL5836GRU -dp 0.3 -bs 8 -aug 1
-lr 0.0001 -epochs 21 -gt "/Work/EYEDIAP/Annotations_final_exps/gt_cam_FT_S.txt" "/Work/EYEDIAP/Annotations_final_exps/gt_cam_FT_M.txt"
-vgt "/Work/EYEDIAP/Annotations_final_exps/gtv_cam_FT_S.txt" "/Work/EYEDIAP/Annotations_final_exps/gtv_cam_FT_M.txt"
-data "/Work/EYEDIAP/Annotations_final_exps/data_FT_S.txt" "/Work/EYEDIAP/Annotations_final_exps/data_FT_M.txt"
-feats "/Work/EYEDIAP/Annotations_final_exps/face_features_FT_S.txt" "/Work/EYEDIAP/Annotations_final_exps/face_features_FT_M.txt"
-test 2_A_FT_S 2_A_FT_M 3_A_FT_S 3_A_FT_M 8_A_FT_S 8_A_FT_M 16_A_FT_S 16_A_FT_M 16_B_FT_S 16_B_FT_M 
-p "/Work"
 ```

### *Test*:

Main script to test the trained gaze estimation models and visualize the predicted outputs. We include a test scenario with images and calibration file in the *Test* folder. In this case, we use [Openface](https://github.com/TadasBaltrusaitis/OpenFace) to compute 3D information of the subject wrt the camera coordinate system (head pose and 3D landmarks to compute eye centers), which can be found in *CCS_3D_info* file. 3D landmarks used to train the model are, again, computed using Bulat et al. code. 

Usage example for NFEL5836 model:
```
python3 Test.py -exp NFEL5836_2918 -data C:\path\Test\data.txt 
-info C:\path\Test\CCS_3D_info.txt -lndmk C:\path\Test\landmarks.txt 
-cal C:\path\Test\calibration.txt -p C:\path\
```

## Requirements

The code was tested using the following versions:
- Python=3.5.4
- tensorflow=1.6
- keras=2.1.5
- [keras_vggface](https://github.com/rcmalli/keras-vggface)=0.5
- numpy=1.14.2
- cv2=3.4.0

You may also use the included Dockerfile.

## EYEDIAP folds distribution
In the paper, we evaluate the cross-subject 3D gaze estimation task by means of N-fold cross-validation. The different folds are detailed below for each target type.

### Floating target
| Num. fold     | Num. subjects | Folders  | 
| --------   | ---------- | -------- |
| 1          | 4          |'5_A_FT_S', '5_A_FT_M', '10_A_FT_S', '10_A_FT_M', '11_A_FT_S', '11_A_FT_M','14_A_FT_S', '14_A_FT_M', '14_B_FT_S', '14_B_FT_M'|
| 2          | 4          |'1_A_FT_S', '1_A_FT_M', '4_A_FT_S', '4_A_FT_M', '6_A_FT_S', '6_A_FT_M', '15_A_FT_S', '15_A_FT_M', '15_B_FT_S', '15_B_FT_M'          |
| 3          | 4          | '2_A_FT_S', '2_A_FT_M', '3_A_FT_S', '3_A_FT_M', '8_A_FT_S', '8_A_FT_M', '16_A_FT_S', '16_A_FT_M', '16_B_FT_S', '16_B_FT_M' |
| 4          | 4          | '7_A_FT_S', '7_A_FT_M', '9_A_FT_S', '9_A_FT_M', '12_B_FT_S', '12_B_FT_M', '13_B_FT_S', '13_B_FT_M' |

### Continuous screen target
| Num. fold     | Num. subjects | Folders  | 
| --------   | --------      | -------- | 
| 1          | 3             |'7_A_CS_S', '7_A_CS_M', '10_A_CS_S', '10_A_CS_M', '15_A_CS_S', '15_A_CS_M'|
| 2          | 3             |'2_A_CS_S', '2_A_CS_M', '4_A_CS_S', '4_A_CS_M', '8_A_CS_S', '8_A_CS_M' |
| 3          | 3             |'3_A_CS_S', '3_A_CS_M', '6_A_CS_S', '6_A_CS_M', '16_A_CS_S', '16_A_CS_M' |
| 4          | 3             |'1_A_CS_S', '1_A_CS_M', '5_A_CS_S', '5_A_CS_M', '9_A_CS_S', '9_A_CS_M'|
| 5          | 2             | '14_A_CS_S', '14_A_CS_M', '11_A_CS_S', '11_A_CS_M' |

## Remarks

If you find any bugs or have any comments or suggestions please contact me on Github or feel free to open an Issue. All contributions are welcomed!

## Citation

If you find this code useful, please cite the following paper:

Palmero, C., Selva, J., Bagheri, M. A., & Escalera, S. Recurrent CNN for 3D Gaze Estimation using Appearance and Shape Cues. Proc. of British Machine Vision Conference (BMVC), 2018.
