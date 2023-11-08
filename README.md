# A Ranking Information Based Network for Facial Beauty Prediction



## Pipeline
![pipeline](./img/1.png)


## Performances
![performances](./img/2.png)


## Codes

### Requirements

- Test on NVIDIA TITAN GPU, Ubuntu 18.04.6, Python3.8, CUDA 12.0 and PyTorch 1.12.0

```
pip install -r requirements.txt
```



### Dataset

Get ready for the SCUT-FBP5500 dataset from [here](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).




### Training

Start training with the required parameters `data_dir`.

```
python train.py --data_dir YOUR_DATASET_DIR
```

Furthermore, you can set other parameters.

- Load pre-trained model before training by using `--load_from`

- Set the directory to store the state-dict by using `--checkpoint`
- Set the hyperparameter of losses by using `--param`
- Set the training batch size by using `--batch_size`
- Set the training epoch by using `--epoch`
- Set the training method, including 6-4 and cross_validation, by using `--method`



### Testing

Start testing with two required parameters `data_dir` and `load_from`.

```
python test.py --data_dir YOUR_DATASET_DIR --load_from YOUR_WEIGHT_OF_MODEL
```

Note that you only need to set the `load_from` parameter to be the same directory of the `checkpoint` which  is set in training phase, unless you have renamed the state-dict file or moved it to somewhere else.

## Apply code to other tasks

Our proposed method can be applied to other tasks with ranking information, including most rating tasks, age estimation, etc. For those tasks without ranking information, if you can preprocess your dataset to gain **reasonable** ranking information, then our method can also be effective.

When applying the code to other tasks, you just need to follow the method in our paper and add the Rank Module and Adaptive Weight Module to your network, and add the corresponding ranking loss during the training stage.

---
Here is an example for applying code to other tasks we have conducted.

To demonstrate the transferability of our proposed method. We conducted tests on age estimation tasks. We keep almost all of the code and only made modifications to the `dataset.py` and the parameter `num_classes`. The results are shown in the following table and MAE is calculated for various methods.

|           Method            | CACD*  | MegaAge-Asian |
| :-------------------------: | :----: | :-----------: |
| Without Ranking Information | 3.8183 |    2.9961     |
|  With Ranking Information   | 3.6021 |    2.8683     |

**CACD***: The original CACD dataset is collected from Internet and has many samples which the labels do not match the images. We manually selected a small portion of reliable images from the CACD dataset and named it CACD*. The results on the original CACD dataset are 4.6462 (without rank-info) and 4.5820 (with rank-info).

This experiment is only for validating the transferability of our proposed network model. Therefore, we did not use popular image preprocessing methods or particular loss functions for age estimation, modify the network architecture specifically, or compare the final results with the state-of-the-art works in age estimation.
