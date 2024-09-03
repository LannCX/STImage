
The implementation of our paper [*Video-to-Image casting: A flatting method for video analysis*](https://dl.acm.org/doi/abs/10.1145/3474085.3475424).


## Run
We use weights pre-trained on imagenet, create a "models/" folder  and put these weights into this folder before training.

After downloading the specific dataset (UCF101, DTDB, Something-Something V1), run the following command to train:
```
python train.py --cfg #File name of each experiment#
``` 

## Citation
If you find our code useful in your work, please consider using the following citation:
```
@inproceedings{chen2021video,
  title={Video-to-Image casting: A flatting method for video analysis},
  author={Chen, Xu and Gao, Chenqiang and Yang, Feng and Wang, Xiaohan and Yang, Yi and Han, Yahong},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4958--4966},
  year={2021}
}
```
