# 猫狗大战

[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)




## 数据

此数据集可以从 kaggle 上下载。[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

此外还有一个数据集也非常好：[The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)


# Pytorch
Pytorch [官网](http://pytorch.org/) 
注意,根据Python版本和已经安装CUDA的版本不一样,需要安装的Pytorch版本也不一致.


# 训练时间
本地有TitanXp * 2
Epochs 是24个,总训练时间是27分43秒

# 数据分离
安装[kaggle-cli](https://github.com/floydwch/kaggle-cli)
`pip install kaggle-cli`


使用kaggle-cli 下载数据:
`kg download -u <用户名> -p <密码> -c dogs-vs-cats-redux-kernels-edition`

这个命令会下载所有猫狗项目下的两个zip压缩包,已经sample submit .

将数据分离到不同的文件夹.文件夹结构如下:
```
├── data
│   ├── train
│   │   ├── cat
│   │   └── dog
│   └── val
│       ├── cat
│       └── dog
├── logs
│   └── train
└── test
    └── test

```

使用的shell代码如下:
```shell
mkdir -p data/

# unzip!
unzip train.zip
unzip test.zip
# prep train directory and split train/trainval
cd train
# sanity check
find ./ -type f -name 'cat*' | wc -l # 12500
find ./ -type f -name 'dog*' | wc -l # 12500
mkdir -p train/dog
mkdir -p train/cat
mkdir -p val/dog
mkdir -p val/cat
# Randomly move 90% into train and val, 
# if reproducability is important you can pass in a source to shuf
find . -name "cat*" -type f | shuf -n11250 | xargs -I file mv file train/cat/
find . -maxdepth 2 -type f -name 'cat*'| xargs -I file mv file val/cat/
# now dogs
find . -name "dog*" -type f | shuf -n11250 | xargs -I file mv file train/dog/
find . -maxdepth 2 -type f -name 'dog*'| xargs -I file mv file val/dog/

# requires gnu utils (brew install coreutils)
echo cat*.jpg | xargs mv -t cat
echo dog*.jpg | xargs mv -t dog
```
