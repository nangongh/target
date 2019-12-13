### 环境初始化
安装依赖

    python 3.7

    pip install -U -r requirements.txt


### 训练

    python train.py --dataset_root C:\data\sample\dataSet --epochs 50 --cuda 1



预加载`weights/vgg16_reducedfc.pth`开始训练。



### 验证 -- 计算mAP

    python eval.py --trained_model weights/gpu.pth --voc_root C:\opt\ssd.pytorch\test --cuda=1


### 测试结果

    python test.py --trained_model weights/gpu.pth --voc_root C:\opt\ssd.pytorch\test --cuda=1


把预测出的位置画在图上，并保存在 ImageTarget目录中
