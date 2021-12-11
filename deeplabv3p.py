import paddle
from paddleseg.models import DeepLabV3P
from paddleseg.models.backbones import ResNet50_vd
import paddle.nn.functional as F
from readdata import read_images,read_labels
ResNet50_vd = ResNet50_vd()
model = DeepLabV3P(num_classes=5,backbone=ResNet50_vd)
import paddleseg.transforms as T

data_path = 'images_seg'
train_path = 'images_seg/train.list'
batch_size = 64

images_list = []
labels_list = []
with open(train_path) as f:
    lines = f.readlines()
    for line in lines:
        image_path,label_path = line.split(' ')
        images_list.append(str(image_path).strip())
        labels_list.append(str(label_path).strip())

# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)

optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
from paddleseg.models.losses import CrossEntropyLoss
#loss = CrossEntropyLoss()
def train(model):
    model.train()
    epochs = 5
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        batch_id = 0
        for batch in images_list:
            x_data = read_images(data_path=data_path,images_path=[batch])
            y_data = read_labels(data_path=data_path,labels_path=[batch])

            x_data = paddle.to_tensor(x_data, dtype='float32')
            y_data = paddle.to_tensor(y_data, dtype='int64')
            #print(y_data)
            predicts = model(x_data)[0]
            #print(predicts)
            loss = F.softmax_with_cross_entropy(predicts, y_data,axis=1)
            #loss = loss(predicts,y_data)
            print(paddle.sum(loss).numpy())
            # 计算损失
            #acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            # if batch_id % 300 == 0:
            #     print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, sum(loss.numpy())))
            optim.step()
            optim.clear_grad()
            #
            # lr = optimizer.get_lr()
            # lr_sche = optimizer._learning_rate
train(model)