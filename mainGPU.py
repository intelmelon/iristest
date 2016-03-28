import numpy as np
import chainer
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import computational_graph as c
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import six
import csv

class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):

	h1 = F.dropout(F.relu(self.l1(x)), train=True)
        return self.l3(h1)

X = np.ones((150,4))
Y = np.ones(150)
f = open('iris_teach.csv','r')
reader = csv.reader(f)

cnt = 0;
for row in reader:
	X[cnt,0] = row[0]
	X[cnt,1] = row[1]
	X[cnt,2] = row[2]
	X[cnt,3] = row[3]
	Y[cnt] = row[4]
	cnt+=1

f.close()

f = open('iris_test.csv','r')
reader = csv.reader(f)

for row in reader:
	X[cnt,0] = row[0]
	X[cnt,1] = row[1]
	X[cnt,2] = row[2]
	X[cnt,3] = row[3]
	Y[cnt] = row[4]
	cnt+=1

f.close()

X = X.astype(np.float32)
Y = Y.flatten().astype(np.int32)
n_units   = 100

mlp = MLP(4, n_units, 3)
model = L.Classifier(mlp)

x_train, x_test = np.split(X,[105])
y_train, y_test = np.split(Y,[105])
N_test = y_test.size

optimizer = optimizers.SGD()
optimizer.setup(model)

#GPU
cuda.get_device(0).use()
model.to_gpu()
xp = cuda.cupy
#xp = np

N =105 
batchsize = 5
n_epoch = 100	

for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):

	x_batch = Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
	y_batch = Variable(xp.asarray(y_train[perm[i:i + batchsize]]))
	optimizer.update(model,x_batch,y_batch)
        sum_loss += float(model.loss.data) * len(y_batch.data)
        sum_accuracy += float(model.accuracy.data) * len(y_batch.data)
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

x = chainer.Variable(xp.asarray(x_train[0:1]),
                             volatile='on')
t = chainer.Variable(xp.asarray(y_train[0:1]),
                             volatile='on')
loss = model(x, t)
print loss.data
loss = mlp(x)
print loss.data
print t.data

serializers.save_npz('./model/gpu.l1', mlp.l1)
serializers.save_npz('./model/gpu.l3', mlp.l3)
serializers.save_npz('./model/gpu.state', optimizer)

