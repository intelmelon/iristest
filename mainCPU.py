import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import computational_graph as c
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
import six
import csv

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
model = FunctionSet(l1=F.Linear(4, n_units),
                    l2=F.Linear(n_units, 3))

def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    y  = model.l2(h1)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

x_train, x_test = np.split(X,[105])
y_train, y_test = np.split(Y,[105])
N_test = y_test.size

optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())
optimizer.zero_grads()

serializers.load_npz('./model/cpu.l1',model.l1)
serializers.load_npz('./model/cpu.l2',model.l2)
serializers.load_npz('./model/cpu.state',optimizer)


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
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

def predict_forward(x_data):
    x = Variable(x_data)
    h1 = F.relu(model.l1(x))
    y  = model.l2(h1)
    return y

for i in range(105,150):
	#print X[i,0:]
	test_x = np.array([[X[i,0] ,  X[i,1] ,  X[i,2],  X[i,3] ]],dtype=np.float32)
	y = predict_forward(test_x)
	print i
	print F.softmax(y).data

serializers.save_npz('./model/cpu.l1',model.l1)
serializers.save_npz('./model/cpu.l2',model.l2)
serializers.save_npz('./model/cpu.state',optimizer)
