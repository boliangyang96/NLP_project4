import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3])
train_loss = np.array([0.5764661234617233, 0.34612741849735273, 0.2706799526140094])
dev_accu = np.array([0.8368983957219251, 0.9064171122994652, 0.93048128342246])
plt.plot(x, train_loss, marker='.', color='skyblue', label='training loss')
plt.plot(x, dev_accu, marker='.', color='green', label='development accuracy')
plt.xticks(x)
plt.xlabel('epochs')
plt.title('Learning Curve')
plt.legend()
plt.savefig('1.jpg')
