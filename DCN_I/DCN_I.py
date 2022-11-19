#gyp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense,Embedding,Reshape,Add,Flatten,Lambda,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import pickle
#('----------------------data----------------------')
data = pd.read_csv('data_DCNI.csv')
features_name = list(data)
features_name_x = features_name[:-1]
features_name_y = features_name[-1]
features_x = data[features_name_x]
features_y = data[features_name_y]
features_name_x_n = features_name_x[3:]
features_name_x_n_a = features_name_x_n

n_s = len(np.unique(features_x[features_name_x[0]]))
n_t = len(np.unique(features_x[features_name_x[1]]))
n_b = len(np.unique(features_x[features_name_x[2]]))

with open('z_score_DCNI.pickle', 'rb') as f:
    z_score = pickle.load(f)
features_x[features_name_x_n_a] -= z_score[:len(z_score)//2]
features_x[features_name_x_n_a] /= z_score[len(z_score)//2:]
with open('alpha.pickle', 'rb') as f:
    alpha = pickle.load(f)

# #('----------------------model----------------------')
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        print(input_shape[1])
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[1, features_x.shape[1]+n_s+n_t+n_b-3], initializer='glorot_uniform', name='w_{}'.format(i),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, features_x.shape[1]+n_s+n_t+n_b-3], initializer='zeros', name='b_{}'.format(i), trainable=True))
        self.built = True

    def call(self, input1):
        input = tf.reshape(input1, shape=[-1, 1, input1.shape[1]])
        for i in range(self.num_layer):
            if i == 0:
                cross = tf.keras.layers.Lambda(lambda x: K.batch_dot(K.dot(x, K.transpose(self.W[i])), x) + self.bias[i] + x)(input)
            else:
                cross = tf.keras.layers.Lambda(lambda x: K.batch_dot(K.dot(x, K.transpose(self.W[i])), input) + self.bias[i] + x)(cross)
                # print(cross)
            return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    def get_config(self):
        config = super(CrossLayer, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_layer": self.num_layer
             }
        )
        return config

def my_loss(y_true, y_pred):
    if y_true[:,0]==0:
        diff = alpha['a1']*K.abs((y_true[:,1] - y_pred) / K.clip(K.abs(y_true[:,1]),
                                            K.epsilon(),
                                            None))
    elif y_true[:,0]==1:
        diff = alpha['a2']*K.abs((y_true[:,1] - y_pred) / K.clip(K.abs(y_true[:,1]),
                                            K.epsilon(),
                                            None))
    else:
        diff = K.abs((y_true[:, 1] - y_pred) / K.clip(K.abs(y_true[:, 1]),
                                                      K.epsilon(),
                                                      None))
    print(y_true)
    return 100. * K.mean(diff, axis=-1)


model = load_model('trained_DCNI.H5', custom_objects={'my_loss': my_loss, "CrossLayer": CrossLayer})

# #('----------------------predict----------------------')
xfit = np.arange(0, np.max(features_y))
def fit(x):
    return x
y_pred = np.squeeze(model.predict(features_x))
plt.plot(xfit, fit(xfit), color='black')
plt.scatter(features_y,y_pred,color='blue',s=8,label='Train')
plt.ylabel('Predict', fontsize=15)
plt.xlabel('True', fontsize=15)
plt.legend(loc=0,ncol=1)
plt.show()

# # MAPE
mape = np.mean(abs(y_pred-features_y)/features_y)*100
print('mape:',mape)