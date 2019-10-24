import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, time

def read_measurement(filename):
  data = np.load('data/'+filename)
  x_train = data[0][0][0]
  y_train = data[0][0][1]
  
  x_test = data[1][0][0]
  y_test = data[1][0][1]
  
  
  metadata = data[0][1]
  for i in metadata:
    print(i, metadata[i])
  print()
  print ('train data ', x_train.shape, y_train.shape)
  print ('test data  ', x_test.shape, y_test.shape)
  ns = len(set(y_train))
  print('\nunique classes', ns, '\nmeasurements per sample',int(y_train.shape[0]/ns))
  print()
  time.sleep(1)

  return (x_train, y_train), (x_test, y_test), metadata

def run_model(filename='', epochs=100, n=10):
  
  (x_train, y_train), (x_test, y_test), metadata = read_measurement(filename)
  n_classes = metadata['n_classes']
  model = neural_net(n_classes=n_classes)
  loss_,acc_=[],[]
  for _ in range(n):
    model.fit(x_train, y_train, epochs=epochs)
    result = model.evaluate(x_test, y_test)
    print(result)
    loss_.append(result[0])
    acc_.append(result[1])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim(0,1.1)
    ax1.plot(loss_,'r--')
    ax2.plot(acc_)
    fig.tight_layout()
    plt.title(filename)
    plt.show()
    result_file = filename[:-4]+' results.txt'
    results = np.vstack((acc_,loss_))
    np.savetxt ('data/'+result_file,results)
  return None



def neural_net(n_classes=10):
  
  model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(900, activation=tf.nn.relu),
  tf.keras.layers.Dense(450, activation=tf.nn.relu),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

def get_filename():
  
  files = os.listdir('data')
  for e,file in enumerate(files):
    print(e,file)
  filename = files[int(input('Enter file number '))]
  print(filename)
  time.sleep(1)
  print()
  return filename

run_model(filename=get_filename(),
            epochs=1,
            n=10)
