
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import keras
import matplotlib.pyplot as plt




train_data, validation_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)


embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
#He didn't uses any RNN here.
#You removed recurrance ok
#NO LONG term memory
#not scalable it need to be very very large network
#this consider bad because the order in text is gone.
#no dependencis in 1 feature vector you created now. so you can't create long term memory.

#https://www.youtube.com/watch?v=ySEx_Bqxvvo&t=5s&ab_channel=AlexanderAmini
#Attendation is all you need similarity.

#"He tossed the tennis ball to serve"

#Self-Attention.
#1-encode poition information (word by word embedding)
#2-extract query,key,value for search.



hub_layer = hub.KerasLayer(embedding, input_shape=[],dtype=tf.string, trainable=True)

model = tf.keras.models.Sequential([
    hub_layer,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')])

model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('saved_Model.h5')
callbacks = [early_stop, checkpoint]

history = model.fit(train_data.shuffle(10000).batch(64),
                    epochs=10,
                    validation_data=validation_data.batch(64),
                    callbacks = callbacks,
                    verbose=1)


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


plot_graphs(history=history,metric='accuracy')

model.save("saved_Model.h5")

loaded_model = tf.keras.models.load_model('saved_Model.h5', custom_objects = {'KerasLayer':hub.KerasLayer})
loaded_model.evaluate(test_data.batch(64), verbose=2)

