import os
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)

@tf.function
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

models = []

conv_model_1 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(
      input_shape=(28,28,1), 
      filters=8,
      kernel_size=3,
      strides=2,
      padding='same',
      activation='relu',
      name='Conv1'
  ),
  tf.keras.layers.Conv2D(
      input_shape=(14,14,8), 
      filters=8,
      kernel_size=3,
      strides=2,
      activation='relu',
      name='Conv2'
  ),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(
      10,
      activation=tf.nn.softmax,
      name='Softmax'
  )
])
models.append(conv_model_1)

conv_model_2 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(
      input_shape=(28,28,1), 
      filters=8,
      kernel_size=3,
      strides=2,
      padding='same',
      activation='relu',
      name='Conv1'
  ),
  tf.keras.layers.Conv2D(
      input_shape=(14,14,8), 
      filters=8,
      kernel_size=3,
      strides=1,
      activation='relu',
      name='Conv2'
  ),
  tf.keras.layers.Conv2D(
      input_shape=(14,14,8), 
      filters=8,
      kernel_size=3,
      strides=2,
      activation='relu',
      name='Conv3'
  ),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(
      10,
      activation=tf.nn.softmax,
      name='Softmax'
  )
])
models.append(conv_model_2)

flat_model_1 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(
      10,
      activation=tf.nn.softmax,
      name='Softmax'
  )
])
models.append(flat_model_1)

[model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    ) for model in models]

[model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test
    ) for model in models]

names = ['conv_model', 'conv_model', 'flat_model']
versions = [1, 2, 1]

for model, name, version in zip(models, names, versions):
    export_path = os.path.join(MODEL_DIR, name, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )