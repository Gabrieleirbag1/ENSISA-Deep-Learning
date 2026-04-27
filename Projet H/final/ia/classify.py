import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image

import numpy as np
import json
import os
import shutil
import tempfile
import zipfile

classifier = None
image_size = 224
dynamic_size = False
max_dynamic_size = 512
classes = []
is_keras_model = False
model_kind = "image"
model_input_rank = 4
model_input_length = None


def _extract_io_shape_from_keras_archive(model_path):
  with zipfile.ZipFile(model_path, "r") as archive:
    config = json.loads(archive.read("config.json").decode("utf-8"))

  input_layers = config.get("config", {}).get("input_layers", [])
  input_name = None
  if input_layers and isinstance(input_layers[0], list) and len(input_layers[0]) > 0:
    input_name = input_layers[0][0]

  for layer in config.get("config", {}).get("layers", []):
    if layer.get("class_name") != "InputLayer":
      continue
    if input_name is not None and layer.get("name") != input_name:
      continue
    layer_cfg = layer.get("config", {})
    batch_shape = layer_cfg.get("batch_shape") or layer_cfg.get("batch_input_shape")
    if isinstance(batch_shape, list) and len(batch_shape) >= 2:
      rank = len(batch_shape)
      input_length = int(batch_shape[1]) if batch_shape[1] is not None else None
      return rank, input_length

  return None, None


def _build_ecg_fallback_model(model_name, input_length):
  if input_length is None:
    raise Exception("Unable to infer ECG input length from model archive")

  n_classes = len(classes) if len(classes) > 0 else 2

  if model_name.upper().startswith("MLP"):
    inp = tf.keras.layers.Input(shape=(input_length,), name="ecg_input")
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="MLP_ECG200")

  if model_name.upper().startswith("CNN"):
    inp = tf.keras.layers.Input(shape=(input_length, 1), name="ecg_input")
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="CNN1D_ECG200")

  if model_name.upper().startswith("RNN"):
    inp = tf.keras.layers.Input(shape=(input_length, 1), name="ecg_input")
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="RNN_LSTM_ECG200")

  raise Exception(f"Unsupported Keras archive model for fallback loading: {model_name}")


def _try_load_keras_archive_fallback(model_path):
  model_name = os.path.basename(model_path)
  rank, input_length = _extract_io_shape_from_keras_archive(model_path)
  model = _build_ecg_fallback_model(model_name, input_length)

  with zipfile.ZipFile(model_path, "r") as archive:
    # Find the actual weights file — name varies by Keras version
    all_files = archive.namelist()
    weights_candidates = [f for f in all_files if f.endswith(".weights.h5") or f == "model.weights.h5"]
    if not weights_candidates:
      # Keras 3 stores weights directly as .h5 at root level
      weights_candidates = [f for f in all_files if f.endswith(".h5")]
    if not weights_candidates:
      raise Exception(f"No weights file found in archive. Contents: {all_files}")
    weights_filename = weights_candidates[0]
    print(f"Using weights file: {weights_filename}")

    with tempfile.TemporaryDirectory() as temp_dir:
      weights_path = os.path.join(temp_dir, os.path.basename(weights_filename))
      with open(weights_path, "wb") as weights_file:
        weights_file.write(archive.read(weights_filename))
      model.load_weights(weights_path, by_name=True, skip_mismatch=True)

  return model, rank, input_length

model_handle_map = {
  "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
  "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2",
  "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2",
  "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2",
  "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2",
  "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2",
  "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2",
  "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2",
  "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2",
  "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2",
  "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2",
  "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2",
  "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2",
  "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2",
  "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2",
  "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2",
  "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2",
  "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2",
  "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2",
  "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2",
  "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2",
  "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2",
  "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2",
  "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
  "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/classification/1",
  "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/classification/1",
  "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/classification/1",
  "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/classification/1",
  "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/classification/1",
  "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/classification/1",
  "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/classification/1",
  "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1",
  "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
  "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4",
  "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4",
  "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4",
  "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/classification/4",
  "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
  "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4",
  "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4",
  "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/classification/4",
  "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/classification/4",
  "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/classification/4",
  "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
  "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4",
  "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4",
  "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5",
  "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
  "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5",
  "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5",
}

model_image_size_map = {
  "efficientnetv2-s": 384,
  "efficientnetv2-m": 480,
  "efficientnetv2-l": 480,
  "efficientnetv2-b0": 224,
  "efficientnetv2-b1": 240,
  "efficientnetv2-b2": 260,
  "efficientnetv2-b3": 300,
  "efficientnetv2-s-21k": 384,
  "efficientnetv2-m-21k": 480,
  "efficientnetv2-l-21k": 480,
  "efficientnetv2-xl-21k": 512,
  "efficientnetv2-b0-21k": 224,
  "efficientnetv2-b1-21k": 240,
  "efficientnetv2-b2-21k": 260,
  "efficientnetv2-b3-21k": 300,
  "efficientnetv2-s-21k-ft1k": 384,
  "efficientnetv2-m-21k-ft1k": 480,
  "efficientnetv2-l-21k-ft1k": 480,
  "efficientnetv2-xl-21k-ft1k": 512,
  "efficientnetv2-b0-21k-ft1k": 224,
  "efficientnetv2-b1-21k-ft1k": 240,
  "efficientnetv2-b2-21k-ft1k": 260,
  "efficientnetv2-b3-21k-ft1k": 300, 
  "efficientnet_b0": 224,
  "efficientnet_b1": 240,
  "efficientnet_b2": 260,
  "efficientnet_b3": 300,
  "efficientnet_b4": 380,
  "efficientnet_b5": 456,
  "efficientnet_b6": 528,
  "efficientnet_b7": 600,
  "inception_v3": 299,
  "inception_resnet_v2": 299,
  "mobilenet_v2_100_224": 224,
  "mobilenet_v2_130_224": 224,
  "mobilenet_v2_140_224": 224,
  "nasnet_large": 331,
  "nasnet_mobile": 224,
  "pnasnet_large": 331,
  "resnet_v1_50": 224,
  "resnet_v1_101": 224,
  "resnet_v1_152": 224,
  "resnet_v2_50": 224,
  "resnet_v2_101": 224,
  "resnet_v2_152": 224,
  "mobilenet_v3_small_100_224": 224,
  "mobilenet_v3_small_075_224": 224,
  "mobilenet_v3_large_100_224": 224,
  "mobilenet_v3_large_075_224": 224,
}

def config (model_name) :

  def load_keras_model (model_name) :
    global classifier
    global image_size
    global dynamic_size
    global classes
    global is_keras_model
    global model_kind
    global model_input_rank
    global model_input_length
    classifier = None
    is_keras_model = False
    classes = []
    model_kind = "image"
    model_input_rank = 4
    model_input_length = None
    model_path = model_name
    if not os.path.isabs(model_path):
      model_path = os.path.abspath(os.path.join("./models", model_name))
    print(f"Loading Keras model from {model_path}")
    if not os.path.exists(model_path):
      raise Exception("keras model not found: {}".format(model_path))
    try:
      classifier = tf.keras.models.load_model(model_path, compile=False)
      print("Model loaded successfully")
    except Exception as e:
      print(f"Error loading model with tf.keras (compile=False): {e}")  # <-- you'll see the real error here
      if model_path.endswith(".keras") and zipfile.is_zipfile(model_path):
        classifier, fallback_rank, fallback_input_length = _try_load_keras_archive_fallback(model_path)
        model_input_rank = fallback_rank if fallback_rank is not None else len(classifier.input_shape)
        model_input_length = fallback_input_length
        print("Keras archive fallback loading succeeded")
      else:
        raise
    is_keras_model = True

    # Infer expected input size from model input shape.
    input_shape = classifier.input_shape
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    model_input_rank = len(input_shape)

    if len(input_shape) == 4 and input_shape[1] and input_shape[2]:
      model_kind = "image"
      image_size = int(input_shape[1])
      dynamic_size = False
      print(f"Image size set to {image_size}")
    elif len(input_shape) in [2, 3]:
      model_kind = "ecg"
      model_input_length = int(input_shape[1]) if input_shape[1] is not None else None
      dynamic_size = True
      print(f"ECG model detected, input length={model_input_length}")
    else:
      dynamic_size = True
      print("Dynamic size enabled")

    labels_path = os.path.splitext(model_path)[0] + ".labels"
    if os.path.exists(labels_path):
      with open(labels_path) as f:
        labels = f.readlines()
        classes = [l.strip() for l in labels]
      print(f"Loaded {len(classes)} classes from {labels_path}")
    else:
      print("No labels file found")

  def load_hub_model (model_name) :
    global classifier
    global is_keras_model
    global model_kind
    global model_input_rank
    global model_input_length
    classifier = None
    is_keras_model = False
    model_kind = "image"
    model_input_rank = 4
    model_input_length = None
    try :
      if classifier is None :
        model_handle = os.path.abspath("./models/"+model_name)
        classifier = hub.load(model_handle)
    except Exception as e:
      print ("local handle not available : {} : {}".format(type(e), e.args))
    try :
      if classifier is None :
        model_handle = model_handle_map[model_name]
        classifier = hub.load(model_handle)
        tf.saved_model.save(classifier, "./models/"+model_name)
    except Exception as e:
      print ("remote handle not available : {} : {}".format(type(e), e.args))

  def load_labels () :
    global classes
    downloaded_file = None
    try :
      if downloaded_file is None :
        labels_file = os.path.abspath("./models/ImageNetLabels.txt")
        downloaded_file = tf.keras.utils.get_file("labels.txt", origin="file://"+labels_file)       
    except Exception as e:
      print ("local file not available : {} : {}".format(type(e), e.args))
    try :
      if downloaded_file is None :
        labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)       
        if downloaded_file is not None :
          shutil.copyfile(downloaded_file, "./models/ImageNetLabels.txt")
    except Exception as e:
      print ("remote file not available : {} : {}".format(type(e), e.args))
    if downloaded_file is None :
      raise Exception ("labels are not available")
    with open(downloaded_file) as f:
      labels = f.readlines()
      classes = [l.strip() for l in labels]

  def compute_sizes (model_name) :
    global image_size
    global dynamic_size
    if model_name in model_image_size_map:
      image_size = model_image_size_map[model_name]
      dynamic_size = False
      print(f"Images will be converted to {image_size}x{image_size}")
    else:
      dynamic_size = True
      print(f"Images will be capped to a max size of {max_dynamic_size}x{max_dynamic_size}")

  def dry_run () :
    global classifier
    input_shape = [1, image_size, image_size, 3]
    warmup_input = tf.random.uniform(input_shape, 0, 1.0)
    warmup_logits = classifier(warmup_input).numpy()

  if model_name.endswith(".keras"):
    load_keras_model(model_name)
  else:
    load_hub_model (model_name)
    load_labels ()
    compute_sizes (model_name)
    dry_run ()
        
def classify (image_name) :

  if model_kind != "image":
    raise Exception("Current model is configured for ECG data. Use /classify_ecg endpoint.")

  def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image

  def load_image(image_url, image_size=256, dynamic_size=False, max_dynamic_size=512):
    fd = tf.io.gfile.GFile(image_url, 'rb')
    img = preprocess_image(Image.open(fd))
    if tf.reduce_max(img) > 1.0:
      img = img / 255.
    if len(img.shape) == 3:
      img = tf.stack([img, img, img], axis=-1)
    if not dynamic_size:
      img = tf.image.resize_with_pad(img, image_size, image_size)
    elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
      img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
    return img

  def build_result (probabilities) :
    global classes
    top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
    if len(classes) == 0:
      classes = ["class_{}".format(i) for i in range(probabilities.shape[1])]
    np_classes = np.array(classes)
    includes_background_class = probabilities.shape[1] == 1001
    result = []
    for i, item in enumerate(top_5):
      class_index = item if includes_background_class else item + 1
      result.append ({ "class" : classes[class_index], "probability" : float(probabilities[0][top_5][i])})
    return result
    
  image = load_image(image_name, image_size, dynamic_size, max_dynamic_size)
  if is_keras_model:
    preds = classifier(image, training=False)
    probabilities = tf.nn.softmax(preds).numpy()
  else:
    probabilities = tf.nn.softmax(classifier(image)).numpy()
  result = build_result (probabilities)
  return result


def _softmax_np(logits):
  logits = np.asarray(logits, dtype=np.float32)
  logits = logits - np.max(logits, axis=1, keepdims=True)
  exp = np.exp(logits)
  return exp / np.sum(exp, axis=1, keepdims=True)


def _normalize_probabilities(raw_output):
  arr = np.asarray(raw_output, dtype=np.float32)
  if arr.ndim == 1:
    arr = arr.reshape(1, -1)

  if arr.shape[1] == 1:
    col = arr[:, 0]
    if np.any(col < 0.0) or np.any(col > 1.0):
      col = 1.0 / (1.0 + np.exp(-col))
    col = np.clip(col, 0.0, 1.0)
    return np.stack([1.0 - col, col], axis=1)

  sums = np.sum(arr, axis=1, keepdims=True)
  looks_like_probs = np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.all(np.abs(sums - 1.0) < 1e-3)
  if looks_like_probs:
    return arr
  return _softmax_np(arr)


def _load_matrix(path):
  loaders = [
    (lambda p: np.loadtxt(p, dtype=np.float32), "np.loadtxt default"),
    (lambda p: np.loadtxt(p, delimiter=",", dtype=np.float32), "np.loadtxt comma"),
    (lambda p: np.loadtxt(p, delimiter="\t", dtype=np.float32), "np.loadtxt tab"),
    (lambda p: np.loadtxt(p, delimiter=" ", dtype=np.float32), "np.loadtxt space"),
    (lambda p: np.genfromtxt(p, dtype=np.float32), "np.genfromtxt default"),
  ]
  for loader, _ in loaders:
    try:
      data = loader(path)
      if data is None:
        continue
      data = np.asarray(data, dtype=np.float32)
      if data.size == 0:
        continue
      if data.ndim == 1:
        data = data.reshape(1, -1)
      return data
    except Exception:
      pass
  raise ValueError(f"Impossible to load ECG signal file: {path}")


def _looks_like_label_column(col):
  col = np.asarray(col, dtype=np.float32)
  if col.ndim != 1 or col.size == 0:
    return False
  if np.any(~np.isfinite(col)):
    return False
  rounded = np.round(col)
  if np.max(np.abs(col - rounded)) > 1e-6:
    return False
  unique_count = np.unique(rounded).size
  return unique_count <= 20


def _prepare_ecg_batch(signal):
  global model_input_length
  global model_input_rank

  arr = np.asarray(signal, dtype=np.float32)
  if arr.size == 0:
    raise ValueError("Empty ECG signal")

  if arr.ndim == 1:
    arr = arr.reshape(1, -1)
  elif arr.ndim > 2:
    arr = arr.reshape(arr.shape[0], -1)

  arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

  if model_input_length is not None:
    if arr.shape[1] == model_input_length + 1 and _looks_like_label_column(arr[:, 0]):
      arr = arr[:, 1:]

    if arr.shape[1] > model_input_length:
      arr = arr[:, :model_input_length]
    elif arr.shape[1] < model_input_length:
      pad = model_input_length - arr.shape[1]
      arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")

  row_min = np.min(arr, axis=1, keepdims=True)
  row_max = np.max(arr, axis=1, keepdims=True)
  denom = row_max - row_min
  denom[denom == 0.0] = 1.0
  arr = (arr - row_min) / denom

  if model_input_rank == 3:
    arr = arr[..., np.newaxis]
  return arr


def _build_batch_result(probabilities):
  global classes

  probabilities = np.asarray(probabilities, dtype=np.float32)
  if probabilities.ndim == 1:
    probabilities = probabilities.reshape(1, -1)

  class_count = probabilities.shape[1]
  if len(classes) == 0 or len(classes) < class_count:
    classes = [f"class_{i}" for i in range(class_count)]

  top_k = min(5, class_count)
  results = []
  for sample_idx in range(probabilities.shape[0]):
    top_indices = np.argsort(probabilities[sample_idx])[::-1][:top_k]
    top_values = probabilities[sample_idx][top_indices]
    sample_result = []
    for rank, class_idx in enumerate(top_indices):
      sample_result.append(
        {
          "class": classes[int(class_idx)],
          "probability": float(top_values[rank]),
        }
      )
    results.append(sample_result)
  return results


def classify_ecg_from_array(signal):
  global classifier
  global model_kind
  if classifier is None:
    raise Exception("Model is not configured. Call /config first.")
  if model_kind != "ecg":
    raise Exception("Current model is configured for images. Use /classify endpoint.")

  batch = _prepare_ecg_batch(signal)
  preds = classifier(batch, training=False)
  if hasattr(preds, "numpy"):
    preds = preds.numpy()
  probabilities = _normalize_probabilities(preds)
  results = _build_batch_result(probabilities)
  return results[0] if len(results) == 1 else results


def classify_ecg_file(file_name):
  ext = os.path.splitext(file_name)[1].lower()
  if ext == ".npy":
    data = np.load(file_name)
  else:
    data = _load_matrix(file_name)

  if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1 and model_input_length is not None:
    if data.shape[0] == model_input_length:
      data = data.reshape(1, -1)

  return classify_ecg_from_array(data)

