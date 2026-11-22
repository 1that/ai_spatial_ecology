import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import random
import tempfile

SAVED_MODEL_DIR = os.path.join(tempfile.mkdtemp(), 'pipe_model_saved')
SAVE_DIR = 'tflite_models'
os.makedirs(SAVE_DIR, exist_ok=True)

model = load_model('model/pipe_model_final.keras')
print("Модель загружена")

model.export(SAVED_MODEL_DIR)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

tflite_fp32 = converter.convert()
with open(f'{SAVE_DIR}/model_fp32.tflite', 'wb') as f:
    f.write(tflite_fp32)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
with open(f'{SAVE_DIR}/model_fp16.tflite', 'wb') as f:
    f.write(tflite_fp16)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()
with open(f'{SAVE_DIR}/model_dynamic.tflite', 'wb') as f:
    f.write(tflite_dynamic)

def representative_dataset():
    paths = glob.glob('pipe_dataset/train/*/*.jpg')
    random.shuffle(paths)
    for img_path in paths[:100]:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)
        yield [img_array]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8 = converter.convert()
with open(f'{SAVE_DIR}/model_int8.tflite', 'wb') as f:
    f.write(tflite_int8)

print("\nСравнение размеров моделей:")
for name in ['model_fp32.tflite', 'model_fp16.tflite', 'model_dynamic.tflite', 'model_int8.tflite']:
    size = os.path.getsize(os.path.join(SAVE_DIR, name)) / 1024 / 1024
    print(f"{name:20s}: {size:.2f} MB")


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'pipe_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


def evaluate_tflite_model(tflite_path, test_generator):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0

    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]

        for j in range(len(x_batch)):
            input_data = x_batch[j:j+1].astype(np.float32)

            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output_data)
            true = np.argmax(y_batch[j])

            if pred == true:
                correct += 1
            total += 1

    return correct / total


print("\nСравнение точности:")
for model_path in ['model_fp32.tflite', 'model_fp16.tflite', 'model_dynamic.tflite', 'model_int8.tflite']:
    acc = evaluate_tflite_model(SAVE_DIR + '/' + model_path, test_generator)
    print(f"{model_path:20s}: accuracy = {acc:.4f}")

with open('labels.txt', 'w') as f:
    for class_name in test_generator.class_indices:
        f.write(f"{class_name}\n")

print("\nМетки классов сохранены в labels.txt")