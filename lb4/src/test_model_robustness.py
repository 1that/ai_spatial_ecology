import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'pipe_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
classes = test_generator.class_indices


def test_model_robustness(model_path, test_dir):
    import cv2
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    variations = ['original', 'bright', 'dark', 'blur', 'noise']
    results = {}

    for var in variations:
        correct = 0
        total = 0

        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_file in os.listdir(class_dir)[:10]:
                img_path = os.path.join(class_dir, img_file)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if var == 'bright':
                    img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
                elif var == 'dark':
                    img = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
                elif var == 'blur':
                    img = cv2.GaussianBlur(img, (15, 15), 0)
                elif var == 'noise':
                    noise = np.random.randint(-30, 30, img.shape)
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)

                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0

                input_data = np.expand_dims(img, axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                pred = np.argmax(output[0])
                true = list(classes.keys()).index(class_name)

                if pred == true:
                    correct += 1
                total += 1

        results[var] = correct / total

    return results


robustness = test_model_robustness('tflite_models/model_fp16.tflite', 'pipe_dataset/test')
print("\nРобастность модели:")
for var, acc in robustness.items():
    print(f"{var:10s}: {acc:.3f}")
