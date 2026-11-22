import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_pipe_dataset(base_dir='pipe_dataset', n_samples_per_class=100):
    classes = {
        'asbestos': {'color': (120, 120, 120), 'texture': 'fibrous'},
        'hdp': {'color': (30, 30, 30), 'texture': 'smooth'},
        'ceramics': {'color': (140, 90, 70), 'texture': 'glazed'},
        'concrete': {'color': (150, 150, 150), 'texture': 'porous'},
        'metal': {'color': (80, 80, 80), 'texture': 'rusty'},
        'geotextile': {'color': (200, 200, 200), 'texture': 'woven'}
    }

    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(f'{base_dir}/{split}/{class_name}', exist_ok=True)

    def apply_texture(img, texture_type):
        h, w = img.shape[:2]

        if texture_type == 'fibrous':
            for i in range(0, h, 5):
                cv2.line(img, (0, i), (w, i), (100, 100, 100), 1)
        elif texture_type == 'porous':
            for _ in range(50):
                x, y = np.random.randint(10, w - 10, 2)
                cv2.circle(img, (x, y), 3, (100, 100, 100), -1)
        elif texture_type == 'rusty':
            for _ in range(30):
                x, y = np.random.randint(0, w, 2)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        elif texture_type == 'woven':
            for i in range(0, h, 20):
                cv2.line(img, (0, i), (w, i), (180, 180, 180), 2)
                cv2.line(img, (i, 0), (i, h), (180, 180, 180), 2)

        return img

    for class_name, props in classes.items():
        for split in ['train', 'val', 'test']:
            if split == 'train':
                n = n_samples_per_class
            elif split == 'val':
                n = int(n_samples_per_class * 0.15)
            else:
                n = int(n_samples_per_class * 0.15)

            for i in range(n):
                img = np.ones((224, 224, 3), dtype=np.uint8) * 255

                center = (np.random.randint(80, 144), np.random.randint(80, 144))
                axes = (np.random.randint(30, 60), np.random.randint(15, 30))
                angle = np.random.randint(0, 180)
                color = props['color']
                cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)

                img = apply_texture(img, props['texture'])

                bg_color = np.random.randint(200, 255, 3)
                mask = np.all(img == [255, 255, 255], axis=-1)
                img[mask] = bg_color

                noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
                img_int16 = img.astype(np.int16)
                img_noisy = img_int16 + noise
                img = np.clip(img_noisy, 0, 255).astype(np.uint8)

                cv2.imwrite(f'{base_dir}/{split}/{class_name}/{class_name}_{i}.jpg', img)

    print(f"Датaсет создан в папке {base_dir}")
    return classes

classes = create_pipe_dataset('pipe_dataset', 150)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, (class_name, props) in enumerate(classes.items()):
    row, col = idx // 3, idx % 3
    img_path = f'pipe_dataset/train/{class_name}/{class_name}_0.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'{class_name}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('pipe_dataset_samples.png', dpi=150)
plt.show()