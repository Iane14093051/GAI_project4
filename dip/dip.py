# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, AveragePooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from torchvision.utils import save_image
import os

# Define normalization functions
def normalize(image):
    return (image / 255.0 - 0.5) * 2

def to_image(normalized_image):
    return ((normalized_image / 2 + 0.5) * 255).astype(np.uint8)

# Read and add noise to the image
im = cv2.imread('img-prior-in/dog.jpg')[:, :, ::-1]  # BGR to RGB
im = cv2.resize(im, (128, 128))
noise_intensity = 50
noise = np.random.randint(-noise_intensity, noise_intensity, size=im.shape)
im_noise = (im + noise).clip(0, 255).astype(np.uint8)

# Define the autoencoder model using functional API
def deep_image_prior_model():
    encoding_size = 128

    input_img = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoding_size, activation='tanh')(x)

    x = Dense(8 * 8 * 128, activation='relu')(encoded)
    x = Reshape((8, 8, 128))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return autoencoder

# Initialize random image and show
x = np.random.random(size=((1,) + im.shape)) * 2 - 1
y = normalize(im_noise[None, :])

# Train the model
model = deep_image_prior_model()
iterations = 30  # in hundreds
results = []
plt.figure(figsize=(16, 8))
for i in range(iterations):
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    output = model.predict(x)
    results.append(output[0])
    plt.axis('off')
    plt.title('Iteration ' + str((i + 1) * 100))
    plt.imshow(to_image(output[0]))


# Display results
plt.figure(figsize=(16, 6))
plt.subplot(151)
plt.axis('off')
plt.title('Ground Truth')
plt.imshow(im)
i = 0
plt.subplot(152)
plt.axis('off')
plt.title('Rectified (Iteration ' + str((i + 1) * 100) + ')')
plt.imshow(to_image(results[i]))
plt.show()
i = 14
plt.subplot(153)
plt.axis('off')
plt.title('Rectified (Iteration ' + str((i + 1) * 100) + ')')
plt.imshow(to_image(results[i]))
plt.show()
i = 29
plt.subplot(153)
plt.axis('off')
plt.title('Rectified (Iteration ' + str((i + 1) * 100) + ')')
plt.imshow(to_image(results[i]))
plt.show()

# Save images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the final rectified image and intermediate results
cv2.imwrite(os.path.join(output_dir, 'ground_truth.jpg'), im[:, :, ::-1])  # RGB to BGR
cv2.imwrite(os.path.join(output_dir, 'noisy_image.jpg'), im_noise[:, :, ::-1])  # RGB to BGR
cv2.imwrite(os.path.join(output_dir, f'rectified_{(14 + 1) * 100}.jpg'), to_image(results[14])[:, :, ::-1])  # RGB to BGR
cv2.imwrite(os.path.join(output_dir, f'rectified_{(29 + 1) * 100}.jpg'), to_image(results[29])[:, :, ::-1])  # RGB to BGR
