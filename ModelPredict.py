import random
import numpy as np
import matplotlib.pyplot as plt


def predict_and_display(model, input_data1, input_data2, output_data):
    random_index = random.randint(0, input_data1.shape[0] - 1)
    input_image1 = input_data1[random_index]
    input_image2 = input_data2[random_index]
    expected_output = output_data[random_index]

    input_image1_expanded = np.expand_dims(input_image1, axis=0)
    input_image2_expanded = np.expand_dims(input_image2, axis=0)

    combined_input = (input_image1_expanded, input_image2_expanded)

    predicted_image = model.predict(combined_input)
    predicted_image = np.squeeze(predicted_image)

    expected_output_squeezed = np.squeeze(expected_output)

    error_image = predicted_image - expected_output_squeezed

    plt.figure(figsize=(24, 5))

    plt.subplot(1, 4, 1)
    plt.title("Noisy Image")
    img1 = plt.imshow(input_image1, cmap='jet', interpolation='none')
    plt.colorbar(img1)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    img2 = plt.imshow(expected_output_squeezed, cmap='jet', interpolation='none')
    plt.colorbar(img2)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Unwrapped Image")
    img3 = plt.imshow(predicted_image, cmap='jet', interpolation='none')
    plt.colorbar(img3)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Error Image")
    img4 = plt.imshow(abs(error_image), cmap='gray', interpolation='none')
    plt.colorbar(img4)
    plt.axis('off')

    plt.show()
