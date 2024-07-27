# Real-Time Insect Detection System

This repository contains a project for detecting and classifying insects using a deep learning model. The system can classify insects from either real-time video feed or uploaded images. The project uses TensorFlow for the deep learning model and OpenCV for handling real-time video feeds and image processing. Additionally, a graphical user interface (GUI) is implemented using Tkinter to provide a user-friendly experience for **selecting between real-time detection and image upload**.

## Table of Contents
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Real-Time Detection](#real-time-detection)
  - [Upload Image](#upload-image)
- [Dataset](#dataset)
- [Pre-trained Model](#pre-trained-model)
- [Model Building](#model-building)
- [GUI Implementation](#gui-implementation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features
- Classify insect species from images or real-time video feed.
- Uses a Convolutional Neural Network (CNN) model for classification.
- Supports re-uploading images for classification.
- User-friendly GUI for selecting between real-time detection and image upload.

## Results
![Screenshot (639)](https://github.com/user-attachments/assets/4bfb99d5-6db2-4c72-8607-049e0929ccaf)
![Screenshot (638)](https://github.com/user-attachments/assets/6fb0cdf9-48ae-45e8-b18b-1d6e46876c94)

![Screenshot (637)](https://github.com/user-attachments/assets/eaab3d35-124c-49de-8ffa-64a18146e916)
![Screenshot (640)](https://github.com/user-attachments/assets/7f2fa48b-6475-4084-8836-81ea9d931976)
![Screenshot (641)](https://github.com/user-attachments/assets/53eb96fc-a78c-4271-89eb-276064b8bed2)


## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Tkinter (for GUI)
- Matplotlib

## Installation
1. Install the required libraries:
    ```bash
    pip install tensorflow opencv-python matplotlib tk
    ```

2. Download the data and pre-trained model from **Dataset_and_Pre_Trained_Model.md** file:
    - Dataset: `data/`
    - Pre-trained Model: `insect_classifier.h5`

    These files can be accessed from the file named `Dataset_and_Pre_Trained_Model.md` and should be added to the suitable path as specified below.

3. Ensure the dataset and pre-trained model are placed in the correct directories:
    - `data/` directory should contain subfolders for each insect category (Butterfly, Dragonfly, Grasshopper, Ladybird, Mosquito).
    - The pre-trained model file (`insect_classifier.h5`) should be in the root directory of the project.

## Usage
To run the insect detection system, execute the following command:
```bash
python Insect Recognition Model Building.ipynb
```

### Real-Time Detection
1. Select "Real-Time Detection" in the GUI.
2. The system will start the webcam feed and display real-time insect classification.
3. Press 'q' to exit the real-time detection mode.

### Upload Image
1. Select "Upload Image" in the GUI.
2. Choose an image file from your system.
3. The system will classify the uploaded image and display the result.
4. You can re-upload another image if desired.

## Dataset
The dataset should be structured as follows:
```
data/
  ├── Butterfly/
  ├── Dragonfly/
  ├── Grasshopper/
  ├── Ladybird/
  └── Mosquito/
```
Each subfolder should contain images of the respective insect category.

## Pre-trained Model
The pre-trained model file (`insect_classifier.h5`) should be placed in the root directory of the project. This model can be accessed from the folder named `Links`.

## Model Building
The model used for insect classification is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. Below is an overview of the model-building process:

1. **Data Preparation**: The dataset is split into training and validation sets using `ImageDataGenerator`.
2. **Model Architecture**: The CNN model consists of multiple convolutional layers followed by max-pooling layers, flattening, and dense layers. The final layer uses a softmax activation function to classify images into one of the five insect categories.
3. **Model Compilation**: The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
4. **Model Training**: The model is trained on the training dataset and validated on the validation dataset. The training process includes multiple epochs and uses data augmentation to improve generalization.
5. **Model Saving**: The trained model is saved as `insect_classifier.h5`.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data directories
train_dir = 'data/train'
val_dir = 'data/val'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)

# Save the model
model.save('insect_classifier.h5')
```

## GUI Implementation
The GUI for the insect detection system is implemented using Tkinter. It provides options for real-time detection and image upload. Below is an overview of the GUI implementation:

1. **Main Window**: The main window provides buttons for selecting either real-time detection or image upload.
2. **Real-Time Detection**: When the "Real-Time Detection" button is clicked, the system starts the webcam feed and displays real-time insect classification.
3. **Image Upload**: When the "Upload Image" button is clicked, a file dialog is opened to select an image file. The selected image is classified, and the result is displayed. Users can re-upload images for classification.

```python
from tkinter import Tk, filedialog, Button, Label, Toplevel

# Load the trained model
model = tf.keras.models.load_model('insect_classifier.h5')

# Data generators to get class labels
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

def classify_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to read the image file.")
            return None
        img = cv2.resize(img, (150, 150)) / 255.0  # Resize the image to (150, 150)
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array)
        return np.argmax(prediction, axis=1)[0]
    except Exception as e:
        print(f"Error in classify_image: {e}")
        return None

def classify_real_time():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_array = cv2.resize(frame, (150, 150)) / 255.0  # Resize the frame to (150, 150)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        class_label = list(train_generator.class_indices.keys())[class_idx]
        
        cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Insect Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_and_classify():
    def reupload_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Selected file: {file_path}")
            class_idx = classify_image(file_path)
            if class_idx is not None:
                class_label = list(train_generator.class_indices.keys())[class_idx]
                print(f'The uploaded image is classified as: {class_label}')
                img = cv2.imread(file_path)
                cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Uploaded Image Classification', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                reupload_image()  # Allow re-uploading another image
            else:
                print("Error: Image classification failed.")
        else:
            print("Error: No file selected.")

    root = Toplevel()
    root.withdraw()
    reupload_image()

def show_gui():
    root = Tk()
    root.title("Insect Detection")

    def on_real_time_click():
        root.destroy()
        classify_real_time()

    def on_upload_click():
        upload_and_classify()

    label = Label(root, text="Choose detection mode:")
    label.pack(pady=10)

    real_time_button = Button(root, text="Real-Time Detection", command=on_real_time_click)
    real_time_button.pack(pady=10)

    upload_button = Button(root, text="Upload Image", command=on_upload_click)
    upload_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    show_gui()
```

### Explanation:
1. **Main GUI**: The `show_gui` function creates the main window with options to select either real-time detection or image upload.
2. **Real-Time Detection**: When the "Real-Time Detection" button is clicked, the webcam feed starts, and real-time insect classification is performed.
3. **Image Upload and Re-Upload**: When the "Upload Image" button is clicked, a file dialog opens to select an image. After classification, the function `reupload_image` allows for re-uploading another image, enabling continuous classification without restarting the program.

### Project Structure
```
insect-detection-system/
├── data/
│   ├── Butterfly/
│   ├── Dragonfly/
│   ├── Grasshopper/
│   ├── Ladybird/
│   └── Mosquito/
├── insect_classifier.h5
├── Insect Recognition Model Building.ipynb
├── README.md
└── Links/
```

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.


### Acknowledgements
- This project uses [TensorFlow](https://www.tensorflow.org/) for the deep learning model.
- [OpenCV](https://opencv.org/) is used for image processing and real-time video feed handling.
- Special thanks to the contributors of the open-source libraries and tools used in this project.

---

This README file provides a comprehensive overview of the project, installation steps, usage instructions, and other relevant details. The detailed description covers the model-building process, the GUI implementation, and ensures users can re-upload images for classification seamlessly.
