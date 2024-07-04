import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

img_width = 180
img_height = 180

data_train_path = 'C:/Users/Anjil/Documents/ProductClassification/large_data_split/train'
data_test_path = 'C:/Users/Anjil/Documents/ProductClassification/large_data_split/test'
data_val_path = 'C:/Users/Anjil/Documents/ProductClassification/large_data_split/validation'

data_train = tf.keras.utils.image_dataset_from_directory(data_train_path, shuffle=True, image_size=(img_height, img_width), batch_size=32, validation_split=False)

model = load_model('C:/Users/Anjil/Documents/ProductClassification/model/modelE50.h5')

def classify_image(img):
    img = tf.image.resize(img, (img_height, img_width))
    img_bat = tf.expand_dims(img, 0)

    data_cat = data_train.class_names

    
    # Calculate probabilities
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    class_name = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    result_text = "{}: {:.2f}%".format(class_name, confidence)

    return result_text

def main():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        is_frame_captured, frame = video_capture.read()
        if not is_frame_captured:
            print("Error: Could not read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
        
        classification_result = classify_image(frame_tensor)
        print(classification_result)

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:  
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
