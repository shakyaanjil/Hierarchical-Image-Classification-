import cv2
import winsound
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
toy_model = load_model('C:/Users/Anjil/Documents/ProductClassification/models/imagenet-model/IMGneT_100_7_29_person.h5')
mtr_size_model = load_model('C:/Users/Anjil/Documents/ProductClassification/MTR_size_imgNet.h5')

Toy_class_names = ['MBE', 'MDR-GR', 'MLN-YL', 'MTR', 'person']
MTR_class_names = ['MMTR', 'MTR']

roi_x, roi_y, roi_width, roi_height = 50, 50, 300, 300  # (x, y, width, height)

csv_file = 'C:/Users/Anjil/Documents/ProductClassification/All-Data/yogibo_product.csv'
try:
    df = pd.read_csv(csv_file, on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error reading {csv_file}: {e}")
    exit()

df_filtered = df[df['pdt_pro_cd'].isin(Toy_class_names)]
product_mapping = {row['pdt_pro_cd']: row['pdt_nm_us'] for _, row in df_filtered.iterrows()}
class_labels = Toy_class_names

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def classify_toy(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = toy_model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    product_name = product_mapping.get(predicted_label, "Unknown")
    return predicted_label, product_name

def classify_mtr_size(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = mtr_size_model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = MTR_class_names[predicted_class]
    return predicted_label

def beep():
    winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

def is_frame_changed(fg_mask, threshold=10):
    non_zero_count = np.count_nonzero(fg_mask)
    return non_zero_count > threshold

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    fgbg.apply(roi)

print("Initialization complete !!!")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    fg_mask = fgbg.apply(roi)
    foreground_present = is_frame_changed(fg_mask)

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('\r'):  
        if not foreground_present:
            print("No item to classify")
            cv2.putText(frame, "No item to classify", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            class_code, product_name = classify_toy(roi)
            if class_code == 'person':
                print("No item to classify")
                cv2.putText(frame, "No item to classify", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                beep()  # Play beep sound
                print(f"Classification result: {class_code} - {product_name}")
                cv2.putText(frame, f"Result: {class_code} - {product_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if class_code == 'MTR':
                    mtr_size = classify_mtr_size(roi)
                    print(f"MTR size classification result: {mtr_size}")
                    cv2.putText(frame, f"MTR Size: {mtr_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show result for 3 seconds
        cv2.imshow('Video', frame)
        cv2.waitKey(3000)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
