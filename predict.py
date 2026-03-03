# import cv2
# import numpy as np
# import tensorflow as tf

# # Load trained model
# model = tf.keras.models.load_model("digit_model.h5")

# def predict_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#     if img is None:
#         print("❌ Not a valid image file")
#         return

#     # Resize
# img = cv2.resize(img, (28, 28))

# # Invert colors if background is white
# if np.mean(img) > 127:
#     img = 255 - img

# # Normalize
# img = img / 255.0
# img = img.reshape(1, 28, 28, 1)

#     # Predict
#     prediction = model.predict(img, verbose=0)
#     confidence = np.max(prediction)
#     digit = np.argmax(prediction)

#     # Decision
#     if confidence < 0.6:
#         print("⚠️ Not a number")
#     else:
#         print(f"✅ Predicted digit: {digit} (confidence {confidence:.2f})")

# # 🔹 CHANGE this filename if needed
# predict_image("test.png")
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Not a valid image file")
        return

    # Resize
    img = cv2.resize(img, (28, 28))

    # Invert colors if background is white
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    digit = np.argmax(prediction)

    # Decision
    if confidence < 0.6:
        print("⚠️ Not a number")
    else:
        print(f"✅ Predicted digit: {digit} (confidence {confidence:.2f})")

# 🔹 CHANGE this filename if needed
predict_image("test.png")