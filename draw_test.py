import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("digit_model.h5")

width = 200
height = 200

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")

        self.canvas = tk.Canvas(root, width=width, height=height, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (width, height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(root, text="Predict", command=self.predict).pack(pady=5)
        tk.Button(root, text="Clear", command=self.clear).pack(pady=5)

    def paint(self, event):
        x1, y1 = event.x-6, event.y-6
        x2, y2 = event.x+6, event.y+6
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (width, height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        img = self.image.resize((28, 28))
        img = np.array(img)

        # Invert if needed
        if np.mean(img) > 127:
          img = 255 - img

        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img, verbose=0)
        confidence = np.max(pred)
        digit = np.argmax(pred)

        if confidence < 0.6:
            print("⚠️ Not a number")
        else:
            print(f"✅ Digit: {digit} (confidence {confidence:.2f})")

root = tk.Tk()
app = App(root)
root.mainloop()