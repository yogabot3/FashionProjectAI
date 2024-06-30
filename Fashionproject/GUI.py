import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
np.random.seed(0)
# Assuming your network implementation and necessary classes are in Train.py
from Main import Network, ConvolutionalLayer, Activation, Reshape, DenseLayer

# Define the network structure (as in your Train.py)
network = [
    ConvolutionalLayer((1, 28, 28), 3, 5, 0.01, "Convolutional1"),
    Activation("relu"),
    Reshape((5, 26, 26), (np.prod((5, 26, 26)), 1)),
    DenseLayer(np.prod((5, 26, 26)), 20, 0.01, "Dense1"),
    Activation("relu"),
    DenseLayer(20, 10, 0.01, "Dense2"),
    Activation("softmax"),
]
net = Network(network)

# Mapping from numeric labels to category names in Hebrew
category_names = [
    "Shirt", "Pants", "Sweater", "Dress", "Coat", 
    "Sandal", "Shirt", "Shoes", "Bag", "Ankle Boot"
]

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        master.title("Fashion-MNIST Prediction")
        master.geometry("500x700")
        master.configure(bg="#F8BBD0")  # Light pink background

        self.style_label = {
            "font": ("Comic Sans MS", 16),  # Girlish font
            "bg": "#F8BBD0",  # Light pink background
            "fg": "#880E4F",  # Darker pink text
            "pady": 10,
        }
        
        self.style_button = {
            "font": ("Comic Sans MS", 14),  # Girlish font
            "bg": "#E91E63",  # Pink button background
            "fg": "#FFFFFF",  # White text
            "activebackground": "#F06292",  # Lighter pink when active
            "activeforeground": "#FFFFFF",  # White text when active
            "bd": 0,
            "padx": 20,
            "pady": 10,
            "highlightthickness": 0,
        }

        self.label = tk.Label(master, text="Upload an image for prediction", **self.style_label)
        self.label.pack(pady=20)

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image, **self.style_button)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(master, bg="#F8BBD0")
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict, **self.style_button)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Comic Sans MS", 14), bg="#F8BBD0", fg="#880E4F")
        self.result_label.pack(pady=20)

        self.net = net
        self.net.load_parameters('C:/Users/yogev/Desktop/Fashionproject/parameters')  # Replace with the correct path

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def display_image(self, file_path):
        gray_image = Image.open(file_path).convert('L')
        gray_image_resized = gray_image.resize((200, 200), Image.Resampling.LANCZOS)
        gray_photo = ImageTk.PhotoImage(gray_image_resized)
        self.image_label.config(image=gray_photo)
        self.image_label.image = gray_photo

    def preprocess_image(self, file_path):
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values
        image = image.reshape(1, 28, 28)
        return image

    def predict(self):
        if hasattr(self, 'image_path'):
            image = self.preprocess_image(self.image_path)
            prediction = self.net.forward_propagation(image)
            predicted_label = np.argmax(prediction)
            category_name = category_names[predicted_label]
            self.result_label.config(text=f"The image shows a {category_name}.")
        else:
            messagebox.showerror("Error", "Please upload an image first")

if __name__ == "__main__":
    root = tk.Tk()
    gui = NeuralNetworkGUI(root)
    root.mainloop()
