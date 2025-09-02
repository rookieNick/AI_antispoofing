# Import the necessary libraries
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import os
from controller.image_controller import get_prediction_label

# Global variables
cap = None
photo = None
recognition_active = False
webcam_active = False
selected_image = None

# Model loading is handled by the backend predictor module

# Function to start the webcam
def start_webcam():
    global cap, webcam_active
    if not webcam_active:
        cap = cv2.VideoCapture(0)  # Open the default webcam (0)
        if not cap.isOpened():
            result_output.insert(tk.END, "Error: Could not open webcam.\n")
            return
        webcam_active = True
        update_frame()
        result_output.insert(tk.END, "Webcam started.\n")

# Function to stop the webcam
def stop_webcam():
    global cap, webcam_active
    if webcam_active:
        cap.release()  # Release the webcam resource
        webcam_active = False
        webcam_label.config(image='')
        result_output.insert(tk.END, "Webcam stopped.\n")

# Function to update the webcam frame
def update_frame():
    global cap, photo
    if webcam_active and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert image from OpenCV BGR format to RGB format
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            # Resize image to fit the frame, maintaining aspect ratio
            img.thumbnail((480, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            webcam_label.config(image=photo)
            webcam_label.image = photo
            root.after(10, update_frame)  # Call update_frame again after 10ms
        else:
            result_output.insert(tk.END, "Error: Could not read frame from webcam.\n")
    elif not webcam_active:
        webcam_label.config(image='')

# Function to start recognition
def start_recognition():
    global recognition_active
    if not recognition_active:
        recognition_active = True
        result_output.insert(tk.END, "Recognition started.\n")
        # Placeholder for actual recognition logic
        # You would integrate your anti-spoofing model here
    else:
        result_output.insert(tk.END, "Recognition is already active.\n")

# Function to stop recognition
def stop_recognition():
    global recognition_active
    if recognition_active:
        recognition_active = False
        result_output.insert(tk.END, "Recognition stopped.\n")
    else:
        result_output.insert(tk.END, "Recognition is not active.\n")

# Function to import an image
def import_image():
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All Files", "*")]
    )
    if file_path:
        result_output.insert(tk.END, f"Imported image: {file_path}\n")
        try:
            img = Image.open(file_path)
            img.thumbnail((200, 200), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_canvas.delete("all")
            image_canvas.create_image(100, 100, image=img_tk)
            image_canvas.image = img_tk
            # Store the image for prediction
            global selected_image
            selected_image = img
        except Exception as e:
            result_output.insert(tk.END, f"Error loading image: {e}\n")

def predict_image_with_selected_model():
    global selected_image, model_var
    if selected_image is None:
        result_output.insert(tk.END, "No image selected. Please import an image first.\n")
        return
    model_type = model_var.get()
    if model_type == "CNN":
        # Use CNN model (cnn_pytorch.pth)
        from backend import predictor
        label_id, confidence = predictor.predict_image(selected_image)
        label = "Live" if label_id == 0 else "Spoof"
        result_output.insert(tk.END, f"Prediction (CNN): {label} | Confidence: {confidence:.2f}\n")
    elif model_type == "CDCN":
        result_output.insert(tk.END, "CDCN model prediction not implemented yet.\n")
    elif model_type == "VIT":
        result_output.insert(tk.END, "VIT model prediction not implemented yet.\n")
    else:
        result_output.insert(tk.END, f"Unknown model type: {model_type}\n")


def start_webcam_and_recognition():
    start_webcam()
    start_recognition()


def launch_ui():
    global root, result_output, webcam_label, image_canvas, start_combined_button, import_image_button, model_var, selected_image
    root = tk.Tk()
    root.title("Anti-Spoofing UI")
    root.geometry("1200x700")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')

    # Main frames
    left_frame = ttk.Frame(root, padding="10")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = ttk.Frame(root, padding="10")
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    # Webcam Display Frame (Left, larger)
    webcam_display_frame = ttk.LabelFrame(left_frame, text="Webcam Feed", padding="10")
    webcam_display_frame.pack(fill=tk.BOTH, expand=True)

    webcam_label = ttk.Label(webcam_display_frame, background="black")
    webcam_label.pack(fill=tk.BOTH, expand=True)

    # Combined Start Button below webcam
    start_combined_button = ttk.Button(left_frame, text="Start Webcam & Recognition", command=start_webcam_and_recognition)
    start_combined_button.pack(fill=tk.X, pady=15)

    # Imported Image Display Frame (Right, fixed size)
    image_display_frame = ttk.LabelFrame(right_frame, text="Imported Image (max 200x200)", padding="10")
    image_display_frame.pack(fill=tk.X, pady=10)

    image_canvas = tk.Canvas(image_display_frame, width=200, height=200, bg="gray", highlightthickness=0)
    image_canvas.pack()

    # Import Image Button above image display
    import_image_button = ttk.Button(right_frame, text="Import Image", command=import_image)
    import_image_button.pack(fill=tk.X, pady=5)

    # Model Selection Frame below image display
    model_frame = ttk.LabelFrame(right_frame, text="Select Model", padding="10")
    model_frame.pack(fill=tk.X, pady=5)
    model_var = tk.StringVar(value="CNN")
    model_dropdown = ttk.Combobox(model_frame, textvariable=model_var, state="readonly", values=["CNN", "CDCN", "VIT"])
    model_dropdown.pack(fill=tk.X)

    # Predict Button
    predict_button = ttk.Button(model_frame, text="Predict", command=predict_image_with_selected_model)
    predict_button.pack(fill=tk.X, pady=5)

    # Output Box Frame (Right)
    output_frame = ttk.LabelFrame(right_frame, text="Anti-Spoofing Results", padding="10")
    output_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    result_output = tk.Text(output_frame, height=15, state='normal', wrap='word', font=("Consolas", 10))
    result_output.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(result_output, command=result_output.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_output.config(yscrollcommand=scrollbar.set)

    root.mainloop()

# Only run launch_ui() if this file is executed directly
if __name__ == "__main__":
    launch_ui()
