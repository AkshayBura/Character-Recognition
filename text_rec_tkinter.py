import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('character_recognition_model1.h5')
character_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l","m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


def preprocess_image(image):
    image = np.bitwise_not(image) 
    if image is not None:
        expected =  [56, 56]
        size = image.shape[:2]
        ratio = max(expected) / max(size)
        new_size = tuple([int(x*ratio) for x in size])
        d_h = expected[0] - new_size[0]
        d_w = expected[1] - new_size[1]
        left, right = d_w // 2, d_w - (d_w // 2)
        top, bottom = d_h // 2, d_h - (d_h // 2)
        # top = 5
        # bottom = 5
        # left = 9
        # right = 9

        # image = np.asarray(image)
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        scale_height = 56 / padded_image.shape[0]
        scale_width = 56 / padded_image.shape[1]
        # resized_image = cv2.resize(padded_image, (56, 56))
        resized_image = cv2.resize(padded_image, (0, 0), fx=scale_width, fy=scale_height)

        shift_x = (56 - resized_image.shape[1]) // 2
        shift_y = (56 - resized_image.shape[0]) // 2

        plain_white_canvas = np.ones((56, 56), dtype=np.uint8) * 255

        plain_white_canvas[shift_y:shift_y+resized_image.shape[0], shift_x:shift_x+resized_image.shape[1]] = resized_image
        
        preprocessed_image = resized_image.astype('float32') / 255.0
        # binary = cv2.threshold(array_image, 80, 255, cv2.THRESH_BINARY_INV)
        
        preprocessed_image = preprocessed_image.reshape(1, 56, 56, 1)

        return preprocessed_image

def separate_lines(image, gap_threshold=20):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_img = cv2.bitwise_not(gray_image)

    row_sum = np.sum(bw_img, axis=1)

    gap_indices = np.where(row_sum <= gap_threshold)[0]

    lines = []
    start_idx = 0
    for idx in gap_indices:
        if idx - start_idx > 1:  
            lines.append(image[start_idx:idx, :])
        start_idx = idx + 1
    lines.append(image[start_idx:, :])

    return lines

def extract_words(line):
    extracted_words = []
    
    if len(line.shape) == 3:
        line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(line, 128, 255, cv2.THRESH_BINARY_INV)

    col_sum = np.sum(binary, axis=0)
    
    char_indices = np.where(col_sum != 0)[0]

    diff = np.diff(char_indices)

    count = 0
    add = 0
    for dif in diff:
        if dif > 1:
            count += 1
            add += dif
    avg = add/count
    avg = avg + 1

    words = []
    start_char = char_indices[0]
    end_char = 0
    for i in char_indices:
        if i - end_char > avg:
            words.append(binary[:, start_char:end_char])
            start_char = i
            end_char = i
            i += 1            
        else:
            end_char = i
            i += 1   
    words.append(binary[:, start_char:end_char])          
    
    for word in words:
        extracted_words.append(word)

    return words

def extract_characters(line, gap_threshold=10):
    extracted_characters = []
    line = cv2.bitwise_not(line)

    if len(line.shape) == 3:
        line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(line, 120, 255, cv2.THRESH_BINARY_INV)

    col_sum = np.sum(binary, axis=0)
    
    gap_indices = np.where(col_sum <= gap_threshold)[0]

    char_indices = np.where(col_sum != 0)[0]

    avg_character_width = np.mean(np.diff(char_indices))

    characters = []
    start_idx = 0
    for idx in gap_indices:
        if idx - start_idx > avg_character_width:
            characters.append(binary[:, start_idx:idx])
        start_idx = idx + 1
    characters.append(binary[:, start_idx:])
    
    for char in characters:
        extracted_characters.append(char)

    return extracted_characters

def recognize_text():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path)

        image = np.array(image)

        lines = separate_lines(image)

        words = []
        
        for line in lines[:-1]:
            word = extract_words(line)
            words.append(word)

        extracted_characters = []

        for word in words:
            for word1 in word[1:]:
                character = extract_characters(word1)
                extracted_characters.append(character[:])

        predicted_characters = []
        for character_region in extracted_characters:
            for character in character_region:
                preprocessed_character = preprocess_image(character)
                prediction = model.predict(preprocessed_character)
                predicted_label = np.argmax(prediction)
                predicted_character = character_classes[predicted_label]
                predicted_characters.append(predicted_character)
            predicted_characters.append(" ")
            
        recognized_text = "".join(predicted_characters)
        
        image = cv2.resize(image, (600, 400))
        resized_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image_rgb))
        image_label.config(image=photo)
        image_label.image = photo
        
        recognized_text_label.config(text=f"Recognized Text: {recognized_text}")

root = tk.Tk()
root.title("Image Text Recognition")

browse_button = tk.Button(root, text="Select Image", command=recognize_text)
browse_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

recognized_text_label = tk.Label(root, text="")
recognized_text_label.pack(pady=10)

root.mainloop()
