import cv2
import os
import numpy as np
import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image


model = tf.keras.models.load_model('character_recognition_model3.h5')
character_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l","m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def pad_image(image):
    image = np.bitwise_not(image) 
    if image is not None:
        expected =  [55, 55]
        size = image.shape[:2]

        resize_ratio = min(expected[1] / size[1], expected[0] / size[0])
        new_size = (int(size[1] * resize_ratio), int(size[0] * resize_ratio))
        resized_image = cv2.resize(image, new_size)

        shift_x = (56 - resized_image.shape[1]) // 2
        shift_y = (56 - resized_image.shape[0]) // 2
        plain_white_canvas = np.ones((56, 56), dtype=np.uint8) * 255
        plain_white_canvas[shift_y:shift_y+resized_image.shape[0], shift_x:shift_x+resized_image.shape[1]] = resized_image

        return plain_white_canvas

def preprocess_image(image):
    image = np.bitwise_not(image) 
    if image is not None:
        expected =  [55, 55]
        size = image.shape[:2]

        resize_ratio = min(expected[1] / size[1], expected[0] / size[0])
        new_size = (int(size[1] * resize_ratio), int(size[0] * resize_ratio))
        resized_image = cv2.resize(image, new_size)

        shift_x = (56 - resized_image.shape[1]) // 2
        shift_y = (56 - resized_image.shape[0]) // 2
        plain_white_canvas = np.ones((56, 56), dtype=np.uint8) * 255
        plain_white_canvas[shift_y:shift_y+resized_image.shape[0], shift_x:shift_x+resized_image.shape[1]] = resized_image
        
        preprocessed_image = plain_white_canvas.astype('float32') / 255.0
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

    _, binary = cv2.threshold(line, 130, 255, cv2.THRESH_BINARY_INV)

    col_sum = np.sum(binary, axis=0)
    
    gap_indices = np.where(col_sum <= gap_threshold)[0]

    char_indices = np.where(col_sum != 0)[0]

    avg_character_width = np.mean(np.diff(char_indices))
    avg_character_width -= 1

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

image_path = r'D:\Training\Pandas\demo\Arial-20.png'  
# image_path = 'test2.png'
image = cv2.imread(image_path)

lines = separate_lines(image)

# linepath = 'line_output/'
# if not os.path.exists(linepath):
#     os.makedirs(linepath)

# for i, line in enumerate(lines[:-1]): 
#     line_output_path = os.path.join(linepath, f"line_{i + 1}.png")
#     cv2.imwrite(line_output_path, line)

words = []
extracted_characters = []
for line in lines[:-1]:
    word = extract_words(line)
    words.append(word)

for word in words:
    for word1 in word[1:]:
        character = extract_characters(word1)
        extracted_characters.append(character[:])


output_folder = "character_output/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

predicted_characters = []
for i, character_region in enumerate(extracted_characters):
    for j, character in enumerate(character_region):
        preprocessed_character = preprocess_image(character)

        padd_image = pad_image(character)
        character_output_path = os.path.join(output_folder, f"word_{i + 1}_character_{j + 1}.png")
        cv2.imwrite(character_output_path, padd_image)

        prediction = model.predict(preprocessed_character)
        predicted_label = np.argmax(prediction)
        predicted_character = character_classes[predicted_label]
        predicted_characters.append(predicted_character)
    predicted_characters.append(" ")

recognized_text = "".join(predicted_characters)
print("Recognized Text:", recognized_text)

