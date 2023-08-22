import cv2
import os
import numpy as np

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

    return extracted_words

image_path = 'test5.png'  
image = cv2.imread(image_path)

lines = separate_lines(image)

words = []
for line in lines[:-1]:
    word = extract_words(line)
    words.append(word)
# words = extract_words(image)
# words = words[0][1:]

output_folder = "word_output/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for lineidx, line in enumerate(words):
    for wordidx, word in enumerate(line[1:]):
        word_output_path = os.path.join(output_folder, f"line_{lineidx}_word_{wordidx + 1}.png")
        # word = np.uint8(word)
        cv2.imwrite(word_output_path, word)
