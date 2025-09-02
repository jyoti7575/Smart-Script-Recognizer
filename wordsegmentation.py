import cv2
import os

class Word:
    def __init__(self, img, bbox):
        self.img = img
        self.bbox = bbox

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prepare_img(img_bgr, height):
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    h, w = img_gray.shape
    scale_factor = height / h
    img_scaled = cv2.resize(img_gray, (int(w * scale_factor), height))
    return img_scaled

def detect_words(img, kernel_size=25, min_area=100):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w * h > min_area:
            word_img = img[y:y+h, x:x+w]
            words.append(Word(word_img, (x, y, x+w, y+h)))
    return words

def sort_words(words):
    return sorted(words, key=lambda w: w.bbox[0])

def save_words(words, output_folder, image_name, line_num):
    base_folder = os.path.join(output_folder, image_name, f"{image_name}-{line_num:03d}")
    ensure_dir(base_folder)
    for idx, word in enumerate(words):
        word_filename = os.path.join(base_folder, f"{idx:02d}.png")
        cv2.imwrite(word_filename, word.img)
        print(f"Saved word {idx:02d} for line {line_num} in {word_filename}")

def segment_words_for_all_lines(linesegmented_base_folder, wordsegmented_base_folder):
    # Iterate over all image_name folders inside linesegmented
    all_image_name_folders = [d for d in os.listdir(linesegmented_base_folder)
                             if os.path.isdir(os.path.join(linesegmented_base_folder, d))]
    
    for image_name in all_image_name_folders:
        line_folder = os.path.join(linesegmented_base_folder, image_name)
        line_files = sorted([f for f in os.listdir(line_folder) if f.lower().endswith('.png')])
        
        for line_num, line_file in enumerate(line_files):
            line_path = os.path.join(line_folder, line_file)
            line_img_bgr = cv2.imread(line_path)
            if line_img_bgr is None:
                print(f"Could not load line image: {line_path}")
                continue
            
            # Prepare grayscale resized image for word detection
            line_img = prepare_img(line_img_bgr, 100)
            
            # Detect words
            words = detect_words(line_img, kernel_size=40, min_area=300)
            if not words:
                print(f"No words detected in line {line_num} of {image_name}")
                continue
            
            # Sort words left to right
            words_sorted = sort_words(words)
            
            # Save each word image accordingly
            save_words(words_sorted, wordsegmented_base_folder, image_name, line_num)

if __name__ == "__main__":
    linesegmented_folder = r"D:\smart script recognizer\linesegmented"
    wordsegmented_folder = r"D:\smart script recognizer\wordsegmented"
    segment_words_for_all_lines(linesegmented_folder, wordsegmented_folder)
