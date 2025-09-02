import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# ---------------------- CONFIG ----------------------
# Multiple input folders
input_folders = [
    r"D:\smart script recognizer\formsA-D",
    r"D:\smart script recognizer\formsE-H",
    r"D:\smart script recognizer\formsI-Z"
]
preprocessed_folder = r"D:\smart script recognizer\preprocessed"
linesegmented_folder = r"D:\smart script recognizer\linesegmented"
ground_truth_base_folder = r"D:\smart script recognizer\lines"  # Base GT folder, can be adjusted dynamically

min_line_height = 20  # minimum height of line to avoid noise

# ----------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(path, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

# ------------------ PREPROCESSING ------------------
def preprocess_image(img):
    """Crop, grayscale, binary inverse."""
    img_cropped = img[720:2780, 100:3000]  # Cropping area can be changed per dataset
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return img_cropped, gray, binary_inv

# ------------------ LINE SEGMENTATION ------------------
def segment_lines_hpp(binary_img, min_line_height=20):
    """
    Segment lines using Horizontal Projection Profile (HPP)
    Returns: list of (y1, y2) for line positions
    """
    h_proj = np.sum(binary_img, axis=1)
    lines = []
    in_line = False
    start = 0
    
    for y, val in enumerate(h_proj):
        if val > 0 and not in_line:
            in_line = True
            start = y
        elif val == 0 and in_line:
            in_line = False
            end = y
            if (end - start) >= min_line_height:
                lines.append((start, end))
    # Handle last line
    if in_line:
        end = len(h_proj)-1
        if (end - start) >= min_line_height:
            lines.append((start, end))
    return lines

def save_segmented_lines(img, binary_img, base_name, output_folder):
    """Segment lines and save images"""
    lines_pos = segment_lines_hpp(binary_img, min_line_height)
    line_images = []
    
    for idx, (y1, y2) in enumerate(lines_pos):
        line_img = img[y1:y2, :]
        line_images.append(line_img)
        
        line_path = os.path.join(output_folder, base_name, f"{base_name}-{idx:03d}.png")
        save_image(line_path, line_img)
    
    return line_images

# ------------------ ACCURACY CHECK ------------------
def check_accuracy(line_images, gt_folder, base_name):
    """Compare segmented lines with ground truth using SSIM"""
    ssim_scores = []
    for idx, line_img in enumerate(line_images):
        gt_path = os.path.join(gt_folder, base_name, f"{base_name}-{idx:03d}.png")
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            line_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
            score = ssim(gt_img, line_gray)
            ssim_scores.append(score)
        else:
            print(f"GT not found for line {idx}: {gt_path}")
    if ssim_scores:
        avg_score = sum(ssim_scores) / len(ssim_scores)
        print(f"Average SSIM accuracy for {base_name}: {avg_score:.4f}")
    else:
        print(f"No ground truth lines found for {base_name}")

# ------------------ MAIN PIPELINE ------------------
def process_folder(input_folder, ground_truth_base_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".png"):
            filepath = os.path.join(input_folder, file)
            base_name = os.path.splitext(file)[0]
            img = cv2.imread(filepath)
            if img is None:
                print(f"Could not read {filepath}")
                continue
            
            # Preprocessing
            img_cropped, gray, binary_inv = preprocess_image(img)
            
            # Save preprocessed images
            save_image(os.path.join(preprocessed_folder, base_name, f"{base_name}_cropped.png"), img_cropped)
            save_image(os.path.join(preprocessed_folder, base_name, f"{base_name}_gray.png"), gray)
            save_image(os.path.join(preprocessed_folder, base_name, f"{base_name}_binary.png"), binary_inv)
            
            # Segment lines and save
            line_images = save_segmented_lines(img_cropped, binary_inv, base_name, linesegmented_folder)
            
            # Use ground truth folder dynamically
            gt_folder_dynamic = ground_truth_base_folder  # You can enhance mapping logic here if needed
            
            # Check accuracy dynamically
            check_accuracy(line_images, gt_folder_dynamic, base_name)

# ------------------ RUN ------------------
for folder in input_folders:
    print(f"Processing folder: {folder}")
    process_folder(folder, ground_truth_base_folder)
