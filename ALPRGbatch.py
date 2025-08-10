import cv2
from fast_alpr import ALPR
import tkinter as tk
from tkinter import filedialog
import os
import csv

# --- Directory Selection Dialog ---
# We don't need the main tkinter window, so we hide it.
root = tk.Tk()
root.withdraw()

# Open a dialog to ask the user to select a directory.
directory_path = filedialog.askdirectory(
    title="Select a Folder of Images for ALPR"
)

# --- ALPR Processing ---
# Proceed only if the user selected a directory.
if directory_path:
    print(f"Directory selected: {directory_path}")
    print("Initializing ALPR models... This might take a moment.")

    # Initialize the ALPR once before the loop for efficiency.
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )

    # --- Setup CSV and Output Directory ---
    csv_path = os.path.join(directory_path, "alpr_results.csv")
    output_dir = os.path.join(directory_path, "annotated_output")
    os.makedirs(output_dir, exist_ok=True) # Create output folder if it doesn't exist

    # Find all image files in the selected directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files (.png, .jpg, .jpeg) found in the selected directory.")
    else:
        # Open the CSV file to write the results
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the header row for the CSV
            csv_writer.writerow(['filename', 'plate_text', 'confidence', 'box_x1', 'box_y1', 'box_x2', 'box_y2'])
            print(f"Processing {len(image_files)} images. Results will be saved to {csv_path}")

            # --- Loop Through All Images ---
            for i, image_name in enumerate(image_files):
                image_path = os.path.join(directory_path, image_name)
                print(f"  ({i+1}/{len(image_files)}) Processing: {image_name}")

                # Load the image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"    - Warning: Could not load image, skipping.")
                    continue

                # --- Get Raw Prediction Data ---
                predictions = alpr.predict(frame)

                # --- Write Data to CSV ---
                if not predictions:
                    print("    - No license plates found.")
                else:
                    for pred in predictions:
                        # FIX: Access the final correct attributes nested within the objects.
                        detection = pred.detection
                        ocr = pred.ocr
                        bounding_box = detection.bounding_box
                        
                        csv_writer.writerow([
                            image_name,
                            ocr.text,
                            f"{ocr.confidence:.4f}",
                            bounding_box.x1,
                            bounding_box.y1,
                            bounding_box.x2,
                            bounding_box.y2
                        ])
                    print(f"    - Found {len(predictions)} plate(s).")

                # --- Save Annotated Image ---
                annotated_frame = alpr.draw_predictions(frame)
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, annotated_frame)

        print("\nProcessing complete!")
        print(f"CSV data saved to: {csv_path}")
        print(f"Annotated images saved in: {output_dir}")

else:
    # This message is shown if the user closes the dialog without selecting a folder.
    print("No directory was selected. Exiting program.")
