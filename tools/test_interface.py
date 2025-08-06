import os
import base64
import json
import requests
import cv2
import numpy as np
from pathlib import Path

class ImageProcessor:
    def __init__(self, input_folder="input", output_folder="output", server_url="http://localhost:8000/detect"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.server_url = server_url
        
        # Create folders if not exist
        Path(self.input_folder).mkdir(exist_ok=True)
        Path(self.output_folder).mkdir(exist_ok=True)
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png'}
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')
    
    def send_to_server(self, base64_image, image_name):
        """Send base64 image to server and return response"""
        payload = {
            "image": base64_image,
            "filename": image_name
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            print(f"Sending {image_name} to server...")
            response = requests.post(self.server_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending request for {image_name}: {e}")
            return None
    
    def parse_results(self, response_data):
        """Parse results from server response"""
        if not response_data:
            return []
        
        if response_data.get("errno") != 0:
            print(f"Server error: {response_data.get('err_msg', 'Unknown error')}")
            return []
        
        results = response_data.get("results", [])
        detections = []
        
        for result in results:
            if "location" in result:
                detection = {
                    "score": result.get("score", 0.0),
                    "location": result["location"]
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections_on_image(self, image_path, detections, output_path):
        """Draw detection boxes on image and save"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return False
        
        height, width = image.shape[:2]
        
        for idx, detection in enumerate(detections):
            location = detection["location"]
            score = detection["score"]
            
            # Convert normalized coordinates to pixel coordinates
            # location contains: left, top, width, height (all normalized 0-1)
            left = int(location["left"] * width)
            top = int(location["top"] * height)
            box_width = int(location["width"] * width)
            box_height = int(location["height"] * height)
            
            # Calculate bottom-right corner
            right = left + box_width
            bottom = top + box_height
            
            # Ensure coordinates are within image bounds
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(0, min(right, width))
            bottom = max(0, min(bottom, height))
            
            # Choose color based on index
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[idx % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Draw score label
            label = f"Score: {score:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            label_bg_top = max(0, top - label_size[1] - 10)
            label_bg_bottom = top
            label_bg_right = min(width, left + label_size[0] + 10)
            
            cv2.rectangle(image, (left, label_bg_top), (label_bg_right, label_bg_bottom), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (left + 5, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Saved result to: {output_path}")
        else:
            print(f"Failed to save result to: {output_path}")
        
        return success
    
    def process_single_image(self, image_path):
        """Process a single image"""
        image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_name)[0]
        output_path = os.path.join(self.output_folder, f"{name_without_ext}_result.jpg")
        
        print(f"\n--- Processing: {image_name} ---")
        
        # Convert to base64
        try:
            base64_image = self.image_to_base64(image_path)
            print(f"Converted to base64, size: {len(base64_image)} characters")
        except Exception as e:
            print(f"Error converting {image_name} to base64: {e}")
            return False
        
        # Send to server
        response_data = self.send_to_server(base64_image, image_name)
        if not response_data:
            return False
        
        # Parse results
        detections = self.parse_results(response_data)
        print(f"Found {len(detections)} detection(s)")
        
        # Draw detections and save
        if detections:
            success = self.draw_detections_on_image(image_path, detections, output_path)
            if success:
                print(f"Successfully processed {image_name}")
                return True
        else:
            # No detections, just copy original image
            import shutil
            shutil.copy2(image_path, output_path)
            print(f"No detections found, saved original image to {output_path}")
            return True
        
        return False
    
    def process_all_images(self):
        """Process all images in input folder"""
        if not os.path.exists(self.input_folder):
            print(f"Input folder '{self.input_folder}' does not exist!")
            return
        
        image_files = []
        for file_name in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file_name)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_name)[1].lower()
                if ext in self.supported_extensions:
                    image_files.append(file_path)
        
        if not image_files:
            print(f"No supported image files found in '{self.input_folder}'")
            print(f"Supported extensions: {', '.join(self.supported_extensions)}")
            return
        
        print(f"Found {len(image_files)} image(s) to process")
        print(f"Server URL: {self.server_url}")
        print(f"Output folder: {self.output_folder}")
        
        successful = 0
        failed = 0
        
        for image_path in image_files:
            try:
                if self.process_single_image(image_path):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Unexpected error processing {image_path}: {e}")
                failed += 1
        
        print(f"\n=== Processing Complete ===")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(image_files)}")

def main():
    # Configuration
    INPUT_FOLDER = "input"
    OUTPUT_FOLDER = "output" 
    SERVER_URL = "http://localhost:6006/tshirt"  # Change this to your server URL
    
    # Create processor and run
    processor = ImageProcessor(INPUT_FOLDER, OUTPUT_FOLDER, SERVER_URL)
    processor.process_all_images()

if __name__ == "__main__":
    main()