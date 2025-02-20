import os
import cv2




def convert_to_grayscale(input_folder, output_folder):
    files = os.listdir(input_folder)
    
    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        
        if os.path.isfile(input_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image = cv2.imread(input_path)
            kernel_size = (9, 9)

            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
            
            cv2.imwrite(output_path, blurred_image)
            print(f"{file} dönüştürüldü.")



if __name__ == "__main__":
    input_folder = "F:\\Restoresyon\\data\\mix"  
    output_folder = "F:\\Restoresyon\\data\\mix_blurred"  
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    convert_to_grayscale(input_folder, output_folder)
