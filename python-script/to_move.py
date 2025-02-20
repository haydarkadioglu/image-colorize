import os
import shutil


def move_photo(input_folder, output_folder):
    files = os.listdir(input_folder)
    
    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        
        if os.path.isfile(input_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            
            try:
                shutil.move(input_path, output_path)
                print(f"{file} taşındı.")
            except FileNotFoundError:
                print(f"'{file}' adında bir fotoğraf bulunamadı.")

            except Exception as e:
                print(f"Hata oluştu: {e}")



# Kullanım örneği:
source_folder = "F:\\Restoresyon\\data\\archive (1)\\lfw-funneled\\lfw_funneled"
destination_folder = "F:\\Restoresyon\\data\\humans"

folders = os.listdir(source_folder)

lenght = len(folders)

count = 0
for folder in folders:
    dir = os.path.join(source_folder, folder)
    move_photo(dir, destination_folder)
    count+=1
    print((count/lenght)*100)
    


