import os
import shutil

# setup
ratio_train_test = 4 / 1;
sampling_interval = ratio_train_test + 1
dataset_name = 'mouse'
path_images = dataset_name
path_labels = f"{dataset_name}_labels"
num_images = sum(os.path.isfile(os.path.join(path_images, f)) for f in os.listdir(path_images))

postfix_images = '.jpg'
postfix_labels = '.txt'
dir_output_images = f'{dataset_name}_val'
dir_output_labels = f'{dataset_name}_val_labels'

# main 
print(f"total images: {num_images}")
if not os.path.exists(dir_output_images):
    os.mkdir(dir_output_images)
if not os.path.exists(dir_output_labels):
    os.mkdir(dir_output_labels)

count = 0
for file_name in os.listdir(path_images):
    if count % sampling_interval == 0:
        # move single image
        shutil.move(
            os.path.join(path_images, file_name), 
            os.path.join(dir_output_images, file_name)
        )
        # move single label if it exists
        output_label_name = file_name.replace(postfix_images, postfix_labels)
        path_single_label = os.path.join(
          path_labels, 
          output_label_name
        )
        if os.path.exists(path_single_label):
          shutil.move(
            path_single_label, 
            os.path.join(dir_output_labels, output_label_name)
          )
    count += 1
