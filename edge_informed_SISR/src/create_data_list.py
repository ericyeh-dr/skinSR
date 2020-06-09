import os
import json


def create_data_list(train_folder, test_folder, eval_folder, output_folder):
    print("Creating......")

    train_images = []
    for img in os.listdir(train_folder):
        img_path = os.path.join(train_folder, img)
        train_images.append(img_path)
    print("There are " + str(len(train_images)) + " images in the train_folder")
    with open(os.path.join(output_folder, "train_images.json"), "w") as j:
        json.dump(train_images, j)

    test_images = []
    for img in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img)
        test_images.append(img_path)
    print("There are " + str(len(test_images)) + " images in the test_folder")
    with open(os.path.join(output_folder, "test_images.json"), "w") as j:
        json.dump(test_images, j)

    eval_images = []
    for img in os.listdir(eval_folder):
        img_path = os.path.join(eval_folder, img)
        eval_images.append(img_path)
    print("There are " + str(len(eval_images)) + " images in the eval_folder")
    with open(os.path.join(output_folder, "eval_images.json"), "w") as j:
        json.dump(eval_images, j)

