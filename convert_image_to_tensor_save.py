
import pickle
import os
from PIL import Image
from torchvision import transforms
import numpy as np

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret
if __name__ == "__main__":
    
    opt = {
        'data_path':'input/prepared/',
        'img_path':'dataset_image/'
    }
    id ={
        "train":load_file(opt["data_path"] + "train_id"),
        "test":load_file(opt["data_path"] + "test_id"),
        "valid":load_file(opt["data_path"] + "valid_id")
    }
    img_dir=opt["img_path"] 

    transform_train = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform = {
        "train": transform_train,
        "valid": transform_valid,
        "test": transform_test
        }
    image_tensor = {
        "train": [],
        "valid": [],
        "test": []
    }
    save_path = 'image_tensor/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for mode in id.keys():
        for idx in id[mode]:
            img_path=os.path.join(
                    img_dir,
                    "{}.jpg".format(idx)
                )
            img = Image.open(img_path)
            img = img.convert('RGB') # convert grey picture
            trainsform_img = transform[mode](img)
            image_tensor[mode].append(trainsform_img.unsqueeze(0))
            np.save(save_path + str(idx) + '.npy', trainsform_img.numpy())
            


    