#! /usr/bin/env python3

import os
import sys
import random
import string
import matplotlib.pylab as plt
from skimage.util import random_noise

image_path = './data/UTKFace_AsianOnly/'
outdir = './data/overSampling/'
AGE_LIST = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-120']
GENDER_LIST = ['Male', 'Female']
def categorize_age(age:str):
    """
    Categorize age into age label
    Arguments:
        age: age number
    Returns:
        Return age label corresponding to age number
    """
    age = int(age)
    age_idx = age//10
    if age_idx <=8:
        return age_idx
    else:
        return 8

def flip_image(img_path, outdir):
    """
    Flip the input image and save as a copy in the specific output directory
    Arguments:
        img_path: path to the image to flip
        outdir: output directory to save flipped image
    Returns:
        None
    """
    file_name = img_path.split('/')[-1]
    name, ext = os.path.splitext(file_name)
    random_char = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    img_name = os.path.join(outdir, name+'_'+random_char+'_flipH'+ext)

    img = plt.imread(img_path)
    img_flipH = img[:, ::-1]
    plt.imsave(img_name, img_flipH, format="jpg")

def noise_image(img_path, outdir):
    """
    Add noise to the input image and save as a copy in the specific output directory
    Arguments:
        img_path: path to the image to add noise
        outdir: output directory to save noised image
    Returns:
        None
    """
    file_name = img_path.split('/')[-1]
    name, ext = os.path.splitext(file_name)
    random_char = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    img_name = os.path.join(outdir, name+'_'+random_char+'_noise'+ext)

    img = plt.imread(img_path)
    img_noise = random_noise(img, mode='gaussian')
    plt.imsave(img_name, img_noise, format="jpg")

def over_sampling(img_list, target:int, outdir:str):
    """
    Oversampling (increase number of sample) a class by flipping and adding noise to the original image 
    Arguments:
        img_list: list storing image file name of the class
        target: number of image in the class to increase to
        outdir: output directory to save oversampling images 
    Returns:
        None
    """
    class_len = len(img_list)
    for i in range((target-class_len)//2):
        flip_image(img_list[i%(class_len-1)], outdir)
        noise_image(img_list[i%(class_len-1)], outdir)

def main():
    if not os.path.exists('data/DEF/'):
        os.mkdir('data/DEF/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    image_filename_list = os.listdir(image_path)
    image_filename_list = [image_path+i for i in image_filename_list if i.endswith('.jpg')]
    
    random.seed(230)
    
    random.shuffle(image_filename_list)

    split_ratio = 0.8

    train_filename_list = image_filename_list[0:int(split_ratio*len(image_filename_list))]
    test_filename_list = image_filename_list[int(split_ratio*len(image_filename_list)):]

    # Create labels for training set 
    train_age_labels = []
    train_gender_labels = []
    for file in train_filename_list:
        file_name = os.path.basename(file) 
        train_age_labels.append(categorize_age(file_name.split('_')[0]))
        train_gender_labels.append(int(file_name.split('_')[1]))

    # Count number of each class image
    train_age_count = [train_age_labels.count(i) for i in range(len(AGE_LIST))]
    print("Number of image per age class:")
    print('\t'.join([AGE_LIST[i] for i in range(len(AGE_LIST))]), sep='\t')
    print('--------------------------------------------------------------')
    print('\t'.join([str(train_age_count[i]) for i in range(len(AGE_LIST))]), sep='\t')

    train_gender_count = [train_gender_labels.count(i) for i in range(len(GENDER_LIST))]
    print("Number of image per gender class:")
    print('\t'.join([GENDER_LIST[i] for i in range(len(GENDER_LIST))]), sep='\t')
    print('--------------')
    print('\t'.join([str(train_gender_count[i]) for i in range(len(GENDER_LIST))]), sep='\t')

    # Over sampling to balance training dataset for age class
    for i in range(len(AGE_LIST)): 
        class_file_list = [train_filename_list[j] for j in range(len(train_age_labels)) if train_age_labels[j]==i]
        over_sampling(class_file_list, max(train_age_count), outdir)
    
    # Merge training set list and oversampling set list
    oversampling_filename_list = os.listdir(outdir)
    oversampling_filename_list = [outdir+i for i in oversampling_filename_list if i.endswith('.jpg')]
    for filename in oversampling_filename_list:
        train_filename_list.append(filename)

    train_age_labels = []
    train_gender_labels = []
    for file in train_filename_list:
        file_name = os.path.basename(file) 
        train_age_labels.append(categorize_age(file_name.split('_')[0]))
        train_gender_labels.append(int(file_name.split('_')[1]))

    # Count number of each class image after oversampling
    print("\nAFTER OVERSAMPLING:")
    train_age_count = [train_age_labels.count(i) for i in range(len(AGE_LIST))]
    print("Number of image per age class:")
    print('\t'.join([AGE_LIST[i] for i in range(len(AGE_LIST))]), sep='\t')
    print('--------------------------------------------------------------')
    print('\t'.join([str(train_age_count[i]) for i in range(len(AGE_LIST))]), sep='\t')

    train_gender_count = [train_gender_labels.count(i) for i in range(len(GENDER_LIST))]
    print("Number of image per gender class:")
    print('\t'.join([GENDER_LIST[i] for i in range(len(GENDER_LIST))]), sep='\t')
    print('--------------')
    print('\t'.join([str(train_gender_count[i]) for i in range(len(GENDER_LIST))]), sep='\t')

    # Write DEF file Age 
    train_age_filename = "data/DEF/age_train.txt"
    if os.path.exists(train_age_filename): os.remove(train_age_filename)
    train_age_file = open(train_age_filename, "a")
    for f in train_filename_list:
        file_name = os.path.basename(f)
        label = str(categorize_age(file_name.split('_')[0]))
        train_age_file.write(f+' '+label)
        train_age_file.write("\n")

    # Write DEF file Age
    test_age_filename = "data/DEF/age_test.txt"
    if os.path.exists(test_age_filename): os.remove(test_age_filename)
    test_age_file = open(test_age_filename, "a")
    for f in test_filename_list:
        file_name = os.path.basename(f)
        label = str(categorize_age(file_name.split('_')[0]))
        test_age_file.write(f+' '+label)
        test_age_file.write("\n")

    # Write DEF file Gender 
    train_gender_filename = "data/DEF/gender_train.txt"
    if os.path.exists(train_gender_filename): os.remove(train_gender_filename)
    train_gender_file = open(train_gender_filename, "a")
    for f in train_filename_list:
        file_name = os.path.basename(f)
        label = file_name.split('_')[1]
        train_gender_file.write(f+' '+label)
        train_gender_file.write("\n")

    # Write DEF file Gender
    test_gender_filename = "data/DEF/gender_test.txt"
    if os.path.exists(test_gender_filename): os.remove(test_gender_filename)
    test_gender_file = open(test_gender_filename, "a")
    for f in test_filename_list:
        file_name = os.path.basename(f)
        label = file_name.split('_')[1]
        test_gender_file.write(f+' '+label)
        test_gender_file.write("\n")

if __name__ == "__main__":
    sys.exit(main())
