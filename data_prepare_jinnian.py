import glob
import os
import csv

def untar():
    data_folder = "datasets/train"
    for filepath in glob.glob(os.path.join(data_folder, "*.tar")):
        folder = filepath.rsplit('.', 1)[0]
        filename = os.path.basename(filepath)
        os.system("mkdir %s"%folder)
        os.system("mv %s %s"%(filepath, folder))
        os.system("tar -xvf %s"%os.path.join(folder, filename))
        os.system("mv *.JPEG %s"%folder)
        os.system("rm %s"%os.path.join(folder, filename))

def create_train_txt():
    data_folder = "datasets/train/"
    raw_train_txt = "datasets/train_raw.txt"
    output_file = "datasets/train_map.txt"
    f = open(raw_train_txt, "r")
    train_map_dict = {}
    for line in f:
        name, label = line.split('_')
        train_map_dict[name] = int(label)
    f.close()
    array = []
    for folderpath in glob.glob(os.path.join(data_folder, "*")):
        assert(os.path.isdir(folderpath))
        for filepath in glob.glob(os.path.join(folderpath, "*")):
            filename = os.path.basename(filepath)
            string = filename.split('_')[0]
            array.append("%s\t%d\n"%(os.path.join(string, filename), train_map_dict[string]-1))
    o = open(output_file, "w")
    o.writelines(array)
    o.close()
    print('done!')

def create_val_txt():
    f = open("datasets/ILSVRC2012_validation_ground_truth.txt", "r")
    array = []
    count = 0
    for line in f:
        count += 1
        imagename = "ILSVRC2012_val_%08d.JPEG"%count
        array.append("%s\t%d\n"%(imagename, int(line)-1))
    o = open("datasets/val_map.txt", "w")
    o.writelines(array)
    o.close()
    print('done!')

def move_validation_to_folders():
    raw_train_txt = "datasets/train_raw.txt"
    f = open(raw_train_txt, "r")
    train_map_dict = {}
    for line in f:
        name, label = line.split('_')
        train_map_dict[int(label)] = name
    f.close()

    f = open("datasets/ILSVRC2012_validation_ground_truth.txt", "r")
    count = 0
    for line in f:
        count += 1
        imagename = "ILSVRC2012_val_%08d.JPEG"%count
        labelname = train_map_dict[int(line)]
        folderpath = os.path.join('datasets/val', labelname)
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        os.system("mv %s %s"%(os.path.join('datasets/val', imagename), folderpath))
    f.close()
    print('done!')


def main():
    #create_train_txt()
    #create_val_txt()
    move_validation_to_folders()

if __name__ == '__main__':
    main()