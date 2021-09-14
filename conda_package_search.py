import os
import shutil

def main():
    filename = "conda_package_list.txt"
    dest_folder = 'pkgs'
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    with open(filename, "r") as f:
        for line in f:
            name_list = line.strip('\n').split(' ')
            name, path = name_list[0], name_list[-1]
            basename = path.rsplit(':', 1)[1]
            foldername = os.path.join('/home/tiger/miniconda3/pkgs', basename)
            conda_filename = os.path.join('/home/tiger/miniconda3/pkgs', basename+'.conda')
            #print(foldername, os.path.exists(foldername))
            if os.path.exists(foldername):
                #print('moving files...')
                #shutil.copytree(foldername, dest_folder)
                #os.system("cp -R %s %s"%(foldername, dest_folder))
                #print('done!')
                pass
            else:
                print(foldername, os.path.exists(foldername))
            #print(conda_filename, os.path.exists(conda_filename))
            if os.path.exists(conda_filename):
                #print('moving files...')
                #shutil.copy(conda_filename, dest_folder)
                #os.system("cp %s %s"%(conda_filename, dest_folder))
                #print('done!')
                pass
            else:
                print(conda_filename, os.path.exists(conda_filename))

if __name__ == '__main__':
    main()