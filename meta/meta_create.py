import os

paths = '/home/sky/code/python2/DCCM/venv/data/[3]Caltech-256/[X]/task03'
f = open('/home/sky/code/python2/DCCM/venv/meta/task03_meta.txt', 'a')
folderlist1 = os.listdir(paths)
folderlist1.sort()
i = 0
for folder1 in folderlist1:
    filenames = os.listdir(os.path.join(paths, folder1))
    for filename in filenames:
        out_path = folder1 + '/' + filename + ' ' + str(i)
        print(out_path)
        f.write(out_path + '\n')
    i = i + 1
f.close()
