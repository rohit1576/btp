import os
path = '/Users/kanishkverma/Desktop/btp/dataset/volume_up'
files = os.listdir(path)
i = 0
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path,'volume_up'+str(i)+'.png'))
    i = i+1