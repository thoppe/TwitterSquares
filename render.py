import glob
import os
import shutil
import random
from unidecode import unidecode
import sys
import cv2

total_images = 200

name = sys.argv[1]
save_dest = "tsne/images/{}".format(name)
load_dest = "data/profile_image/{}".format(name)
subimage_dest = "data/processed_image/{}".format(name)

if not os.path.exists(save_dest):
    os.system('mkdir -p "{}"'.format(save_dest))

if not os.path.exists(subimage_dest):
    os.system('mkdir -p "{}"'.format(subimage_dest))

F_IN = sorted(glob.glob(os.path.join(load_dest, '*')))
F_IN = [x for x in F_IN if os.stat(x).st_size>0]
random.shuffle(F_IN)

for f0 in tqdm(F_IN):
    f1 = os.path.join(subimage_dest, os.path.basename(f0)) + '.jpg'
    if os.path.exists(f1):
        continue
    
    img = cv2.imread(f0)
    x,y,c = img.shape

    if x > y:
        dx = (x - y)//2
        img = img[dx:dx+y, :, :]
    if y > x:
        dy = y - x
        img = img[:, dy:dy+x, :]

    img = cv2.resize(img, (224,224))
    x,y,c = img.shapels
    assert(x==y==224)

    cv2.imwrite(f1, img)
    print ("Saved", f1)

exit()

for f_img in F_IN:
    f_base = os.path.basename(f_img)
    shutil.copyfile(f_img, os.path.join(save_dest, f_base))

n_found_files = len(glob.glob(os.path.join(save_dest, '*')))

n_size =  (math.floor(n_found_files**0.5))
n_size = min(12, n_size)

print (name, len(F_IN), n_size)

cmd = "python grid.py -s {} -d '{}' -p tsne/ --name '{}.jpg'".format(
    n_size, save_dest, name)

os.system(cmd)

os.system('eog "tsne/{}.jpg"'.format(name))
