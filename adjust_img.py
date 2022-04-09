import cv2
import sys
import os

path = sys.argv[1]
output_path = sys.argv[2]
for filename in os.listdir(path):
    img_path = os.path.join(path,filename)
    output_img_path = os.path.join(output_path,filename)
    print(img_path)
    img = cv2.imread(img_path)
    h,w,c = img.shape
    img = cv2.resize(img,(int(w / 4),int(h / 4)))
    print("output : ",output_img_path)
    cv2.imwrite(output_img_path,img)
sys.exit()
