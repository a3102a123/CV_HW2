import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys

def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img = cylindricalProjection(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def cylindricalProjection(img) :
    rows = img.shape[0]
    cols = img.shape[1]

    #f = cols / (2 * math.tan(np.pi / 8))
    result = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    alpha = math.pi / 4
    f = cols / (2 * math.tan(alpha / 2))
    for  y in range(rows):
        for x in range(cols):
            theta = math.atan((x- center_x )/ f)
            point_x = int(f * math.tan( (x-center_x) / f) + center_x)
            point_y = int( (y-center_y) / math.cos(theta) + center_y)

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                result[y , x, :] = img[point_y , point_x ,:]
    return result

def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SIFT(img):
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(img, None)
    return kp, des

def draw_SIFT(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def matcher(kp1, des1, kp2, des2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))
    return np.array(matches)

def knnMatch(kp1, des1, kp2, des2, threshold):
    # KNN
    print("KNN matching ...")
    k_matches = []
    for i,d1 in enumerate(des1):
        min_kp = [-1,np.inf]
        sec_min_kp = [-1,np.inf]
        for j,d2 in enumerate(des2):
            dist = np.linalg.norm(d1 - d2)
            if min_kp[1] > dist:
                sec_min_kp = np.copy(min_kp)
                min_kp = [j,dist]
            elif sec_min_kp[1] > dist:
                sec_min_kp = [j,dist]
        k_matches.append((min_kp,sec_min_kp))
    # ratio test
    print("Ratio test ...")
    matches = []
    for i,(m1,m2) in enumerate(k_matches):
        # print("index : {}".format(i),m1,m2)
        # print(m1[1] , threshold*m2[1])
        if m1[1] < threshold*m2[1]:
            # unpacking the tuple to let one match stores 4 element (p1.x , p1.y , p2.x , p2.y)  
            # It doesn't mean summoing up two position
            matches.append(list(kp1[i].pt + kp2[m1[0]].pt))
    return np.array(matches)



def draw_matches(matches, img1,img2):
    match_img = np.concatenate((img1, img2), axis=1)
    offset = img1.shape[1]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)
    plt.show()

def homography(pairs):
    # first way Ax = 0
    rows = []
    for i in range(len(pairs)):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)

    # second way Ax = B
    B = np.zeros((8,1))
    A = np.zeros((8,8))
    for i in range(len(pairs)):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        A[i,:] = [p1[0],p1[1],1,0,0,0,-p1[0] * p2[0],-p1[1] * p2[0]]
        A[i+4,:] = [0,0,0,p1[0],p1[1],1,-p1[0] * p2[1],-p1[1] * p2[1]]
        B[i] = p2[0]
        B[i+4] = p2[1]

    # sloving Ax = B by pseudo inverse
    AI = np.linalg.pinv(A.T@A)
    H = AI @ A.T @ B
    temp = np.ones((9,1))
    temp[0:8] = H
    H = np.array(temp).reshape(3,3)
    # sloving A = h0 use svd 
    # # V contain the eigen vectors of matrix A
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1 to fit the format of homography matrix
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = []
    for i in idx:
        point.append(matches[i])
    return np.array(point)

def get_error(points, H):
    num = len(points)
    p1_arr = np.ones((num,3))
    p1_arr[:,0:2] = points[:,0:2]
    p2_arr = points[:,2:4]
    estimate_p2 = np.zeros((num, 2))
    for i in range(num):
        temp = np.dot(H, p1_arr[i])
        # set index 2 to 1 and slice the index 0, 1
        estimate_p2[i] = (temp/temp[2])[0:2]
    # Compute square error between estimation & ground truth 
    errors = np.linalg.norm(p2_arr - estimate_p2 , axis=1) ** 2

    return errors

def RANSAC(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_img(img1, img2, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    img1 = cv2.normalize(img1.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    img2 = cv2.normalize(img2.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # img1 image
    height1, width1, channel1 = img1.shape
    # 4 corners of image1 in homogenus coordinate
    corners = [[0, 0, 1], [width1, 0, 1], [width1, height1, 1], [0, height1, 1]]
    # calc the new corner after homography (Perspective Transformation)
    corners_new = []
    for corner in corners:
        corners_new.append(np.dot(H, corner))
    # dimension : 3(x,y,s) x 4(4 corners)
    corners_new = np.array(corners_new).T 
    # let the last coordinate is 1
    corners_new = corners_new[0:3,:]/corners_new[2,:].reshape(1,-1)
    # now I get the position on image2 of 4 corners from image 1
    # so I can find the most left-down point to know how big new imgae needing to append
    # because it stitchs the image on left and right , the result should be bigger
    # and the result of homograpy should contain minus position. 
    # It's not necessary to make sure that the x_min , y_min aren't > 0 
    x_min = min(min(corners_new[0]),0)
    y_min = min(min(corners_new[1]),0)

    # construct a translation matrix translate image 2 to right top and give some space to image 1
    A = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # let translation matirx work on image2 space
    H = A @ H
    
    height2, width2, channel2 = img2.shape
    
    height_new = int(round(abs(y_min) + height2))
    width_new = int(round(abs(x_min) + width2))
    size = (width_new, height_new)

    # let two image in same space and same size
    warped_l = cv2.warpPerspective(src=img1, M=H, dsize=size)
    warped_r = cv2.warpPerspective(src=img2, M=A, dsize=size)
    # creat_im_window("left",img1)
    # creat_im_window("right",img2)
    # creat_im_window("left_w",warped_l)
    # creat_im_window("right_w",warped_r)
    # im_show()
     
    black = np.zeros(3)  # Black pixel.

    # blending parameter
    blending_l = math.floor(-x_min)
    blending_r = math.ceil(max(corners_new[0]) - x_min)
    blending_total_w = blending_r - blending_l
    
    # Stitching procedure, store results in warped_l.
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            # blending
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                weight = j - blending_l
                # blending
                warped_l[i, j, :] = (pixel_l * (blending_total_w - weight) + pixel_r * weight) / blending_total_w
                # average
                # warped_l[i, j, :] = (pixel_l + pixel_r ) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


def SIFT_img_concate(img_left,img_left_gray,img_right,img_right_gray):
    left_kp,left_des = SIFT(img_left_gray)
    right_kp,right_des = SIFT(img_right_gray)
    # matches = knnMatch(left_kp,left_des,right_kp,right_des, 0.5)

    # for test using opencv macher is faster
    matches = matcher(left_kp, left_des, right_kp, right_des, 0.5)
    # draw_matches(matches,img_left_rgb,img_right_rgb)
    # exit()
    inliers, H = RANSAC(matches, 0.5, 2000)
    # H = cv2.findHomography(matches[:,0:2],matches[:,2:4])[0]
    # draw_matches(inliers,img_left_rgb,img_right_rgb)
    result = stitch_img(img_left, img_right, H)
    # convert the dtype from float to uint8
    result = (result * 255).astype("uint8")
    result_gray = img_to_gray(result)
    return result, result_gray

if __name__ == '__main__':
    result , result_gray = read_img("test/library/h_s/m1.jpg")
    for i in range(2,3):
        img_path = "test/library/h_s/m{}.jpg".format(i)
        img , img_gray = read_img(img_path)
        result, result_gray =  SIFT_img_concate(result ,result_gray ,img ,img_gray)
    creat_im_window("Result",result)
    im_show()
    cv2.imwrite("result.jpg",result)