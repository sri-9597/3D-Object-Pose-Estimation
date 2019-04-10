import cv2
import numpy as np
import math

def poseEstimation(image_points):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-45.0, -45.0, 0.0),
        (75.0, -45.0, 0.0),
        (-45.0, 75.0, 0.0),
        (75.0, 0.0, 0.0),
        (0.0, 75.0, 0.0)

    ])
    camera_matrix = np.array(
        [[707.975174, 0.000000, 326.343126],
         [0.000000, 711.961538, 232.354595],
         [0.000000, 0.000000, 1.000000]], dtype="double"
    )

    dist_coeffs = np.array([[0.085624],
                            [-0.579964],
                            [-0.007464],
                            [0.008612],
                            [0.000000]])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
'''
    print(
        "Rotation Vector:\n {0}".format(rotation_vector))
    print(
        "Translation Vector:\n {0}".format(translation_vector))
'''

cap = cv2.VideoCapture(1)
# image = cv2.imread("pic1.jpg")
while(True):
    ret, image = cap.read()
    grayimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thres = cv2.inRange(grayimg,190,255)
    kernel = np.ones((3, 3))
    closed = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    im2, contours, hierarchy = cv2.findContours(closed,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours[0]) >= 6:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image,(cx,cy),3,(0,255,0))
            epsilon = 0.01 * cv2.arcLength(contours[0],True)
            approx = cv2.approxPolyDP(contours[0],epsilon,True)

            min_dis = 10000
            center_point = [cx,cy]
            cent_index = 0
            ind = 0
            for point in approx:
                distance = math.sqrt((point[0][1] - cy)**2 + (point[0][0] - cx)**2)
                if distance < min_dis:
                    min_dis = distance
                    centx = point[0][0]
                    centy = point[0][1]
                    center_point = [centx,centy]
                    cent_index = ind
                ind +=1
            max_dis = 0
            p1_ind = 0
            p2_ind = 0
            # for i in range(len(approx)):
            #     a1 = approx[i][0][0]
            #     a2 = approx[i][0][1]
            #     for j in range(i+1,len(approx)):
            #         b1 = approx[j][0][0]
            #         b2 = approx[j][0][1]
            #         if i is not cent_index:
            #             distance = math.sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2)
            #             if distance > max_dis:
            #                 max_dis = distance
            #                 point1 = (a1,a2)
            #                 point2 = (b1,b2)
            #                 p1_ind = i
            #                 p2_ind = j

            point1 = (0, 0)
            point2 = (0, 0)
            image_points = [center_point]
            for i in range(0,len(approx)):
                a1 = approx[i][0][0]
                a2 = approx[i][0][1]
                for j in range(i+1,len(approx)):
                    b1 = approx[j][0][0]
                    b2 = approx[j][0][1]
                    countl = 0
                    countr = 0
                    if i!= cent_index and j!= cent_index:
                        for k in range(0,len(approx)):
                            if k!= i and k!= j:
                                d = ((approx[k][0][0] - approx[i][0][0]) * (approx[j][0][1] - approx[i][0][1])) - (
                                            (approx[k][0][1] - approx[i][0][1]) * (approx[j][0][0] - approx[i][0][0]))
                                if d < 0:
                                    countl += 1

                                elif d > 0:
                                    countr += 1

                        if countr ==2 and countl ==2:
                            point1 = [a1,a2]
                            point2 = [b1,b2]
                            p1_ind = i
                            p2_ind = j

            countl = 0
            countr = 0
            c2_indexl = 0
            c2_indexr = 0
            c2_index = 0
            corner_point2 = [0, 0]
            for i in range(0,len(approx)):
                if i!= p1_ind and i!=p2_ind and i!=cent_index:
                    d = ((approx[i][0][0] - point1[0])*(point2[1] - point1[1])) - ((approx[i][0][1] - point1[1])*(point2[0] - point1[0]))
                    if d < 0:
                        countl += 1
                        c2_indexl = i
                    elif d > 0:
                        countr += 1
                        c2_indexr = i
            # print(countl, countr)
            if countl == 1:
                corner_point2 = approx[c2_indexl][0]
                c2_index = c2_indexl
            elif countr == 1:
                corner_point2 = approx[c2_indexr][0]
                c2_index = c2_indexr
            # print(corner_point2)
            image_points.append(corner_point2)
            rem_points_index = []
            for i in range(0,len(approx)):
                if i!=cent_index and i!=p1_ind and i!=p2_ind and i!= c2_index:
                    rem_points_index.append(i)

            if corner_point2[1] > center_point[1]:
                if point1[0] > center_point[0]:
                    image_points.append(point1)
                    image_points.append(point2)
                elif point2[0] > center_point[0]:
                    image_points.append(point2)
                    image_points.append(point1)
                if approx[rem_points_index[0]][0][0] > center_point[0]:
                    image_points.append(approx[rem_points_index[0]][0])
                    image_points.append(approx[rem_points_index[1]][0])
                elif approx[rem_points_index[1]][0][0] > center_point[0]:
                    image_points.append(approx[rem_points_index[1]][0])
                    image_points.append(approx[rem_points_index[0]][0])
            else:
                if point1[0] < center_point[0]:
                    image_points.append(point1)
                    image_points.append(point2)
                elif point2[0] < center_point[0]:
                    image_points.append(point2)
                    image_points.append(point1)
                if approx[rem_points_index[0]][0][0] < center_point[0]:
                    image_points.append(approx[rem_points_index[0]][0])
                    image_points.append(approx[rem_points_index[1]][0])
                elif approx[rem_points_index[1]][0][0] < center_point[0]:
                    image_points.append(approx[rem_points_index[1]][0])
                    image_points.append(approx[rem_points_index[0]][0])
            print(image_points)

            #Image Points
            img_pts = np.array(image_points,dtype='double')
            if len(img_pts) == 6:
                # print(img_pts)
                poseEstimation(img_pts)
            #cv2.line(image,point1,point2,(0,0,255),3)
            cv2.drawContours(image,approx,-1,(255,0,0),5)
    cv2.imshow("Closed",closed)
    cv2.imshow("Original",image)
    cv2.imwrite("pic2.jpg",image)
    cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()