import javabridge
import bioformats
import numpy as np
from pathlib import Path
import os
import cv2.cv2 as cv2
import math

# def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
#     dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
#     pts= []
#     for i in  np.arange(0,dist,gap):
#         r=i/dist
#         x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
#         y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
#         p = (x,y)
#         pts.append(p)
#
#     if style=='dotted':
#         for p in pts:
#             cv2.circle(img,p,thickness,color,-1)
#     else:
#         s=pts[0]
#         e=pts[0]
#         i=0
#         for p in pts:
#             s=e
#             e=p
#             if i%2==1:
#                 cv2.line(img,s,e,color,thickness)
#             i+=1
#
# def is_point_in_triangle(test_point, main_point, con_angle, min_len):
#     is_x_in_triangle = test_point[0] < main_point[0] + min_len
#     # Find y frame based on known angle and the height of the triangle (con_angle)
#     sin = math.sin(math.radians(con_angle))
#     y_frame = abs(math.sin(math.radians(con_angle)) * (main_point[0] - test_point[0]))
#     is_y_in_triangle = main_point[1] - y_frame <= test_point[1] <= main_point[1] + y_frame
#     return is_x_in_triangle and is_y_in_triangle
#
#
# def rotate_point(rotated_point, main_point, rot_angle):
#     """
#     Based on https://math.stackexchange.com/questions/3244392/getting-transformed-rotated-coordinates-relative-to
#     -the-center-of-a-rotated-ima example the new coordinates of the candidates' starting point (p1,p2) after a rotation by θ
#     degrees around the ending point of the main line (a,b): (a + x cos θ − y sin θ,   b + x sin θ + y cos θ)
#     where x = p1−a, y = p2−b
#     """
#     x = rotated_point[0] - main_point[0]
#     y = rotated_point[1] - main_point[1]
#     new_coordinates = (int(main_point[0] + x * math.cos(math.radians(rot_angle)) - y * math.sin(math.radians(rot_angle))),
#                        int(main_point[1] + x * math.sin(math.radians(rot_angle)) + y * math.cos(math.radians(rot_angle))))
#     return new_coordinates
#
#
#
#
# if __name__ == '__main__':
#     img = np.zeros((512, 512))
#
#     min_len = 250
#     con_searching_angle = 30
#
#     # 1. Start and end points of lines:
#     main_line_start_p = (50, 450)
#     main_line_end_p = (150, 350)
#
#     lines_start_ps = [(350, 200), (300, 350)]
#     lines_end_ps = [(480, 180), (500, 350)]
#
#
#     # 2. Draw lines
#     line1 = cv2.line(img, main_line_start_p, main_line_end_p, 5)
#     line2 =cv2.line(img, lines_start_ps[0], lines_end_ps[0], 5)
#     line3 =cv2.line(img, lines_start_ps[1], lines_end_ps[1], 5)
#     cv2.imshow("Test img", img)
#     cv2.waitKey()
#
#
#
#     # 3. Find rotation angle of the main line
#     angle_sin = (main_line_end_p[1] - main_line_start_p[1]) / cv2.norm(main_line_end_p, main_line_start_p)
#     rot_angle = - math.degrees(math.asin(angle_sin))
#
#     # 3.1 Draw all rotated lines
#     img2 = np.zeros((512, 512))
#     # central_point = (img2.shape[1] // 2, img2.shape[0] // 2)
#     central_point = (700, 700)
#     r_main_line_start_p = rotate_point((50, 450), central_point, rot_angle)
#     r_main_line_end_p = rotate_point((150, 350), central_point, rot_angle)
#     r_lines_start_ps = [rotate_point((350, 200), central_point, rot_angle), rotate_point((300, 350), central_point, rot_angle)  ]
#     r_lines_end_ps = [rotate_point((480, 180), central_point, rot_angle), rotate_point((500, 350), central_point, rot_angle) ]
#     r_line1 = cv2.line(img2, r_main_line_start_p, r_main_line_end_p, 5)
#     r_line2 =cv2.line(img2, r_lines_start_ps[0], r_lines_end_ps[0], 5)
#     r_line3 =cv2.line(img2, r_lines_start_ps[1], r_lines_end_ps[1], 5)
#     cv2.imshow("Test rotation", img2)
#     cv2.waitKey()
#
#
#     # 4. Find points that are located within the considered distance
#     central_point = (img.shape[1]//2, img.shape[0]//2)
#     rotated_main_point = rotate_point(main_line_end_p, central_point, rot_angle)
#
#     for i, point in enumerate(lines_start_ps):
#         if cv2.norm(main_line_end_p, point) <= min_len:
#             # 5. Rotate point and check if this point is located within the triangle:
#             rotated_point = rotate_point(point, central_point, rot_angle)
#
#             if is_point_in_triangle(rotated_point, rotated_main_point, con_searching_angle, min_len):
#                 drawline(img, main_line_end_p, point, color=(255, 255, 0), thickness=1, style='dotted', gap=10)
#                 # cv2.line(img, main_line_end_p, point, thickness=5, color=(255, 255, 0), lineType=cv2.LINE_8)
#                 distance = cv2.norm(main_line_end_p, point)
#                 print("______________________")
#                 print(f"Line {i} is in the triangle"
#                       f"- starting point is: {point} "
#                       f"- ending point is {lines_end_ps[i]} "
#                       f"- distance between lines is {distance}")
#     cv2.imshow("Test img", img)
#     cv2.waitKey()
main_line_end_p = (377, 263)
main_line_start_p = (348, 261)
angle_sin = (main_line_end_p[1] - main_line_start_p[1]) / cv2.norm(main_line_end_p, main_line_start_p)










