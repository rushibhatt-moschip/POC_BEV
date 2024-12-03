''' 
Calibrate the blending onto an image. 
    script format : arg_1 = image_path 


    CONTROLS: 
        Quit                    : Esc
        translation             : 'q' : left | 'w' : up | 'e' : down | 'r' : right 
        change point            : space 
        clean blending          : 'c'
        get out of clean blend  : 'x'
        flip the mask           : 'f'
'''


import numpy as np 
import cv2
import sys 
import argparse
import os 

if len(sys.argv) < 1:
    print("give image path as argument \n")
    exit(0)

img = cv2.imread(sys.argv[1])
HEIGHT, WIDTH = img.shape[:2]
print(HEIGHT, WIDTH, img.shape)

# defining x1, x2, x3, x4, x5, x6, x7, x8 

#points = np.array([[ 10, 10], [10, HEIGHT - 10], 
#    [100, 10], [100, HEIGHT - 10],
#    [ WIDTH-100, 10], [WIDTH-100, HEIGHT - 10],
#    [WIDTH-50, 10], [WIDTH-50, HEIGHT - 10]])


points = np.array([[ 340, 220], [465, 530], 
    [460, 160], [495, 520],
    [ 720, 100], [725, 505],
    [930, 105], [760, 520]])

circle_selected = 0
clean  = False 
flip=0

while True:
    canvas = img.copy()
    key = cv2.waitKey(1)

    ########################## keys and command define part #################################
    #press (x) after pressing c to reset the canvas
    #press (f) to flip the mask from 1->0 or vice versa
    if key == 27:
        break

    if key == 32:
        circle_selected = (circle_selected + 1) if circle_selected < 7 else 0

    if key == ord ('q') or key == ord('w') or key == ord('e') or key == ord('r'):
        if key == ord('q') and points[circle_selected, 0] > 0:
            points[circle_selected, 0] -= 5
        elif key == ord('w') and points[circle_selected, 1] > 0:
            points[circle_selected, 1] -= 5
        elif key == ord('e') and points[circle_selected, 1] < HEIGHT:
            points[circle_selected, 1] += 5
        elif key == ord('r') and points[circle_selected, 0] < WIDTH:
            points[circle_selected, 0] += 5

    if key == ord('c'):
        clean = True

    if key == ord('f'):
        flip += 1




    #########################################################################################




    #################################### final drawing part #################################
    x_min = np.min(points[0:2, 0])
    x_max = np.max(points[2:4, 0])

    x_min_new = np.min(points[4:6, 0])
    x_max_new = np.max(points[6:8, 0])

    # generating the blended area 
    if (flip%2):
        blended_area = np.linspace(1,0, x_max - x_min)
        blended_area_new = np.linspace(1,0, x_max_new - x_min_new)
    else:
        blended_area = np.linspace(0,1, x_max - x_min)
        blended_area_new = np.linspace(0,1, x_max_new - x_min_new)

    blended_area = np.tile( blended_area, (3, HEIGHT, 1) )
    blended_area_new = np.tile( blended_area_new, (3, HEIGHT, 1) )
    blended_area = np.moveaxis(blended_area, 0, -1)
    blended_area_new = np.moveaxis(blended_area_new, 0, -1)

    blended_canvas = np.ones (canvas.shape).astype(np.float32)
    blended_canvas[:, x_min : x_max] = blended_area 
    blended_canvas[:,x_min_new : x_max_new]= blended_area_new


    ################################## calculation ##########################################
    if clean:

#Line 2
        m32 = (points[3, 1] - points[2, 1]) / (points[3, 0] - points[2, 0])
        b32 = points[3, 1] - m32 * points[3,0]
#Line 1
        m01 = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        b01 = points[1, 1] - m01 * points[1,0]

#Line 1 up
        m02 = (points[2, 1] - points[0, 1]) / (points[2, 0] - points[0, 0])
        b02 = points[2, 1] - m02 * points[2,0]
 

#Line 3
        m45 = (points[5, 1] - points[4, 1]) / (points[5, 0] - points[4, 0])
        b45 = points[5, 1] - m45 * points[5,0]
#Line 4 
        m67 = (points[7, 1] - points[6, 1]) / (points[7, 0] - points[6, 0])
        b67 = points[7, 1] - m67 * points[7,0]

#Line 3 up

        m46 = (points[6, 1] - points[4, 1]) / (points[6, 0] - points[4, 0])
        b46 = points[6, 1] - m46 * points[6,0]

        for y in range(blended_canvas.shape[0]):
            for x in range(blended_canvas.shape[1]):
       
#line 1 UP
                if y < m02 * x + b02 and x > points[0,0]-1 and x<points[2,0]+1: 
                    blended_canvas[y, x, :] = [1, 1, 1]

#line 3 UP
                if y < m46 * x + b46 and x > points[4,0]-1 and x<points[6,0]+1: 
                    blended_canvas[y, x, :] = [1, 1, 1]

#line 1        
                if y > m01 * x + b01 and x>(x_min-1):
                    blended_canvas[y, x, :] = [1, 1, 1]
#line 2
                if y < m32 * x + b32 and x<x_max+1:
                    blended_canvas[y, x, :] = [1, 1, 1]
#line 3 
                if y > m45 * x + b45 and x>x_min_new-1:
                    blended_canvas[y, x, :] = [1, 1, 1]
#line 4      
                if y > m67 * x + b67 and x<x_max_new+1:
                    blended_canvas[y, x, :] = [1, 1, 1]
            

    canvas = cv2.multiply (canvas.astype(np.float32), blended_canvas).astype(np.uint8)

    # drawing the lines 
    cv2.line (canvas, points[0], points[1], (0, 0, 255), 1)
    cv2.line (canvas, points[2], points[3], (0, 0, 255), 1)

# drawing the lines for new points

    cv2.line (canvas, points[4], points[5], (0, 0, 255), 1)
    cv2.line (canvas, points[6], points[7], (0, 0, 255), 1)



    # drawing the circles
    for i in range(8):
        if i == circle_selected:
            cv2.circle (canvas, points[i], 5, (255, 255, 0), 3)
        else:
            cv2.circle (canvas, points[i], 5, (0, 255, 0), 2)
    ######################################################################################
    # display the canvas

    cv2.imshow("canvas", canvas)
    cv2.resizeWindow("canvas", 800, 400)  # Example: resize to 800x600
    if clean:
        if (cv2.waitKey(0) == 120): # press x
            print(points)
            clean = False


cv2.destroyAllWindows()

