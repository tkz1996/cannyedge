import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

def makeinvariant(fdr):
    scaling_factor = np.absolute(fdr[1])
    fdr = fdr[1:]
    for i,point in enumerate(fdr):
        fdr[i] = np.absolute(point)/scaling_factor
    return fdr

def make_fdr_window(fdr_image, fdr_template, window_leftsize=20, window_rightsize=20):
    if window_leftsize == 0:
        return (fdr_image[:window_rightsize], fdr_template[:window_rightsize])
    if window_rightsize == 0:
        return (fdr_image[-window_leftsize:], fdr_template[-window_leftsize:]) 
    new_fdr_image = fdr_image[-window_leftsize:]
    new_fdr_image = np.append(new_fdr_image, fdr_image[:window_rightsize])
    new_fdr_template = fdr_template[-window_leftsize:]
    new_fdr_template = np.append(new_fdr_template, fdr_image[:window_rightsize])
    
    return (new_fdr_image, new_fdr_template)

def plswork(image_filename):
    print('Progress:')
    print('0%                   100%')
    show_graph_count = 1 #Change amount of graphs to print
    img_original = cv2.imread(image_filename, 1)
    retval, imgc = cv2.threshold(img_original,90,255,cv2.THRESH_BINARY) #Change threshold for blackwhite
    
    cropped_2 = imgc[195:195+235, 67:67+168].copy()
    cropped_c = imgc[184:184+200, 490:490+135].copy()
    print("  " , end =">" )
    template_images = [cropped_2, cropped_c]
    template_contour_list = []
    for i,img in enumerate(template_images):
        img_blur = img
        img_blur = cv2.GaussianBlur(img,(3,3), 0.25) #Change gaussian size and sigma

        image_canny = cv2.Canny(img_blur, 127, 255) #Change canny threshold
        image_canny2 = image_canny
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) #Change dilute kernal size
        image_canny2 = cv2.dilate(image_canny, dilate_kernel) #Change number of iterations
        cv2.imwrite(str(i)+'_canny.jpeg',image_canny2)
        contours, hierarchy = cv2.findContours(image_canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        drawing = np.zeros([image_canny2.shape[0], image_canny2.shape[1], 3])
        cv2.drawContours(drawing, contours, -1, (255,255,255))
        cv2.imwrite(str(i)+'_contour.jpeg',drawing)

        template_contour_list.append(contours)
    
    print("" , end =">>>>" )
    img = imgc.copy()
    img_blur = img
    img_blur = cv2.GaussianBlur(img,(3,3), 0.25) #Change gaussian size and sigma
    
    image_canny = cv2.Canny(img_blur, 127, 255) #Change canny threshold
    image_canny2 = image_canny
    print("" , end =">" )
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) #Change dilute kernal size
    image_canny2 = cv2.dilate(image_canny, dilate_kernel, iterations=2) #Change number of iterations
    print("" , end =">" )
    cv2.imwrite('image_canny.jpeg',image_canny2)
    print("" , end =">" )
    image_contours, hierarchy = cv2.findContours(image_canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("" , end =">" )
    
    drawing = np.zeros([image_canny2.shape[0], image_canny2.shape[1], 3])
##    for i in range(len(image_contours)):
##        cv2.drawContours(drawing, image_contours, i, (0,255,0), 2)
##        cv2.imwrite(str(i)+'image_contour.jpeg',drawing)

    cv2.drawContours(drawing, image_contours, -1, (255,255,255))
    cv2.imwrite('image_contour.jpeg',drawing)
    print("" , end =">" )

    complex_list_template_contours = []
    for contour in template_contour_list:
        complex_contour = []
        for point in contour[0]:
            complex_contour.append(complex(point[0][0],point[0][1]))
        complex_contour = np.fft.fft(complex_contour)
        complex_list_template_contours.append(complex_contour)
    print("" , end =">" )

    complex_image_contours = []
    for contour in image_contours:
        complex_contour = []
        for point in contour:
            complex_contour.append(complex(point[0][0],point[0][1]))
        complex_contour = np.fft.fft(complex_contour)
        complex_image_contours.append(complex_contour)
    print("" , end =">" )

    mag_list_template_contours = []
    mag_image_contours = []

    for i,contour in enumerate(complex_list_template_contours):
        mag_list_template_contours.append(makeinvariant(contour))
    print("" , end =">" )
    
    for i,contour in enumerate(complex_image_contours):
        mag_image_contours.append(makeinvariant(contour))
    print("" , end =">" )

    total_detected = 0
    colour = [(0,0,255),(0,255,0)] #Change colour of text
    template_code = ['2','C'] #Change template name
    threshold = 0.016 #Change threshold to accept/reject matching
    
    for i,template_contour in enumerate(mag_list_template_contours):
        for j,image_contour in enumerate(mag_image_contours):

            if show_graph_count:
                plt.subplot(1,2,1)
                plt.plot(template_contour.real)
                plt.title("Template Contour")

                plt.subplot(1,2,2)
                plt.plot(image_contour.real)
                plt.title("Image Contour")

                plt.suptitle("FFT Magnitude Graph --- Before Windowing")
                plt.savefig('Non-win mag_graph {}.jpg'.format(show_graph_count))
                plt.show()
            
            is_detected = 0
            if len(image_contour)<390: #Change threshold for minimum contour size
                continue
            windowleft = 0 #Change left window size for mag windowing
            windowright = 5 #Change right window size for mag windowing
            fdr_tuple = make_fdr_window(image_contour, template_contour, windowleft, windowright)
            image_contour, template_contour2 = fdr_tuple[0], fdr_tuple[1]
            avg_squared_dist = np.mean((template_contour2-image_contour)**2)

            if show_graph_count:
                plt.subplot(1,2,1)
                plt.plot(template_contour2.real)
                plt.title("Template Contour")

                plt.subplot(1,2,2)
                plt.plot(image_contour.real)
                plt.title("Image Contour")

                plt.suptitle("FFT Magnitude Graph --- After Windowing")
                plt.savefig('win mag_graph {}.jpg'.format(show_graph_count))
                plt.show()
                show_graph_count -= 1

            if avg_squared_dist < threshold:
                total_detected += 1
                is_detected = 1
                
            rect = cv2.boundingRect(image_contours[j])
            text = str(template_code[i]) + ': ' + str(round(avg_squared_dist.real,3))
            if is_detected:
                cv2.rectangle(img_original, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255,0,0), 3)
            cv2.putText(img_original, text, (rect[0]-rect[2]//4,rect[1]-(10+36*i)), cv2.FONT_HERSHEY_SIMPLEX, 1, colour[is_detected], 1)
        print("" , end =">>" )

    print(">")
    print("Finished")

    cv2.imwrite('results.jpg', img_original)

plswork('a4.bmp')
