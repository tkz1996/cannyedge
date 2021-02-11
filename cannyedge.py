import cv2
import numpy as np
import cmath
import os

def gaussian_kernel(size, sigma=1): #function to create gaussian mask
    size = size//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g_coefficient = 1/(sigma*(2*np.pi)**0.5)
    g_distribution = np.exp(-(x**2+y**2)/(2*sigma**2))*g_coefficient #gaussian equation
##    g_distribution = -1/(sigma**4 * np.pi)*(1-(-(x**2+y**2)/(2*sigma**2)))*np.exp(-(x**2+y**2)/(2*sigma**2)) #Mexican Hat filter
    return g_distribution

def sobel_gradient(img):
    
    #Create Sobel filter, 2 masks
    mat1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    mat2 = mat1.transpose()
    
    #Convolute each sobel mask with image to get gradient on x and y axis in both directions (since unsigned int, -ve derivatives are lost)
    #Since only unsigned int, negative gradient lost. So sobel need to be applied in 4 directions
    x_grad1 = cv2.filter2D(img, -1, mat2)
    y_grad1 = cv2.filter2D(img, -1, mat1)
    x_grad2 = cv2.filter2D(img, -1, np.fliplr(mat2))
    y_grad2 = cv2.filter2D(img, -1, np.flipud(mat1))

    x_grad = np.absolute(x_grad1) + np.absolute(x_grad2)
    y_grad = np.absolute(y_grad1) + np.absolute(y_grad2)
    
    #Convert to larger number type to find magnitude
    x_grad = x_grad.astype(np.float32)
    y_grad = y_grad.astype(np.float32)
    
    #Find gradient between x grad image and y grad image and put to new image
    img_grad = np.sqrt(np.square(x_grad) + np.square(y_grad))
    
    #Normalize value to within 0 - 255
    img_grad = img_grad * 255.0 / img_grad.max()

    #Find the edge direction and put in matrix
    img_direction = np.arctan2(y_grad, x_grad)

    img_grad = img_grad.astype(np.uint8)
    
    return (img_grad, img_direction)

def non_max_suppression(img_grad, img_direction):
    #Zero intialize the new edge image
    x, y = img_grad.shape
    img_edge = np.zeros((x,y), dtype=np.uint8)
    
    #Change direction imgfrom radians to degrees
    img_direction = img_direction * 180 / np.pi

    #Loop through all elements in one circle and check front direction
    #This is to cover all 360 degrees
    #Find if current pixel is max along edge direction
    #If it is the highest, set as itself in edge image
    #Otherwise set as 0 to not be an edge
    for i in range(1,x-1):
        for j in range(1,y-1):
            try:
                front = 0

                if (-22.5 <= img_direction[i,j] < 22.5):
                    front = img_grad[i, j+1]

                elif (22.5 <= img_direction[i,j] < 67.5):
                    front = img_grad[i-1, j+1]

                elif (67.5 <= img_direction[i,j] < 112.5):
                    front = img_grad[i-1, j]

                elif (112.5 <= img_direction[i,j] < 157.5):
                    front = img_grad[i-1, j-1]

                elif (157.5 <= img_direction[i,j] <= 180) or (-180 <= img_direction[i,j] < -157.5):
                    front = img_grad[i, j-1]

                elif (-67.5 <= img_direction[i,j] < -22.5):
                    front = img_grad[i+1, j+1]

                elif (-112.5 <= img_direction[i,j] < -67.5):
                    front = img_grad[i+1, j]

                elif (-157.5 <= img_direction[i,j] < -112.5):
                    front = img_grad[i+1, j-1]

                if (img_grad[i,j] > front):
                    img_edge[i,j] = img_grad[i,j]
                else:
                    img_edge[i,j] = 0

            except IndexError as e:
                print(e)
                
    return img_edge

def hysteresis(img_nonmax):
    #Double Thresholding
    highThreshold = img_nonmax.max() * 0.5 #Change threshold for edge detection
    lowThreshold = img_nonmax.max() * 0.1 #Change threshold for non-edge detection

    x,y = img_nonmax.shape
    Threshold_img = np.zeros((x,y),dtype=np.uint8)

    low = np.uint8(0) #Change pixel value for non-edge
    high = np.uint8(255) #Change pixel value for edge

    high_i, high_j = np.asarray(img_nonmax >= highThreshold).nonzero()
    low_i, low_j = np.asarray(np.logical_and(lowThreshold <= img_nonmax, img_nonmax < highThreshold)).nonzero()

    Threshold_img[high_i, high_j] = high
    Threshold_img[low_i, low_j] = low

    #Hysteresis function
    
    for i in range(1, x-1):
        for j in range(1, y-1):
            if (lowThreshold <= Threshold_img[i,j] < highThreshold):
                try:
                    if ((Threshold_img[i+1, j-1] == highThreshold) or (Threshold_img[i+1, j] == highThreshold) or (Threshold_img[i+1, j+1] == highThreshold)
                        or (Threshold_img[i, j-1] == highThreshold) or (Threshold_img[i, j+1] == highThreshold)
                        or (Threshold_img[i-1, j-1] == highThreshold) or (Threshold_img[i-1, j] == highThreshold) or (Threshold_img[i-1, j+1] == highThreshold)):
                        Threshold_img[i,j] = highThreshold
                    else:
                        Threshold_img[i,j] = low
                except IndexError as e:
                    print(e)
    return Threshold_img

def draw_contours(image_edge, contours, hierarchy):
    drawing = np.zeros((image_edge.shape[0], image_edge.shape[1]), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255) #Change for contour colour
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

    return drawing

def read_contours(contours, location, to_print = False):
    #Reads the list of contours and combines into 1 contour.
    #Saves a file of all the coordinates of the contour
    #Converts the list of coordinates to Complex vectors for DFT.
    #The complex vector array is returned
    
    if to_print:
        f = open(location + "_contours_points_cw.txt", "w")

    combined_contour = np.vstack(contours)
    complex_vectors = []
    coordinates = []
    
    for point in combined_contour:
        coordinates.append([point[0][0],point[0][1]])
        complex_vectors.append(complex(point[0][0],point[0][1]))

    if to_print:
        f.write(str(coordinates))
        f.close()
    
    return np.array(complex_vectors)

        
    
def run(filename):
    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = gaussian_kernel(5,1)
    location = filename.split('.')[0]
    
    image_smooth = cv2.filter2D(img, -1, kernel)
##    cv2.imwrite(location +'_Smooth' + '.jpg',image_smooth)
    
    image_data = sobel_gradient(image_smooth)    
##    cv2.imwrite(location + '_Gradient' + '.jpg',image_data[0])

    image_grad = image_data[0]
    image_direction = image_data[1]
    image_nonmax = non_max_suppression(image_grad, image_direction)
##    cv2.imwrite(location + '_Nonmax' + '.jpg',image_nonmax)

    image_edge = hysteresis(image_nonmax)
    cv2.imwrite(location + '_Edge' + '.jpg',image_edge)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) #Change for dilute kernal size
    image_edge = cv2.dilate(image_edge, dilate_kernel)

    contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_contours = draw_contours(image_edge, contours, hierarchy)
    cv2.imwrite(location + '_Contours' + '.jpg', image_contours)

    complex_vectors = read_contours(contours, location)
    dft_vectors = np.fft.fft(complex_vectors)
    np.save(location, dft_vectors)
    np.savetxt(location+'_DFT.txt', dft_vectors)
    
    return



if __name__ == "__main__":
    cmd = ''
    print('Type q to quit.')
    while True:
        cmd = input('Type in template name: ')
        if cmd == 'q':
            break
        if not os.path.exists(cmd):
            print("File does not exist. Try again.")            
        else:
            run(cmd)
            print('Finished')
            print('---------------------------------------')
