import cv2  # this is OpenCV
import matplotlib.pyplot as plt
import skimage
import numpy as np
import math

GREEN = (34,255,13)
RED = (255,36,12)
BLUE = (0,0,255)

# Form lines template
# NOTE: horizontal lines only cover main table area. Some offsets estimated from surrounding offsets
vert_lines = [93,124,145,173,223,274,327,369,397,646,737,762,806,842,885,939,1047,1081,1114,1153,1208,1256,1271,1472,1508,1764,1804,1850,1856,1892,1910,1931,1964]
horiz_lines = [299,320,594,611,642,673,704,735,766,797,830,861,893,924,955,988,1018,1050,1082,1113,1144,1175,1208,1241,1270,1303,1333,1364,1395,1427,1459,1490,1522,1554]

plt.rcParams['figure.figsize'] = (5,5)
plt.rcParams['figure.dpi'] = 72

# Notebook-friendly OpenCV image display
def viewGray(image):
    #reduced = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    plt.imshow(image, cmap='gray')
    plt.show()

def view(image, txt=''):
    #reduced = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

def hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def get_border_stencil(grayimage, debug=False):
    # You can use a Python slice operation to crop and image object.
    height, width = grayimage.shape
    middlegrayimg = grayimage[int(height/4):int(height*3/4), int(width/4):int(width*3/4)]

    # Note that the threshold function returns a threshold number and a binary image based upon that number.
    # In this line we only keep the threshold number that was calculated using the "Otsu Algorithm".
    threshold = cv2.threshold(middlegrayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    if debug:
        print(f'OpenCV calculated a threshold of {threshold} in the get_border_stencil() function')

    # Now we will apply our Otsu threshold number to a blurred version of the whole grayscale image.
    blurgray = cv2.blur(grayimage, (6,6))
    if debug:
        view(blurgray, "blurred")

    # Note that we save the second return value this time, the binary image obtained using the threshold from the prior step.
    binary_img = cv2.threshold(blurgray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    if debug:
        view(binary_img, "thresholded")
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel2)
    binary_inverted_img = cv2.bitwise_not(binary_img)
    if debug:
        view(binary_inverted_img, "inverted")
    cnts = cv2.findContours(binary_inverted_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        debug_image = cv2.cvtColor(grayimage,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, cnts[0], -1, GREEN, 4)
        view(debug_image, "contours")

    stencil = np.zeros(binary_img.shape).astype(binary_img.dtype)
    cv2.drawContours(stencil, cnts[0], -1, 1, thickness=-1)
    if debug:
        view(stencil, "stencil")
    return stencil

def get_radians_to_unrotate(skeleton, debug=False):
    debug_image = None
    if debug:
        debug_image = cv2.cvtColor(skeleton,cv2.COLOR_GRAY2BGR)

    min = np.pi/2 - np.pi/40  # only angles close to horizontal
    max = np.pi/2 + np.pi/40
    lines = cv2.HoughLines(skeleton, 1, np.pi / 180, 300, 0, 0, 0, 0)  
    thetas = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if min < theta < max:
                thetas.append(theta)

            # Our rotation function only needs theta values (line angles in radians)
            # However, we use the calculations below to draw lines for debug
            if debug:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2500*(-b)), int(y0 + 2500*(a)))
                pt2 = (int(x0 - 2500*(-b)), int(y0 - 2500*(a)))
                color = BLUE
                if min < theta < max:
                    color = GREEN
                cv2.line(debug_image, pt1, pt2, color, 3, cv2.LINE_AA)

    # Now we group the angles in a histogram and pick the largest group.
    # The hist variable captures the number of lines in each bin produced by the histogram function.
    # The edges variable gives us the angles that define the boundary values of the bins.
    hist, edges = np.histogram(thetas, bins='auto')
    bin_idx = 0
    for i in range(0, len(hist)):
        if hist[i] > hist[bin_idx]:
            bin_idx = i
    thetas = [ x for x in thetas if edges[bin_idx] < x <= edges[bin_idx+1]]
    
    # All of this debug code is just to draw the main bin angles in red in the image center.
    # Often the angles are identical and overlap, becoming one red line.
    if debug:
        for t in thetas:
            a = math.cos(t)
            b = math.sin(t)
            halfway_down = debug_image.shape[1] / 2
            x0 = a * halfway_down
            y0 = b * halfway_down
            pt1 = (int(x0 + 2500*(-b)), int(y0 + 2500*(a)))
            pt2 = (int(x0 - 2500*(-b)), int(y0 - 2500*(a)))
            cv2.line(debug_image, pt1, pt2, RED, 3, cv2.LINE_AA)
        view(debug_image, "Houghlines in blue and green, with green being close to horizontal")

    tau = np.average(thetas)
    result = np.pi/2 - tau
    return result

def rotate_image(image, radians):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, radians*180/np.pi, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def maximize_templated_ink(ink, template, translation=(-35,35), zoom_px=(-10, 5) ):
    result = (0,0)
    max_ink = 0
    for translation in range(translation[0], translation[1]):
        for zpx in range(zoom_px[0], zoom_px[1]):  # zoom range in pixels add/subtracted for entire dimension
            found_ink = 0
            zoom_factor = zpx/len(ink)  # zoom = zpx / total pixels (in this dimension only)
            for l in template:  # l represents each line position in the template
                l_z = int(l + l*zoom_factor)  # template line is "zoomed"
                test_offset = l_z + translation  # "zoomed" line is translated (shifted left or right)
                try: 
                    # This line's ink is contributed to the total found ink at this zoom and translation
                    # Neighbor ink is also included
                    found_ink = found_ink + ink[test_offset] + ink[test_offset-1] + ink[test_offset+1]
                except IndexError:  # We are ignoring any errors due to line tests beyond the range of our ink array..
                    print(test_offset)
                    pass
            if found_ink > max_ink:  # If we found more ink than before, set a new max and new result.
                max_ink = found_ink
                result = (zoom_factor, translation)
    return result  # return the combination that yeilded maximum ink

# Find significant vertical lines with a matching rectangular kernel shape.
# Input image is our inverted binary image with the mat removed.
def detectDarkVertLine(image, debug=False):
    debug_image = None
    if debug:
        debug_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # First we heal the breaks in any thick vertical lines
    dkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
    dilated = cv2.dilate(image, dkernel)

    # A 2 x 50 rectangle is used as a kernel to detect heavy vertical lines.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 50))
    remove_vertical = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Of the shapes detected, find the one that is the longest vertically
    rect = None
    for c in cnts:
        r = cv2.boundingRect(c)
        (x, y, w, h) = r
        if debug:
            cv2.rectangle(debug_image, r, BLUE, thickness=2)
        if rect is None or r[3] > rect[3]:
            rect = r

    if debug:
        cv2.rectangle(debug_image, rect, RED, thickness=2)
        view(debug_image, "Dilated image with contour bounding rectangles in blue, vertically longest in red.")
    return rect[1]

def extract(image, filename, debug=False):
    result = []
    height, width, depth = image.shape
    height = int(height/2)
    width = int(width/2)
    image = cv2.resize(image, (width, height))
    grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    stencil = get_border_stencil(grayimage, debug=False)

    # Otsu threshold and invert image in one step. Once again threshold based on middle of image.
    middlegrayimg = grayimage[int(height/4):int(height*3/4), int(width/4):int(width*3/4)]
    threshold = cv2.threshold(middlegrayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    binary_inverted_img = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    # Apply the border mask with a bitwise AND operation
    binary_inverted_img = cv2.bitwise_and(binary_inverted_img, binary_inverted_img, mask=stencil)
    if debug and False:
        view(binary_inverted_img, "inverted binary image without mat")
    
    # The zero specifies Zhuang Suen algorithm to the thinning operation, as per NCSA paper
    skeleton = cv2.ximgproc.thinning(binary_inverted_img, 0)
    if debug and False:
        view(skeleton[int(height/2)-500:int(height/2)+500, int(width/2)-500:int(width/2)+500], "the skeleton (zoomed in)")

    radians = get_radians_to_unrotate(skeleton, debug=False)
    if debug and False:
        print(f"The rotate angle for {filename} is {radians*180/np.pi} degrees.")

    # We want to fix the rotation in all of our versions of the image..
    skeleton = rotate_image(skeleton, radians)
    binary_inverted_img = rotate_image(binary_inverted_img, radians)
    image = rotate_image(image, radians)

    # Position the vertical template.
    v_lines_threshold = image.shape[1] / 7  # ink threshold is a proportion of image dimension
    v_ink = np.sum(skeleton, axis=0)  # Sum the ink along the Y axis, giving us a total for each position on the X axis.
    v_ink_thresh = [ x if x > v_lines_threshold else 0 for x in v_ink ]  # Zero out any ink below our threshold value.
    vzfactor, v_ink_offset = maximize_templated_ink(v_ink_thresh, vert_lines)  # search for best template position
    my_v_lines = [ int(l+l*vzfactor)+v_ink_offset for l in vert_lines]  # adjust template

    dark_vert_line_top = detectDarkVertLine(binary_inverted_img, debug=False)  # Find the top of the heavy vertical line..
    # The top of the heavy line corresponds to the first line in the horizontal template
    # So the translation values to explore should position the first line near there.
    # 't' below is the ideal translation for the heavy line, around which we will try to find max ink again.
    t = dark_vert_line_top - horiz_lines[0]  

    # Reusing vzfactor in our search for horizontal lines
    zpx = int(vzfactor * image.shape[0])

    h_lines_threshold = image.shape[0] / 7  # ink threshold is a proportion of image dimension
    h_ink = np.sum(skeleton, axis=1)  # Note that we are using a different axis here and the line above.
    h_ink_thresh = [ x if x > h_lines_threshold else 0 for x in h_ink ]  # Only include ink above the threshold level
    # Below we explore a very limited range of translation that puts the first line at the top of the heavy vertical.
    # We are also now constraining the zoom, with a max zoom equal to the zoom of the vertical lines template.
    hzfactor, h_ink_offset = maximize_templated_ink(h_ink_thresh, horiz_lines, translation=(t-2, t+2), zoom_px=(zpx-10, zpx))
    my_h_lines = [ int(l+l*hzfactor)+h_ink_offset for l in horiz_lines ]

    if debug:
        for l in my_h_lines:
            cv2.line(image, (0, l), (image.shape[1], l), RED, 1, cv2.LINE_AA)
        for l in my_v_lines:
            cv2.line(image, (l, 0), (l, image.shape[0]), RED, 1, cv2.LINE_AA)
        view(image, f"horiz template drawn at {hzfactor}, {h_ink_offset} for {filename}")

    return (image, my_v_lines, my_h_lines)