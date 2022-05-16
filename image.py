import cv2
import numpy as np
import time

webcam = cv2.VideoCapture(0)  #capture

pTime = 0
cTime = 0

while True:
    # medium_bx = 0
    # medium_gx = 0
    # medium_rx = 0

    success, imageFrame = webcam.read()
    imageFrame = cv2.flip(imageFrame, 1)
    imageFrame = cv2.resize(imageFrame, (700,700))

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)   # BGR(RGB color space) to HSV(hue-saturation-value)

    #######################################
    rows, cols, success = imageFrame.shape
    center_x = int(rows / 2)
    center_y = int(rows / 2)
    cv2.line(imageFrame, (center_x+100, 0), (center_x+100, 1300), (24, 81, 200), 1)
    cv2.line(imageFrame, (center_x-100, 0), (center_x-100, 1300), (24, 81, 200), 1)
    #######################################


    def red_op():
        medium_rx = 0
        cache = 0
        # Setting range
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) # mask

        kernal = np.ones((5, 5), "uint8")

        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >500:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w*h >= 8000:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                        cv2.putText(imageFrame, "Red Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

                        medium_rx = int((x + x+w) / 2)
                        medium_ry = int((y + y+h) / 2)

        if medium_rx > center_x+100:
            cv2.putText(imageFrame, "Red right= "+ str(medium_rx), (260,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_rx == 0:
            cv2.putText(imageFrame, "Red Not in screen", (260,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_rx < center_x-100:
            cv2.putText(imageFrame, "Red left= "+ str(medium_rx), (260,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_rx <= center_x + 100 and medium_rx >= center_x - 100:
            cv2.putText(imageFrame, "Red center= "+ str(medium_rx), (260,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    def green_op():
        medium_gx = 0
        flag_g = 0
        # green_lower = np.array([25, 52, 72], np.uint8)        ### Overlaps with blue
        # green_upper = np.array([102, 255, 255], np.uint8)
    
        green_lower = np.array([36, 50, 70], np.uint8)     ### better
        green_upper = np.array([89, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        kernal = np.ones((5, 5), "uint8")

        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >500:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w*h >= 8000:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                        cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

                        medium_gx = int((x + x+w) / 2)
                        medium_gy = int((y + y+h) / 2)
            
        if medium_gx > center_x+100:
            cv2.putText(imageFrame, "Green right= "+ str(medium_gx), (260,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_gx == 0:
            cv2.putText(imageFrame, "Green Not in screen", (260,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_gx < center_x-100:
            cv2.putText(imageFrame, "Green left= "+ str(medium_gx), (260,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_gx <= center_x + 100 and medium_gx >= center_x - 100:
            cv2.putText(imageFrame, "Green center= "+ str(medium_gx), (260,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))


    def blue_op():
        medium_bx = 0
        # blue_lower = np.array([94, 80, 2], np.uint8)       ### Overlaps with green 
        # blue_upper = np.array([120, 255, 255], np.uint8)
        blue_lower = np.array([100,70,2], np.uint8)   ### better
        blue_upper = np.array([128, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        kernal = np.ones((5, 5), "uint8")

        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)

        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >500:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w*h >= 8000:
                        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

                        medium_bx = int((x + x+w) / 2)
                        medium_by = int((y + y+h) / 2)

        if medium_bx > center_x+100:
            cv2.putText(imageFrame, "Blue right= " + str(medium_bx), (260,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_bx == 0:
            cv2.putText(imageFrame, "Blue Not in screen", (260,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_bx <= center_x + 100 and medium_bx >= center_x - 100:
            cv2.putText(imageFrame, "Blue center=" + str(medium_bx), (260,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif medium_bx < center_x-100:
            cv2.putText(imageFrame, "Blue left= "+ str(medium_bx), (260,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    if __name__ == '__main__':
        red_op()
        green_op()
        blue_op()
        ########## Fps count  ###############
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(imageFrame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
        cv2.waitKey(2)
        