import cv2
import imutils
import numpy as np

# background global val
bg = None


def run_avg(image, aWeight):
    # init background
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return
# computes wieghted average and updates the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
# find the difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
# threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
# gets the contours in the thresholded area
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # no contours detected
    if len(cnts) == 0:
        return
    else:
        # get the maximum contour detected (which is the hand)
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


if __name__ == "__main__":
    # tune for outline wieght
    aWeight = 0.5

    # webcam call
    camera = cv2.VideoCapture(0)

    # segment size
    top, right, bottom, left = 10, 350, 225, 590

    # fram initialize
    num_frames = 0

    while(True):
        # read the current frame and call it day :P
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        # compares frames and gets the roi (return vals)
        roi = frame[top:bottom, right:left]

        # grayscale this mofo
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # get avg every 30 frames
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # hand segment is now seperate
            hand = segment(gray)

            if hand is not None:

                (thresholded, segmented) = hand
                # draw the region around the hand and display it
                cv2.drawContours(
                    clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
        # more drawing
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # moves onto next frame
        num_frames += 1

        # display the hand
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        # kill switch
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
