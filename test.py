import cv2
import numpy as np
import Finder
import VideoCapture

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('thresh')
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'thresh', 10, 255, nothing)
    vid = VideoCapture.MyVideoCapture(0)
    finder = Finder.FeatureFinder()
    while True:
        ret, frame = vid.get_frame()
        thresh_val = r = cv2.getTrackbarPos('threshold', 'thresh')
        finder.thresh_val = thresh_val
        if ret:
            finder.set_image(frame)
            finder.find_face()
            finder.find_eyes()
        if(finder.threshold != []):
            t = cv2.resize(finder.threshold, (500,500))
            cv2.imshow('thresh',t)
            i = finder.eye_img
            if i != []:
                i = cv2.resize(i, (500,500))
                cv2.imshow('image',i)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
