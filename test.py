import cv2
import numpy as np
import Finder
import VideoCapture

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 10, 255, nothing)
    vid = VideoCapture.MyVideoCapture(0)
    finder = Finder.FeatureFinder()
    while True:
        ret, frame = vid.get_frame()
        thresh_val = r = cv2.getTrackbarPos('threshold', 'image')
        finder.thresh_val = thresh_val
        if ret:
            finder.set_image(frame)
            finder.find_face()
            finder.find_eyes()
        if(finder.threshold != []):
            t = cv2.resize(finder.threshold, (500,500))
            cv2.imshow('image',t)

        # _, frame = cap.read()
        # face_frame = detect_faces(frame, face_cascade)
        # if face_frame is not None:
        #     eyes = detect_eyes(face_frame, eye_cascade)
        #     for eye in eyes:
        #         if eye is not None:

        #             eye = cut_eyebrows(eye)
        #             keypoints = blob_process(eye, threshold, detector)
        #             eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
