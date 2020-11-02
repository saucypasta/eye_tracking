import cv2

class FeatureFinder:
    def __init__(self, image = []):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.image = image
        self.face = []
        self.roi = None
        self.eyes = []
        self.find_face()
        self.find_eyes()


    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def find_face(self):
        if len(self.image) == 0:
            return
        faces = self.face_cascade.detectMultiScale(self.image, 1.3, 5)
        if(len(faces) != 0):
            self.face = faces[0]

    def find_eyes(self):
        if len(self.face) == 0:
            return
        (x, y, w, h) = self.face
        self.roi = self.image[y:y+h, x:x+w]
        self.eyes = self.eye_cascade.detectMultiScale(self.roi)

    def get_left_eye(self):
        if(len(self.eyes) == 2):
            (x1, y1, w1, h1) = self.eyes[0]
            (x2, y2, w2, h2) = self.eyes[1]
            left_eye = self.eyes[0]
            if x2 < x1:
                left_eye = self.eyes[1]
            (x, y, w, h) = left_eye
            img = self.roi[y:y+x, x:x+w]
            return [left_eye, img]
        return []


    def draw_face_boundary(self):
        if len(self.face) == 0:
            return
        (x, y, w, h) = self.face
        cv2.rectangle(self.image, (x,y), (x+w, y+h), (255,0,0), 2)

    def draw_eye_boundaries(self):
        if len(self.eyes) == 0:
            return
        for (x, y, w, h) in self.eyes:
            cv2.rectangle(self.roi, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imwrite("eye.jpg", self.roi[y:y+h, x:x+w])
