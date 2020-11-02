from tkinter import *
import VideoCapture
import Finder
from csv import writer
import numpy as np
import cv2

class TrainingApp:
    def __init__(self, root, cols = 6, rows = 4, radius = 50, video_source = 0, img_width = 50, img_height = 50, file_name = "training_data.csv"):
        self.root = root
        self.cols = cols
        self.rows = rows
        self.circle_radius = radius
        self.video_source = video_source
        self.img_width = img_width
        self.img_height = img_height
        self.vid = VideoCapture.MyVideoCapture(self.video_source)
        self.finder = Finder.FeatureFinder()
        self.left_eye = []
        self.file_name = file_name
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.circles = []
        self.color_ind = 2
        self.clicked = "cyan"
        self.eye_detected = "green"
        self.no_detection = "red"
        self.detection = False
        self.col_scale = (self.screen_width - (self.circle_radius * 2))/self.cols
        self.row_scale = (self.screen_height - (self.circle_radius * 2))/self.rows
        self.canvas = Canvas(self.root, width = self.screen_width, height=self.screen_height, bg="white")
        self.canvas.bind("<Button-1>", self.mouse_pressed)
        self.canvas.pack()
        self.init_circles()
        self.delay = 15
        self.update()
        self.root.mainloop()

    def draw_circle(self, circle):
        (center_x, center_y, color) = circle
        self.canvas.create_oval(center_x-self.circle_radius, center_y - self.circle_radius,
                           center_x+self.circle_radius, center_y + self.circle_radius, fill=color)

    def find_nearest_circle(self, x, y):
        col = round(x / self.col_scale)
        row = round(y / self.row_scale)
        ind = row * (self.cols+1) + col
        return ind

    #center x, center y, left eye x, left eye y, width scaling, height scaling, img 2500 elements (50x50)
    def append_training(self, ind):
        (center_x, center_y, color) = self.circles[ind]
        (x, y, w, h) = self.left_eye_coords
        w_scaling = self.img_width/w
        h_scaling = self.img_height/h
        img = self.left_eye_img
        cv2.imwrite("normal_eye.jpg",img)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation = cv2.INTER_AREA)
        img = np.reshape(img, self.img_width * self.img_height)
        row = [center_x, center_y, x, y, w_scaling, h_scaling] + img.tolist()
        with open(self.file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(row)

    def mouse_pressed(self,event):
        if(self.detection):
            ind = self.find_nearest_circle(event.x, event.y)
            self.circles[ind][self.color_ind] = self.clicked
            self.append_training(ind)

    def init_circles(self):
        for i in range(0, self.rows+1):
            for j in range(0, self.cols+1):
                center_x = j*self.col_scale + self.circle_radius
                center_y = i*self.row_scale + self.circle_radius
                self.circles.append([center_x, center_y, self.no_detection])

    def update(self):
        ret, frame = self.vid.get_frame()
        color = self.no_detection
        if ret:
            self.finder.set_image(frame)
            self.finder.find_face()
            self.finder.find_eyes()
            left_eye_info = self.finder.get_left_eye()
            self.detection = False
            if(len(left_eye_info) != 0):
                color = self.eye_detected
                self.detection = True
                self.left_eye_coords = left_eye_info[0]
                self.left_eye_img = left_eye_info[1]

        for circle in self.circles:
            if(circle[self.color_ind] != self.clicked):
                circle[self.color_ind] = color
            self.draw_circle(circle)

        self.root.after(self.delay, self.update)

root = Tk()
root.state("zoomed")
TrainingApp(root)
