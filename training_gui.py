from tkinter import *
import VideoCapture
from csv import writer
import numpy as np
import cv2
import EyeTracking

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
        self.eye_tracker = EyeTracking.EyeTracker(self.vid)
        self.left_eye = []
        self.file_name = file_name
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.circles = []
        self.color_ind = 2
        self.clicked = "cyan"
        self.eye_detected = "green"
        self.no_detection = "red"
        self.data = []
        self.col_scale = (self.screen_width - (self.circle_radius * 2))/self.cols
        self.row_scale = (self.screen_height - (self.circle_radius * 2))/self.rows
        self.canvas = Canvas(self.root, width = self.screen_width, height=self.screen_height, bg="black")
        self.canvas.bind("<Button-1>", self.mouse_pressed)
        self.canvas.pack()
        self.init_circles()
        self.delay = 15
        self.eye_update()
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

    #center x, center y, left eye x, left eye y, right eye x, right eye y
    def append_training(self, ind):
        (center_x, center_y, color) = self.circles[ind]
        row = [center_x, center_y]
        [centers, face_points, dists] = self.data
        for (x,y) in centers:
            row.append(x)
            row.append(y)
        for (x,y) in face_points:
            row.append(x)
            row.append(y)
        for dist in dists:
            row.append(dist)
        with open(self.file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(row)

    def mouse_pressed(self,event):
        self.data = self.eye_tracker.get_data()
        if(self.data != []):
            ind = self.find_nearest_circle(event.x, event.y)
            self.circles[ind][self.color_ind] = self.clicked
            self.append_training(ind)

    def init_circles(self):
        for i in range(0, self.rows+1):
            for j in range(0, self.cols+1):
                center_x = j*self.col_scale + self.circle_radius
                center_y = i*self.row_scale + self.circle_radius
                self.circles.append([center_x, center_y, self.no_detection])

    def eye_update(self):
        self.eye_tracker.mainloop()
        self.root.after(1,self.eye_update())

    def update(self):
        self.data = self.eye_tracker.get_data()
        color = self.no_detection
        if(self.data != []):
            color = self.eye_detected
            self.detection = True

        for circle in self.circles:
            if(circle[self.color_ind] != self.clicked):
                circle[self.color_ind] = color
            self.draw_circle(circle)

        self.root.after(self.delay, self.update)

root = Tk()
root.state("zoomed")
TrainingApp(root)
