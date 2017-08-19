from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Mesh

import cv2
import numpy as np

from util.query import get_by_kvid
from util.window import window


class Zoomer(Widget):
    camera = ObjectProperty()
    frame_width = NumericProperty()
    frame_height = NumericProperty()
    frame_size = ReferenceListProperty(frame_width, frame_height)

    def __init__(self, **kwargs):
        super(Zoomer, self).__init__(**kwargs)
        self.mesh_x = 0.0
        self.mesh_y = 0.0
        self.mesh_width = 100.0
        self.mesh_height = 100.0
        self.centroid_window = []
        self.frame_size_trigger = Clock.create_trigger(self.update_frame_size, -1)

        with self.canvas:
            self.mesh = self.build_mesh()

    def build_mesh(self):
        index_table = [[0, 3, 5, 7], [2, 1, 4, 6], [8, 9, 10, 11], [12, 13, 14, 15]]
        indices = []
        for y in range(3):
            for x in range(3):
                indices.extend((
                    index_table[y][x],
                    index_table[y + 1][x],
                    index_table[y + 1][x + 1],
                    index_table[y][x],
                    index_table[y + 1][x + 1],
                    index_table[y][x + 1]))

        return Mesh([], indices)

    def on_pos(self, target, pos):
        self.update_mesh()

    def on_size(self, target, size):
        self.update_mesh()

    def on_frame_size(self, target, frame_size):
        self.frame_size_trigger()

    def update_frame_size(self, dt):
        self.texture = Texture.create(size=self.frame_size, colorfmt='bgr')
        self.update_mesh()

    def update_mesh(self):
        view_aspect = self.width / self.height
        frame_aspect = self.frame_size[0] / self.frame_size[1]

        if view_aspect > frame_aspect:
            self.mesh_width = self.height * frame_aspect
            self.mesh_height = self.height
            self.mesh_x = self.x + (self.width - width) / 2.0
            self.mesh_y = self.y
        else:
            self.mesh_height = self.width / frame_aspect
            self.mesh_width = self.width
            self.mesh_x = self.x
            self.mesh_y = self.y + (self.height - height) / 2.0

    def on_camera(self, target, camera):
        ret, frame = camera.read()
        if not ret:
            Logger.error('Application: Failed to read frame from camera')
            return

        self.frame_size = frame.shape[1] * 1.0, frame.shape[0] * 1.0
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.camera.read()
        if not ret:
            Logger.error('Application: Failed to read frame from camera')
            return

        raw = cv2.flip(frame, 0)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 0.0, 31000.0, apertureSize=7)

        lines = cv2.HoughLinesP(edges, 1.4, 0.01, 120, 100.0, 13.0)
        line_buffer = np.zeros_like(raw)
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                x3 = 0
                y3 = ((y2 - y1) * (x3 - x1)) / (x2 - x1) + y1
                x4 = frame.shape[1]
                y4 = ((y2 - y1) * (x4 - x1)) / (x2 - x1) + y1
                cv2.line(
                    line_buffer,
                    (x3, y3),
                    (x4, y4),
                    (255, 255, 255),
                    10)
        inverted_line_buffer = cv2.bitwise_not(line_buffer)

        gray = cv2.cvtColor(inverted_line_buffer, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(raw)
        num_contours = len(contours)
        for i in range(num_contours):
            color = int(i * 255.0 / num_contours)
            cv2.drawContours(out, contours, i, (color, color, color), cv2.cv.CV_FILLED)

        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(out, kernel, iterations=5)

        gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        line_float = np.float32(gray)
        corner = cv2.cornerHarris(line_float, 30, 3, 0.07)
        corner_buffer = np.zeros_like(raw)
        corner_buffer[corner > 0.01 * corner.max()] = [255, 255, 255]

        gray = cv2.cvtColor(corner_buffer, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for contour in contours:
            moments = cv2.moments(contour)
            if abs(0.0 - moments['m00']) < 0.1:
                continue

            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])
            if x < 50 or y < 50 or x > self.frame_width - 50 or y > self.frame_height - 50:
                continue

            centroids.append((x, y))

        self.centroid_window.extend(centroids)
        self.centroid_window = self.centroid_window[-16:]

        polygon_buffer = np.zeros_like(raw)
        for triple in window(self.centroid_window):
            points = np.array(triple, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(polygon_buffer, [points], (255, 255, 255))

        gray = cv2.cvtColor(polygon_buffer, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            box_points = cv2.cv.BoxPoints(cv2.minAreaRect(contour))

        x = self.mesh_x
        y = self.mesh_y
        w = self.mesh_width
        h = self.mesh_height
        self.mesh.vertices = [
            x, y,
            x, y,
            x, y,
            x, y,
            w, y,
            w, y,
            w, y,
            w, y,
            x, h,
            x, h,
            w, h,
            w, h,
            w, y,
            w, y,
            w, h,
            w, h
        ]

        self.texture.blit_buffer(raw.flatten(), colorfmt='bgr', bufferfmt='ubyte')
        self.rect.texture = self.texture


class ZoomerApp(App):
    def build(self):
        self.camera = cv2.VideoCapture(-1)
        root = Builder.load_file('main.kv')
        get_by_kvid(root, 'zoomer').camera = self.camera
        return root

    def on_stop(self):
        self.camera.release()

if __name__ == '__main__':
    ZoomerApp().run()
