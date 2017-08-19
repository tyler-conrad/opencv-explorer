from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.gridlayout import GridLayout

import cv2
import numpy as np

from util.query import get_by_kvid
from util.window import window


class Grid(GridLayout):
    camera = ObjectProperty()
    frame_width = NumericProperty()
    frame_height = NumericProperty()
    frame_size = ReferenceListProperty(frame_width, frame_height)

    def __init__(self, **kwargs):
        super(Grid, self).__init__(**kwargs)
        self.centroid_window = []
        Clock.schedule_once(self.assign_views, -1)
        self.frame_size_trigger = Clock.create_trigger(self.dispatch_frame_size, -1)

    def assign_views(self, dt):
        self.views = {
            'raw': get_by_kvid(self, 'raw'),
            'gray': get_by_kvid(self, 'gray'),
            'edges': get_by_kvid(self, 'edges'),
            'lines1': get_by_kvid(self, 'lines1'),
            'contours': get_by_kvid(self, 'contours'),
            'dilated': get_by_kvid(self, 'dilated'),
            'corners1': get_by_kvid(self, 'corners1'),
            'centroids': get_by_kvid(self, 'centroids'),
            'polygon': get_by_kvid(self, 'polygon'),
            'min_area_rect': get_by_kvid(self, 'min_area_rect')
        }

    def on_frame_size(self, target, frame_size):
        self.frame_size_trigger()

    def dispatch_frame_size(self, dt):
        for view in self.views.values():
            view.set_frame_size(self.frame_size)

    def on_camera(self, target, camera):
        ret, frame = camera.read()
        if not ret:
            Logger.error('Application: Failed to read frame from camera')
            return

        self.frame_size = frame.shape[1] * 1.0, frame.shape[0] * 1.0
        self.centroid_buffer = np.zeros_like(frame)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def get(self, kvid):
        return get_by_kvid(self.parent, kvid).value

    def update(self, dt):
        ret, frame = self.camera.read()
        if not ret:
            Logger.error('Application: Failed to read frame from camera')
            return

        raw = cv2.flip(frame, 0)
        self.views['raw'].buffer = raw.flatten()

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.views['gray'].buffer = gray_bgr.flatten()

        edges = cv2.Canny(
            gray,
            self.get('edges_min'),
            self.get('edges_max'),
            apertureSize=int(self.get('edges_aperture_size')))
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.views['edges'].buffer = edges_bgr.flatten()

        lines = cv2.HoughLinesP(
            edges,
            self.get('lines1_rho'),
            self.get('lines1_theta'),
            int(self.get('lines1_threshold')),
            self.get('lines1_min_line_len'),
            self.get('lines1_max_line_gap'))
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
                    int(self.get('lines1_line_width')))
        inverted_line_buffer = cv2.bitwise_not(line_buffer)
        self.views['lines1'].buffer = inverted_line_buffer.flatten()

        gray = cv2.cvtColor(inverted_line_buffer, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(raw)
        num_contours = len(contours)
        for i in range(num_contours):
            color = int(i * 255.0 / num_contours)
            cv2.drawContours(out, contours, i, (color, color, color), cv2.cv.CV_FILLED)
        self.views['contours'].buffer = out.flatten()

        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(out, kernel, iterations=5)
        self.views['dilated'].buffer = dilated.flatten()

        gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
        line_float = np.float32(gray)
        corner = cv2.cornerHarris(
             line_float,
             int(self.get('corners1_block_size')),
             int(self.get('corners1_ksize')),
             self.get('corners1_k'))
        corner_buffer = np.zeros_like(raw)
        corner_buffer[corner > self.get('corners1_cutoff') * corner.max()] = [255, 255, 255]
        self.views['corners1'].buffer = corner_buffer.flatten()

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

        for centroid in centroids:
            x = centroid[0]
            y = centroid[1]
            cv2.rectangle(self.centroid_buffer, (x, y), (x + 1, y + 1), (0, 0, 255))
        self.views['centroids'].buffer = self.centroid_buffer.flatten()

        self.centroid_window.extend(centroids)
        self.centroid_window = self.centroid_window[-16:]

        polygon_buffer = np.zeros_like(raw)
        for triple in window(self.centroid_window):
            points = np.array(triple, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(polygon_buffer, [points], (255, 255, 255))
        self.views['polygon'].buffer = polygon_buffer.flatten()

        gray = cv2.cvtColor(polygon_buffer, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area_rect_buffer = np.zeros_like(raw)
        for contour in contours:
            box_points = cv2.cv.BoxPoints(cv2.minAreaRect(contour))
            cv2.drawContours(
                min_area_rect_buffer,
                [np.int0(box_points)],
                0,
                (255, 255, 255),
                cv2.cv.CV_FILLED)
        self.views['min_area_rect'].buffer = min_area_rect_buffer.flatten()


class OpenCVApp(App):
    def build(self):
        self.camera = cv2.VideoCapture(-1)
        root = Builder.load_file('grid.kv')
        get_by_kvid(root, 'grid').camera = self.camera
        return root

    def on_stop(self):
        self.camera.release()

if __name__ == '__main__':
    OpenCVApp().run()