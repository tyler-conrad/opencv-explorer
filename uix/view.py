from kivy.properties import ObjectProperty
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget


class View(Widget):
    buffer = ObjectProperty(force_dispatch=True)

    def __init__(self, **kwargs):
        super(View, self).__init__(**kwargs)
        self.frame_size = [100.0, 100.0]

        with self.canvas:
            self.rect = Rectangle()

    def set_frame_size(self, frame_size):
        self.frame_size = frame_size
        self.texture = Texture.create(size=frame_size, colorfmt='bgr')
        self.update_rect()

    def on_size(self, target, size):
        self.update_rect()

    def update_rect(self):
        view_aspect = self.width / self.height
        frame_aspect = self.frame_size[0] / self.frame_size[1]

        if view_aspect > frame_aspect:
            width = self.height * frame_aspect
            self.rect.size = (width, self.height)
            self.rect.pos = self.x + (self.width - width) / 2.0, self.y
        else:
            height = self.width / frame_aspect
            self.rect.size = (self.width, height)
            self.rect.pos = (self.x, self.y + (self.height - height) / 2.0)

    def on_buffer(self, target, buffer):
        self.texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.rect.texture = self.texture
