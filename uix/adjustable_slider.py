from kivy.properties import StringProperty
from kivy.properties import NumericProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


class AdjustableSlider(BoxLayout):
    text = StringProperty()
    min = NumericProperty()
    max = NumericProperty()
    step = NumericProperty()
    init_value = NumericProperty()
    value = NumericProperty()

Builder.load_file('uix/adjustable_slider.kv')