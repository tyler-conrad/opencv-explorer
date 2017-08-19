from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

class LabeledContainer(BoxLayout):
    text = StringProperty()

Builder.load_file('uix/labeled_container.kv')