from kivy.uix.slider import Slider


class ValidatingSlider(Slider):
    def set_min(self, min):
        try:
            fmin = float(min)
        except ValueError:
            pass

        if not fmin:
            return

        self.min = fmin

    def set_max(self, max):
        try:
            fmax = float(max)
        except ValueError:
            pass

        if not fmax:
            return

        self.max = fmax

    def set_step(self, step):
        try:
            fstep = float(step)
        except ValueError:
            pass

        if not fstep:
            return

        self.step = fstep
