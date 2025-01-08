import numpy as np

import _ped_trk.timer


class Detector(_ped_trk.timer.Timeable):
    """
    """

    def __init__(self, logger, settings):
        """
        Sets up a model to infer with
        """
        super().__init__(logger=logger)

    @_ped_trk.timer.timer(label='obj_det')
    def detect(self, frame):
        """
        """
        height, width, *_ = frame.shape

        boxes = []
        num = np.random.randint(low=0, high=5)
        ls = np.random.randint(low=0, high=width, size=num)
        ts = np.random.randint(low=0, high=height, size=num)
        for l, t in zip(ls, ts):
            if (width - l) < 2 or (height - t) < 2:
                continue

            w = np.random.randint(low=1, high=(width - l))
            h = np.random.randint(low=1, high=(height - t))
            boxes.append((l, t, w, h))
        scores = np.random.rand(len(boxes))

        class_objs = [{'lt_wh': b, 'score': s} for b, s in zip(boxes, scores)]
        return class_objs
