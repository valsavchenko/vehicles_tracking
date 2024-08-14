import random
import uuid

import cv2

from timer import timer


class _Track:
    """
    """
    __STABLE_ID_GENERATOR = random.Random(x=7292)

    @classmethod
    def __get_unique_stable_id(cls):
        """
        """
        return uuid.UUID(int=cls.__STABLE_ID_GENERATOR.getrandbits(128), version=4)

    @staticmethod
    def __iou(rect_a, rect_b):
        """
        """
        i_l = max(rect_a[0], rect_b[0])
        i_t = max(rect_a[1], rect_b[1])
        i_r = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2])
        i_b = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3])
        rect_i = i_l, i_t, i_r - i_l, i_b - i_t

        area = lambda r: r[2] * r[3]
        iou = area(rect_i) / float(area(rect_a) + area(rect_b) - area(rect_i))
        return iou

    @timer(label='Track creation')
    def __init__(self, engine, frame, seed_detection):
        """
        """
        self.__engine = engine
        self.__engine.init(image=frame, boundingBox=seed_detection['lt_wh'])
        self.__points = [seed_detection['lt_wh'], ]
        self.__id = self.__get_unique_stable_id()

    @timer(label='Track extension')
    def extend(self, frame):
        """
        """
        tracked, box = self.__engine.update(image=frame)
        if tracked:
            self.__points.append(box)

        return tracked

    @timer(label='Track association')
    def select_best_match(self, iou_threshold, detections):
        """
        """
        best_iou, best_i = None, None

        last_point = self.__points[-1]
        for i, d in enumerate(detections):
            iou = self.__iou(rect_a=d['lt_wh'], rect_b=last_point)
            if iou < iou_threshold:
                continue

            if best_iou is None or best_iou < iou:
                best_iou, best_i = iou, i

        return best_i

    def get_id(self):
        """
        """
        return self.__id

    def get_centers(self):
        """
        """
        cs = [(l + w // 2, t + h // 2) for (l, t, w, h) in self.__points]
        return cs

    def get_last_rect(self):
        return self.__points[-1]


class Tracker:
    """
    """

    @staticmethod
    def __get_tracking_engine_creator(strategy):
        """
        """
        creator_per_id = {
            'kcf': cv2.TrackerKCF_create,
            'csrt': cv2.TrackerCSRT_create
        }

        return creator_per_id[strategy]

    def __init__(self, detector, settings):
        """
        """
        self.__detector = detector

        self.__frames_seen = 0
        self.__tracking_engine_creator = self.__get_tracking_engine_creator(strategy=settings['strategy'])
        self.__iou_threshold = settings['iou_threshold']
        self.__re_detect_every = settings['re_detect_every']
        self.__tracks = []

    @timer(label='Frame analysis')
    def track(self, frame):
        """
        """
        self.__frames_seen += 1

        # Identify tracks that couldn't be extended natively
        done_inds = []
        for ti, t in enumerate(self.__tracks):
            tracked = t.extend(frame=frame)
            if tracked:
                continue

            done_inds.append(ti)

        # Wrap up, if there are some tracks and they were extended natively
        all_tracks_were_extended = not done_inds and self.__tracks
        force_re_detection = 0 == self.__frames_seen % self.__re_detect_every
        if all_tracks_were_extended and not force_re_detection:
            return [], []

        # Dismiss tracks failed to extended natively
        done_ids = [self.__tracks[ti].get_id() for ti in done_inds]
        self.__tracks = [t for ti, t in enumerate(self.__tracks) if ti not in done_inds]

        done_inds = []
        detections = self.__detector.detect(frame=frame)
        for ti, t in enumerate(self.__tracks):
            di = t.select_best_match(iou_threshold=self.__iou_threshold, detections=detections)
            # Identify tracks that fail to match detections of the frame
            if di is None:
                done_inds.append(ti)
                continue

            # Identify trackless detections
            detections = detections[:di] + detections[di + 1:]

        # Dismiss tracks failed to match detection of the frame
        done_ids.extend([self.__tracks[ti].get_id() for ti in done_inds])
        self.__tracks = [t for ti, t in enumerate(self.__tracks) if ti not in done_inds]

        # Establish tracks for unbound detections of the frame
        est_ids = []
        for d in detections:
            try:
                t = _Track(engine=self.__tracking_engine_creator(), frame=frame, seed_detection=d)
            except Exception as e:
                print(f'{e}')
            else:
                est_ids.append(t.get_id())
                self.__tracks.append(t)

        return done_ids, est_ids

    def get_tracks(self):
        """
        """
        return self.__tracks
