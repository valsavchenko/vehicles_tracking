import logging
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

    @staticmethod
    def center(rect):
        """
        """
        l, t, w, h = rect
        return l + w // 2, t + h // 2

    @timer(label='trk_creation')
    def __init__(self, logger, engine, frame, seed_detection):
        """
        """
        self.__logger = logger
        self.__engine = engine
        self.__engine.init(image=frame, boundingBox=seed_detection['lt_wh'])
        self.__points = [seed_detection['lt_wh'], ]
        self.__id = self.__get_unique_stable_id()

    def get_logger(self):
        """
        """
        return self.__logger

    @timer(label='trk_update')
    def extend(self, frame):
        """
        """
        tracked, box = self.__engine.update(image=frame)
        if tracked:
            self.__points.append(box)

        return tracked

    @timer(label='trk_det_association')
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
        """
        """
        return self.__points[-1]

    def get_last_center(self):
        """
        """
        return self.get_centers()[-1]


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

    @staticmethod
    def __within(rect, center):
        """
        """
        l, t, w, h = rect
        x, y = center
        return l <= x <= l + w and t <= y <= t + h

    def __init__(self, logger, detector, settings, roi):
        """
        """
        self.__logger = logger
        self.__detector = detector

        self.__tracking_engine_creator = self.__get_tracking_engine_creator(strategy=settings['strategy'])
        self.__iou_threshold = settings['iou_threshold']
        self.__re_detect_every = settings['re_detect_every']
        self.__roi = roi
        self.get_logger().info(f'Establish a tracker: {{'
                               f'"strategy": {settings["strategy"]}, '
                               f'"iou_threshold": {self.__iou_threshold}, '
                               f'"re_detect_every": {self.__re_detect_every}, '
                               f'"roi": {self.__roi}'
                               f'}}')

        self.__frames_seen = 0
        self.__tracks = []

    def get_logger(self):
        """
        """
        return self.__logger

    @timer(label='frame_handling')
    def track(self, frame):
        """
        """
        self.__frames_seen += 1
        self.get_logger().debug(f'Track objects: {{'
                                f'"frame_id": {self.__frames_seen}'
                                f'}}')

        # Init ROI, if applicable
        if self.__roi is None:
            self.__roi = 0, 0, frame.shape[1] - 1, frame.shape[0] - 1
            self.get_logger().info(f'Settle the ROI: {{"ltwh": {self.__roi}}}')

        failed_to_extend_inds, stick_out_roi_inds = [], []
        for ti, t in enumerate(self.__tracks):
            tracked = t.extend(frame=frame)
            if not tracked:
                # Identify tracks that couldn't be extended natively
                failed_to_extend_inds.append(ti)
            elif not self.__within(rect=self.__roi, center=t.get_last_center()):
                # Identify tracks that stick out of the ROI
                stick_out_roi_inds.append(ti)

        # Wrap up, if there are some tracks, they were extended natively, it's not the time to look for extra objects
        all_tracks_were_extended = not bool(failed_to_extend_inds + stick_out_roi_inds)
        force_re_detection = 0 == self.__frames_seen % self.__re_detect_every
        if all_tracks_were_extended and not force_re_detection:
            self.get_logger().debug(f'Extend all tracks successfully: {{'
                                    f'"frame_id": {self.__frames_seen}, '
                                    f'"ids": {[t.get_id() for t in self.get_tracks()]}'
                                    f'}}')
            return [], []

        # Dismiss tracks that failed to meet all criteria
        done_ids = [self.__tracks[ti].get_id() for ti in failed_to_extend_inds + stick_out_roi_inds]
        self.get_logger().log(level=logging.INFO if done_ids else logging.DEBUG,
                              msg=f'Dismiss unqualified tracks: {{'
                                  f'"frame_id": {self.__frames_seen}, '
                                  f'"ids": {done_ids}'
                                  f'}}')
        self.__tracks = [t for ti, t in enumerate(self.__tracks)
                         if ti not in failed_to_extend_inds + stick_out_roi_inds]

        # Re-detect all the legit objects
        detections = [d for d in self.__detector.detect(frame=frame)
                      if self.__within(rect=self.__roi, center=_Track.center(rect=d['lt_wh']))]
        self.get_logger().debug(f'Detect objects of a frame: {{'
                                f'"frame_id": {self.__frames_seen}, '
                                f'"data": {detections}'
                                f'}}')

        done_inds = []
        for ti, t in enumerate(self.__tracks):
            di = t.select_best_match(iou_threshold=self.__iou_threshold, detections=detections)
            # Identify tracks that fail to match detections of the frame
            if di is None:
                done_inds.append(ti)
                continue

            # Identify trackless detections
            detections = detections[:di] + detections[di + 1:]

        # Dismiss tracks failed to match detection of the frame
        phantom_ids = [self.__tracks[ti].get_id() for ti in done_inds]
        self.get_logger().log(level=logging.INFO if done_inds else logging.DEBUG,
                              msg=f'Dismiss phantom tracks: {{'
                                  f'"frame_id": {self.__frames_seen}, '
                                  f'"ids": {phantom_ids}'
                                  f'}}')
        done_ids.extend(phantom_ids)
        self.__tracks = [t for ti, t in enumerate(self.__tracks) if ti not in done_inds]

        # Establish tracks for unbound detections of the frame
        est_ids = []
        for d in detections:
            try:
                t = _Track(logger=self.get_logger(), engine=self.__tracking_engine_creator(), frame=frame,
                           seed_detection=d)
            except Exception as e:
                self.__logger.error(f'{e}')
            else:
                est_ids.append(t.get_id())
                self.__tracks.append(t)

        self.get_logger().log(level=logging.INFO if est_ids else logging.DEBUG,
                              msg=f'Establish some tracks: {{'
                                  f'"frame_id": {self.__frames_seen}, '
                                  f'"ids": {est_ids}'
                                  f'}}')

        return done_ids, est_ids

    def get_tracks(self):
        """
        """
        return self.__tracks

    def get_roi(self):
        """
        """
        return self.__roi
