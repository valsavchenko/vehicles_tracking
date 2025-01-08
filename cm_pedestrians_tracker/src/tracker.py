import logging
import random
import uuid

import _ped_trk.timer
import _ped_trk.tracker


class _Track(_ped_trk.tracker.Track, _ped_trk.timer.Timeable):
    """
    """
    __STABLE_ID_GENERATOR = random.Random(x=7292)

    @classmethod
    def __get_unique_stable_id(cls):
        """
        """
        return uuid.UUID(int=cls.__STABLE_ID_GENERATOR.getrandbits(128), version=4)

    @staticmethod
    def center(rect):
        """
        """
        l, t, w, h = rect
        return l + w // 2, t + h // 2

    @_ped_trk.timer.timer(label='trk_creation')
    def __init__(self, logger, frame, seed_detection):
        """
        """
        super().__init__(logger=logger)

        self.__points = [seed_detection['lt_wh'], ]
        self.__id = self.__get_unique_stable_id()

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


class Tracker(_ped_trk.tracker.Tracker, _ped_trk.timer.Timeable):
    """
    """

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
        super().__init__(logger=logger)

        self.__detector = detector

        self.__roi = roi
        self.get_logger().info(f'Establish a tracker: {{'
                               f'"roi": {self.__roi}'
                               f'}}')

        self.__frames_seen = 0
        self.__tracks = {}

    def __add_track(self, track):
        """
        """
        self.__tracks[track.get_id()] = track

    def __dismiss_track(self, t_id):
        """
        """
        del self.__tracks[t_id]

    def __dismiss_tracks(self, t_ids):
        """
        """
        [self.__dismiss_track(t_id=t_id) for t_id in t_ids]

    def __get_track_ids(self):
        """
        """
        return sorted(self.__tracks.keys())

    @_ped_trk.timer.timer(label='frame_handling')
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

        # Dismiss all known tracks
        done_ids = [t_id for t_id in self.__get_track_ids()]
        self.get_logger().log(level=logging.INFO if done_ids else logging.DEBUG,
                              msg=f'Dismiss all active tracks: {{'
                                  f'"frame_id": {self.__frames_seen}, '
                                  f'"ids": {done_ids}'
                                  f'}}')
        self.__dismiss_tracks(t_ids=done_ids)

        # Detect all the legit objects
        detections = [d for d in self.__detector.detect(frame=frame)
                      if self.__within(rect=self.__roi, center=_Track.center(rect=d['lt_wh']))]
        self.get_logger().debug(f'Detect objects of a frame: {{'
                                f'"frame_id": {self.__frames_seen}, '
                                f'"data": {detections}'
                                f'}}')

        # Establish tracks for all detections of the frame
        est_ids = []
        for d in detections:
            try:
                track = _Track(logger=self.get_logger(), frame=frame, seed_detection=d)
            except Exception as e:
                self.__logger.error(f'{e}')
            else:
                self.__add_track(track=track)
                est_ids.append(track.get_id())

        self.get_logger().log(level=logging.INFO if est_ids else logging.DEBUG,
                              msg=f'Establish some tracks: {{'
                                  f'"frame_id": {self.__frames_seen}, '
                                  f'"ids": {est_ids}'
                                  f'}}')

        return done_ids, est_ids

    def get_tracks(self):
        """
        """
        return self.__tracks.values()

    def get_roi(self):
        """
        """
        return self.__roi
