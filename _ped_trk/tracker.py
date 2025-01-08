import abc


class Track(metaclass=abc.ABCMeta):
    """
    Provides a common interface for a track
    """

    @abc.abstractmethod
    def get_id(self):
        """
        """
        pass

    @abc.abstractmethod
    def get_centers(self):
        """
        """
        pass

    @abc.abstractmethod
    def get_last_rect(self):
        """
        """
        pass


class Tracker(metaclass=abc.ABCMeta):
    """
    Provides a common interface for a tracker
    """

    @abc.abstractmethod
    def track(self, frame):
        """
        """
        pass

    @abc.abstractmethod
    def get_tracks(self):
        """
        """
        pass

    @abc.abstractmethod
    def get_roi(self):
        """
        """
        pass
