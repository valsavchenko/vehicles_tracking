import abc
import os
import shutil

import cv2


class Visualiser(metaclass=abc.ABCMeta):
    """
    Provides a common interface for all kinds of visualizers
    """

    def __init__(self):
        self.__color_per_id = {}

    def __get_color(self, t_id):
        """
        Computes a stable color for a track with stable id
        """
        if t_id not in self.__color_per_id:
            self.__color_per_id[t_id] = [f % 255 for f in t_id.fields[:3]]

        return self.__color_per_id[t_id]

    def _draw(self, frame, done_ids, est_ids, tracker):
        """
        Draws tracking results for a frame at it
        """
        # Show ROI
        roi = tracker.get_roi()
        cv2.rectangle(img=frame, pt1=roi[:2], pt2=[roi[i] + roi[i + 2] for i in range(2)], color=(0, 255, 0),
                      thickness=8)

        for t in tracker.get_tracks():
            t_id = t.get_id()
            t_color = self.__get_color(t_id=t_id)

            # Show object at the frame
            last_rect = t.get_last_rect()
            thickness = 4 if t_id in est_ids else 2
            cv2.rectangle(img=frame, pt1=last_rect[:2], pt2=[last_rect[i] + last_rect[i + 2] for i in range(2)],
                          color=t_color, thickness=thickness)

            # Show positions of the object at the previous frames
            for p in t.get_centers():
                cv2.circle(img=frame, center=p, radius=thickness, color=t_color, lineType=cv2.FILLED)

        # Dismiss colors for deceased tracks
        for d_id in done_ids:
            del self.__color_per_id[d_id]

    @abc.abstractmethod
    def act(self, frame, done_ids, est_ids, tracker):
        """
        Visualizes tracking results for a frame in a specific way
        """
        return False


class Viewer(Visualiser):
    """
    Visualizes tracking results with a live video via GUI
    """

    def __init__(self):
        super().__init__()
        self.__window_name = 'Tracking some objects'

    def __del__(self):
        cv2.destroyWindow(winname=self.__window_name)

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        cv2.imshow(winname=self.__window_name, mat=frame)

        # Wrap up, if user told so
        stop = cv2.waitKey(delay=1) & 0xFF == ord('q')
        return not stop


class Tracer(Visualiser):
    """
    Visualizes tracking results with separate frames
    """

    def __init__(self, trace_folder_path):
        super().__init__()

        try:
            os.mkdir(path=trace_folder_path)
        except FileExistsError:
            print(f'Purge the tracking location at {trace_folder_path}')
            shutil.rmtree(path=trace_folder_path)
            os.mkdir(path=trace_folder_path)

        self.__frame_counter = -1
        self.__trace_folder_path = trace_folder_path

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        self.__frame_counter += 1
        file_name = os.path.join(self.__trace_folder_path, f'frame_{self.__frame_counter}.jpg')
        print(f'Annotate a frame at {file_name}')
        cv2.imwrite(filename=file_name, img=frame)

        return True


class Writer(Visualiser):
    """
    Visualizes tracking results with a video
    """

    def __init__(self, annotated_video_path, reader):
        super().__init__()

        print(f'Annotate video at {annotated_video_path}')

        # Force Motion JPEG to mitigate codecs hassles
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = int(reader.get(cv2.CAP_PROP_FPS))
        size = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__writer = cv2.VideoWriter(filename=annotated_video_path, fourcc=fourcc, fps=fps, frameSize=size)

    def __del__(self):
        self.__writer.release()

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        self.__writer.write(image=frame)

        return True
