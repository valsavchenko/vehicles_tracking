import abc
import os
import shutil

import cv2
import numpy as np


class Visualiser(metaclass=abc.ABCMeta):
    """
    """

    def __init__(self):
        self.__color_per_id = {}

    def __get_color(self, t_id):
        """
        """
        if t_id not in self.__color_per_id:
            self.__color_per_id[t_id] = [int(np.random.uniform(0, 255)) for _ in range(3)]

        return self.__color_per_id[t_id]

    def _draw(self, frame, done_ids, est_ids, tracker):
        """
        """
        for t in tracker.get_tracks():
            t_id = t.get_id()
            t_color = self.__get_color(t_id=t_id)

            last_rect = t.get_last_rect()
            thickness = 4 if t_id in est_ids else 2
            cv2.rectangle(img=frame, pt1=last_rect[:2], pt2=[last_rect[i] + last_rect[i + 2] for i in range(2)],
                          color=t_color, thickness=thickness)

            for p in t.get_centers():
                cv2.circle(img=frame, center=p, radius=5, color=t_color, lineType=cv2.FILLED)

        for d_id in done_ids:
            del self.__color_per_id[d_id]

    @abc.abstractmethod
    def act(self, frame, done_ids, est_ids, tracker):
        """
        """
        return False


class Viewer(Visualiser):
    """
    """

    def __init__(self):
        super().__init__()
        self.__window_name = 'Tracking some objects'

    def __del__(self):
        cv2.destroyWindow(winname=self.__window_name)

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        cv2.imshow(winname=self.__window_name, mat=frame)
        stop = cv2.waitKey(delay=1) & 0xFF == ord('q')

        return not stop


class Tracer(Visualiser):
    """
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
        """
        """
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        self.__frame_counter += 1
        file_name = os.path.join(self.__trace_folder_path, f'frame_{self.__frame_counter}.jpg')
        print(f'Annotate a frame at {file_name}')
        cv2.imwrite(filename=file_name, img=frame)

        return True


class Writer(Visualiser):
    """
    """

    def __init__(self, annotated_video_path, reader):
        super().__init__()

        print(f'Annotate video at {annotated_video_path}')

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
