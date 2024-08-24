import abc
import os
import shutil

import cv2


class Visualiser(metaclass=abc.ABCMeta):
    """
    Provides a common interface for all kinds of visualizers
    """

    def __init__(self, logger):
        self._logger = logger
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

        for track in tracker.get_tracks():
            t_id = track.get_id()
            t_color = self.__get_color(t_id=t_id)

            # Show an object at the frame
            last_rect = track.get_last_rect()
            thickness = 4 if t_id in est_ids else 2
            cv2.rectangle(img=frame, pt1=last_rect[:2], pt2=[last_rect[i] + last_rect[i + 2] for i in range(2)],
                          color=t_color, thickness=thickness)

            # Show object's id
            text = str(t_id)[:8]
            font_face = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1
            text_thickness = 2
            text_size, _ = cv2.getTextSize(text=text, fontFace=font_face, fontScale=font_scale,
                                           thickness=text_thickness)
            cv2.rectangle(img=frame, pt1=last_rect[:2],
                          pt2=[c + k * s for c, s, k in zip(last_rect[:2], text_size, [1, -1])],
                          color=t_color, thickness=cv2.FILLED)
            cv2.putText(img=frame, text=text, org=last_rect[:2], fontFace=font_face, fontScale=font_scale,
                        color=(0, 0, 0))

            # Show positions of the object at the previous frames
            for p in track.get_centers():
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

    def __init__(self, logger):
        super().__init__(logger=logger)

        self.__window_name = 'Tracking some objects'
        self.__width = 800
        self.__height = None

    def __del__(self):
        cv2.destroyWindow(winname=self.__window_name)

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        if self.__height is None:
            self.__height = int(self.__width * frame.shape[0] / frame.shape[1])

        frame = cv2.resize(src=frame, dsize=(self.__width, self.__height))
        cv2.imshow(winname=self.__window_name, mat=frame)

        # Wrap up, if user told so
        stop = cv2.waitKey(delay=1) & 0xFF == ord('q')
        return not stop


class Tracer(Visualiser):
    """
    Visualizes tracking results with separate frames
    """

    def __init__(self, logger, args):
        super().__init__(logger=logger)

        trace_folder_path = os.path.abspath(os.path.join(args['output_root'], 'trace'))
        try:
            os.mkdir(path=trace_folder_path)
        except FileExistsError:
            self._logger.warning(f'Purge the tracking location at {trace_folder_path}')
            shutil.rmtree(path=trace_folder_path)
            os.mkdir(path=trace_folder_path)

        self.__frame_counter = 0
        self.__trace_folder_path = trace_folder_path

    def act(self, frame, done_ids, est_ids, tracker):
        self._draw(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

        self.__frame_counter += 1
        file_path = os.path.abspath(os.path.join(self.__trace_folder_path, f'frame_{self.__frame_counter:05}.jpg'))
        self._logger.debug(f'Annotate a frame at {file_path}')
        cv2.imwrite(filename=file_path, img=frame)

        return True


class Writer(Visualiser):
    """
    Visualizes tracking results with a video
    """

    def __init__(self, logger, args, reader):
        super().__init__(logger=logger)

        video_name, video_ext = os.path.splitext(os.path.basename(args['video_path']))
        annotated_video_path = os.path.abspath(os.path.join(args['output_root'], f'{video_name}_annotated.avi'))
        self._logger.info(f'Annotate video at {annotated_video_path}')

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
