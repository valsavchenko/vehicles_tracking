import argparse
import json
import logging
import os

import cv2
from _ped_trk.visualiser import Viewer, Tracer, Writer
from pm_pedestrians_tracker.src.detector import Detector
from pm_pedestrians_tracker.src.tracker import Tracker


def _collect_arguments():
    """
    """
    parser = argparse.ArgumentParser(description='Tracks pedestrians on a video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', type=str,
                        default=os.path.join('samples', 'pedestrians_0.mp4'),
                        help='A path to a video to analyze')
    parser.add_argument('--roi', type=int, nargs=4, default=None,
                        help='A region to watch out for objects at. '
                             'Must be specified as a left-top-width-height tuple')
    parser.add_argument('--settings_file_path', type=str,
                        default=os.path.join('pm_pedestrians_tracker', 'config', 'settings.json'),
                        help='A path to a file with settings for the detector')
    parser.add_argument('--visualizer_type', type=str, choices=['viewer', 'tracer', 'writer'], default='tracer',
                        help='A way to present results')
    parser.add_argument('--output_root', type=str, required=True, help='A path to a output root')

    return vars(parser.parse_args())


def _secure_output_root(args):
    """
    """
    os.makedirs(name=args['output_root'], exist_ok=True)


def _setup_logging(args):
    """
    """
    logger = logging.getLogger('Tracker')
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s')
    logger.setLevel(logging.DEBUG)

    # Configure console logging
    stdout = logging.StreamHandler()
    stdout.setLevel(level=logging.INFO)
    stdout.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=stdout)

    # Configure file logging
    log_path = os.path.abspath(os.path.join(args['output_root'], 'tracker.log'))
    file = logging.FileHandler(filename=log_path, encoding='utf-8', mode='a')
    file.setLevel(level=logging.DEBUG)
    file.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=file)
    logger.info(f'Store logs at {log_path}')

    return logger


def _create_frames_reader(args):
    """
    """
    reader = None

    # Check the input video
    video_path = args['video_path']
    reader = cv2.VideoCapture(filename=video_path)
    if not reader.isOpened():
        raise ValueError(f'Failed to open {video_path}')

    return reader


def _create_visualiser(logger, args, reader):
    """
    """
    vt = args['visualizer_type']
    visualiser = {
        'tracer': lambda: Tracer(logger=logger, args=args),
        'writer': lambda: Writer(logger=logger, args=args, reader=reader),
        'viewer': lambda: Viewer(logger=logger)
    }[vt]()

    return visualiser


if __name__ == '__main__':
    args = _collect_arguments()
    _secure_output_root(args=args)
    logger = _setup_logging(args=args)

    reader = _create_frames_reader(args=args)

    settings = json.load(fp=open(file=args['settings_file_path']))
    detector = Detector(logger=logger, settings=settings['detector'])
    tracker = Tracker(logger=logger, detector=detector, settings=settings['tracker'], roi=args['roi'])

    visualiser = _create_visualiser(logger=logger, args=args, reader=reader)

    status = True
    while status:
        status, frame = reader.read()
        if not status:
            continue

        done_ids, est_ids = tracker.track(frame=frame)
        status = visualiser.act(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

    reader.release()
