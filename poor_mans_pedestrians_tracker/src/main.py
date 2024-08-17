import argparse
import json
import os

import cv2

from detector import Detector
from tracker import Tracker
from visualiser import Viewer, Tracer, Writer


def _collect_arguments():
    """
    """
    parser = argparse.ArgumentParser(description='Tracks pedestrians on a video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', type=str, default=os.path.join('..', 'samples', 'pedestrians_0.mp4'),
                        help='A path to a video to analyze')
    parser.add_argument('--roi', type=int, nargs=4, default=None,
                        help='A region to watch out for objects at. '
                             'Must be specified as a left-top-width-height tuple')
    parser.add_argument('--settings_file_path', type=str, default=os.path.join('..', 'config', 'settings.json'),
                        help='A path to a file with settings for the detector')
    parser.add_argument('--visualizer_type', type=str, choices=['viewer', 'tracer', 'writer'], default='tracer',
                        help='A way to supply results')
    parser.add_argument('--output_root', type=str, required=True, help='A path to a output root')

    return vars(parser.parse_args())


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


def _create_visualiser(args, reader):
    """
    """
    visualiser = None

    vt = args['visualizer_type']
    if 'tracer' == vt:
        tracer_folder_path = os.path.join(args['output_root'], 'trace')
        visualiser = Tracer(trace_folder_path=tracer_folder_path)
    elif 'writer' == vt:
        video_name, video_ext = os.path.splitext(os.path.basename(args['video_path']))
        annotated_video_path = os.path.join(args['output_root'], video_name + '_annotated.avi')
        visualiser = Writer(annotated_video_path=annotated_video_path, reader=reader)
    else:
        visualiser = Viewer()

    return visualiser


if __name__ == '__main__':
    args = _collect_arguments()
    settings = json.load(fp=open(file=args['settings_file_path']))

    reader = _create_frames_reader(args=args)
    detector = Detector(settings=settings['detector'])
    tracker = Tracker(detector=detector, settings=settings['tracker'], roi=args['roi'])
    visualiser = _create_visualiser(args=args, reader=reader)

    status = True
    while status:
        status, frame = reader.read()
        if not status:
            continue

        done_ids, est_ids = tracker.track(frame=frame)
        status = visualiser.act(frame=frame, done_ids=done_ids, est_ids=est_ids, tracker=tracker)

    reader.release()
