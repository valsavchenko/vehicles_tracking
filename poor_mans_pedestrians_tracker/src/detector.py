import cv2
import numpy as np

from timer import timer


class Detector:
    """
    """

    def __init__(self, settings):
        """
        Sets up a model to infer with
        """
        # Read pre-processing params
        self.__pre_processing_params = {
            'scalefactor': eval(settings['scale_factor']),
            'size': tuple(settings['size']),
            'mean': tuple(settings['mean']),
            'swapRB': settings['swap_rb'],
            'crop': settings['crop']
        }

        # Establish the model
        self.__model = cv2.dnn.readNetFromONNX(onnxFile=settings['weights_file_path'])
        layer_names = self.__model.getLayerNames()
        self.__output_layers = [layer_names[ol - 1] for ol in self.__model.getUnconnectedOutLayers()]

        # Read post-processing params
        self.__confidence_threshold = settings['confidence_threshold']
        labels = open(file=settings['labels_file_path']).read().rstrip('\n').split(sep='\n')
        self.__class_id = labels.index(settings['class_label'])
        self.__post_processing_params = {
            'score_threshold': settings['score_threshold'],
            'nms_threshold': settings['nms_threshold'],
            'eta': settings['eta'],
            'top_k': settings['top_k']
        }

    def _pre_process(self, frame):
        """
        """
        blob = cv2.dnn.blobFromImage(image=frame, **self.__pre_processing_params)
        return blob

    @timer(label='Obj detection')
    def detect(self, frame):
        """
        """
        blob = self._pre_process(frame=frame)

        self.__model.setInput(blob=blob)
        ([objects, ],) = self.__model.forward(outBlobNames=self.__output_layers)

        frame_h, frame_w = frame.shape[:2]
        blob_h, blob_w = self.__pre_processing_params['size']
        coeff_h, coeff_w = frame_h / blob_h, frame_w / blob_w
        class_objs = self._post_process(objects=objects, coeff_h=coeff_h, coeff_w=coeff_w)

        return class_objs

    def _post_process(self, objects, coeff_h, coeff_w):
        """
        """
        # Discard unconfident detections
        objects = [o for o in objects if self.__confidence_threshold < o[4]]

        # Discard objects that don't belong to the class
        objects = [o for o in objects if self.__class_id == np.argmax(a=o[5:])]

        # Discard the class objects with low score
        objects = [o for o in objects if self.__post_processing_params['score_threshold'] < o[5:][self.__class_id]]

        # Suppress non-maximal detections of the class
        boxes = []
        for obj in objects:
            c_x, c_y, w, h = obj[:4]
            width = int(w * coeff_w)
            height = int(h * coeff_h)
            left = int(c_x * coeff_w - width / 2)
            top = int(c_y * coeff_h - height / 2)
            boxes.append((left, top, width, height))
        scores = [float(o[4]) for o in objects]
        max_box_ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, **self.__post_processing_params)

        class_objs = [{'lt_wh': boxes[bi], 'score': scores[bi]} for bi in max_box_ids]
        return class_objs
