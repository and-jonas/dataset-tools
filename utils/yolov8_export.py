import fiftyone as fo
from fiftyone.utils.yolo import YOLOv5DatasetExporter, YOLOAnnotationWriter, _write_file_lines
import fiftyone.core.labels as fol
import warnings


class YOLOv8KeypointExporter(YOLOv5DatasetExporter):
    def __init__(self, export_dir=None, split="val", data_path=None, labels_path=None, yaml_path=None, export_media=None, rel_dir=None, classes=None, include_confidence=False, image_format=None, include_path=True, image_size=None, class_sizes=None):
        super().__init__(export_dir, split, data_path, labels_path, yaml_path, export_media, rel_dir, classes, include_confidence, image_format, include_path)
        self.class_sizes = class_sizes
        self.image_size = image_size

    def setup(self):
        super().setup()
        self._writer = YOLOKeypointAnnotationWriter(self.class_sizes, self.image_size)

    @property
    def label_cls(self):
        return fol.Keypoints



class YOLOKeypointAnnotationWriter(YOLOAnnotationWriter):
    """Class for writing annotations in YOLO-style TXT format."""
    def __init__(self, class_sizes, image_size):
        super().__init__()
        self.class_sizes = class_sizes
        self.image_size = image_size

    # make this consider class and image size
    def _make_keypoint_yolo_row(self, points_pair, target, confidence=None):
        x, y = points_pair
        w = self.class_sizes[target]/self.image_size[0]  
        h = self.class_sizes[target]/self.image_size[1] 
        row = "%d %f %f %f %f %f %f 1" % (target, x, y, w, h, x, y)  # the 1 at the end enables predicting confidence

        return row


    def write(
        self,
        keypoints,
        txt_path,
        labels_map_rev,
        dynamic_classes=False,
        include_confidence=False,
    ):
        """Writes the detections to disk.
        Args:
            detections: a :class:`fiftyone.core.labels.Detections` instance
            txt_path: the path to write the annotation TXT file
            labels_map_rev: a dictionary mapping class label strings to target
                integers
            dynamic_classes (False): whether to dynamically add new labels to
                ``labels_map_rev``
            include_confidence (False): whether to include confidences in the
                export, if they exist
        """
        rows = []
        for keypoint in keypoints.keypoints:
            label = keypoint.label

            if dynamic_classes and label not in labels_map_rev:
                target = len(labels_map_rev)
                labels_map_rev[label] = target
            elif label not in labels_map_rev:
                msg = (
                    "Ignoring detection with label '%s' not in provided "
                    "classes" % label
                )
                warnings.warn(msg)
                continue
            else:
                target = labels_map_rev[label]

            if include_confidence:
                confidence = keypoint.confidence
            else:
                confidence = None
            for point_pair in keypoint.points:
                row = self._make_keypoint_yolo_row(
                    point_pair, target, confidence=confidence
                )
                rows.append(row)

        _write_file_lines(rows, txt_path)