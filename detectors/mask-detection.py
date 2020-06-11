import imutils
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

from detection.configs import Config
from detection.detections import DetectionType, DetectionResult, limit_detection_output
from models.model_io import load_checkpoint

# PATHS
MASK_RECOGNITION_PATH = "models/saved_models/mask_recognition.pth"
PROTOTXT_PATH = 'models/saved_models/deploy.prototxt.txt'
CAFFEMODEL_PATH = 'models/saved_models/res10_300x300_ssd_iter_140000.caffemodel'

# CONST VALUES
MIN_FACE_RECOG_CONFIDENCE = 0.3

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MaskDetector:
    def __init__(self):
        self.mask_model = load_checkpoint(MASK_RECOGNITION_PATH)
        self.face_recog_model = cv2.cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        self.config = Config()

    def run(self, frame_size=800):
        capture = self._prepare_capture()

        continue_running = True

        while continue_running:
            frame = self._read_frame(capture)
            if frame is None:
                break
            processed_frame, blob = self._preprocess_frame(frame, frame_size)
            face_detections = self._get_face_recog_detections(blob)
            mask_detections = self._analyze_frame(face_detections, processed_frame)
            print(mask_detections)
            self._show_frame(processed_frame, mask_detections)
            continue_running = self._check_quit()

        self._close_capture(capture)

    def _analyze_frame(self, face_detections, frame):
        face_boundaries = self._get_face_boundaries(face_detections, frame)
        mask_predictions = self._get_mask_predictions(face_boundaries, frame)
        return mask_predictions

    def _preprocess_frame(self, resized_frame, frame_width):
        resized_frame = imutils.resize(resized_frame, width=frame_width)
        blob = cv2.cv2.dnn.blobFromImage(
            image=cv2.cv2.resize(resized_frame, (400, 400)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0))
        return resized_frame, blob

    def _get_face_recog_detections(self, blob):
        self.face_recog_model.setInput(blob)
        return self.face_recog_model.forward()

    def _get_face_boundaries(self, face_detections, frame):
        (frame_h, frame_w) = frame.shape[:2]
        face_boundaries = []
        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence < MIN_FACE_RECOG_CONFIDENCE:
                break

            # compute the (x, y)-coordinates of the bounding box for the object
            box = np.vectorize(limit_detection_output)(face_detections[0, 0, i, 3:7]) * np.array(
                [frame_w, frame_h, frame_w, frame_h])
            (startX, startY, endX, endY) = box.astype("int")
            coords = {
                'x': max(startX, 0),
                'y': max(startY, 0),
                'x+w': max(startX, 0) + endX - startX,
                'y+h': max(startY, 0) + endY - startY
            }

            face_boundaries.append(coords)
        return face_boundaries

    def _get_mask_predictions(self, face_boundaries, frame):
        if not len(face_boundaries):
            return [DetectionResult(DetectionType.NOTHING)]

        mask_results = []

        for face_boundary in face_boundaries:

            prediction = self._get_mask_detection(face_boundary, frame)

            if prediction == 0:
                mask_results.append(DetectionResult(DetectionType.MASK_ON,
                                                    (face_boundary['x'], face_boundary['y']),
                                                    (face_boundary['x+w'], face_boundary['y+h'])))

            elif prediction == 1:
                mask_results.append(DetectionResult(DetectionType.MASK_OFF,
                                                    (face_boundary['x'], face_boundary['y']),
                                                    (face_boundary['x+w'], face_boundary['y+h'])))
        return mask_results

    def _get_mask_detection(self, face_boundary, frame):
        cropped_img = frame[face_boundary['y']:face_boundary['y+h'], face_boundary['x']:face_boundary['x+w']]
        if not len(cropped_img):
            return None
        pil_image = Image.fromarray(cropped_img, mode="RGB")
        pil_image = train_transforms(pil_image)
        image = pil_image.unsqueeze(0)
        result = self.mask_model(image.to('cuda'))
        _, maximum = torch.max(result.data, 1)
        prediction = maximum.item()
        return prediction

    def _prepare_capture(self):
        return cv2.cv2.VideoCapture(0)

    def _read_frame(self, capture):
        if capture.isOpened():
            result, frame = capture.read()
            return frame if result else None
        return None

    def _check_quit(self):
        if (cv2.cv2.waitKey(1) & 0xFF) == ord('q'):
            return False
        return True

    def _close_capture(self, capture):
        capture.release()
        cv2.cv2.destroyAllWindows()

    def _show_frame(self, frame, mask_detections):
        for detection in mask_detections:
            if detection.result == DetectionType.MASK_ON:
                cv2.cv2.putText(frame, "Masked", (detection.point_start[0], detection.point_start[1] - 10),
                                self.config.font, self.config.font_scale, self.config.green, self.config.thickness)
                cv2.cv2.rectangle(frame, detection.point_start, detection.point_end, self.config.green, 2)

            elif detection.result == DetectionType.MASK_OFF:
                cv2.cv2.putText(frame, "No mask", (detection.point_start[0], detection.point_start[1] - 10),
                                self.config.font, self.config.font_scale, self.config.red, self.config.thickness)
                cv2.cv2.rectangle(frame, detection.point_start, detection.point_end, self.config.red, 2)

        cv2.cv2.imshow('frame', frame)


if __name__ == '__main__':
    f = MaskDetector()
    f.run()
