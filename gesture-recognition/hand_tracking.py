import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self, static_image_mode=False, max_num_hands=2, 
                min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence 

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands


    def find_hands(self, image, draw=True):
        with self.mp_hands.Hands(
            static_image_mode = self.static_image_mode,
            max_num_hands = self.max_num_hands,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence) as hands:

            # Convert BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not 
            # writeable to pass by reference.
            image.flags.writeable = False
            self.results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.results.multi_hand_landmarks:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    if draw:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS)

        return image


    def find_positions(self, image, hand_number=0, draw=True):
        landmark_list = []

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[hand_number]
            for idx, landmark in enumerate(hand_landmarks.landmark):
                height, width, channels = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return landmark_list

