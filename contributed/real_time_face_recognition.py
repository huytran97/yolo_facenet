# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""

import argparse
import sys
import time
import numpy as np

import cv2

import face


def add_overlays(frame, faces, frame_rate, COLORS):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            color = [int(c) for c in COLORS[faces.index(face)]]
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          color, 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()
    outputFile = "result.avi"
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if args.debug:
        print("Debug enabled")
        face.debug = True
    COLORS = np.random.randint(0, 255, size=(3, 3),
		    dtype="uint8")
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
        
        
        add_overlays(frame, faces, frame_rate, COLORS)

        # Write the frame with the detection boxes
        vid_writer.write(frame.astype(np.uint8))

        frame_count += 1
        cv2.imshow('Realtime Regconition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
