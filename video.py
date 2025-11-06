import main
import cv2



def read_vid():
    vid = cv2.VideoCapture(0)

    first = None

    if not vid.isOpened:
        print("Could not load video")
        exit()

    while True:
        ret, frame = vid.read()

        if not ret:
            print("Cannot find next frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        if first is None:
            first = gray
            continue

        frame_delta = cv2.absdiff(first, gray)
        thresh = cv2.threshold(frame_delta, 120, 200, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    vid.release()
