import main
import cv2

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def read_vid():
    vid = cv2.VideoCapture(0)

    first = None

    if not vid.isOpened:
        print("Could not load video")
        exit()

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=70, detectShadows=True)
    prev = None

    while True:
        ret, frame = vid.read()

        if not ret:
            print("Cannot find next frame")

        fg_mask = back_sub.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is None:
            prev = gray
            continue
        diff = cv2.absdiff(gray, prev)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        prev = gray

        combined_mask = cv2.bitwise_and(fg_mask, diff_mask)
        combined_mask = cv2.medianBlur(combined_mask, 5)
        combined_mask = cv2.dilate(combined_mask, None, iterations=2)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Moving Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    vid.release()
