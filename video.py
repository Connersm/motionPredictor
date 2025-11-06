import main
import cv2



def read_vid():
    vid = cv2.VideoCapture(0)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('vid/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

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

        out.write(frame)

        cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
read_vid()