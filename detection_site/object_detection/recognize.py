# recognize_function
import cv2


def recognize_function():
    source = cv2.VideoCapture(0)
    assert source.isOpened()
    while True:

        # time.sleep(0.01)
        ret, img = source.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.Canny(img, 80, 100)
        faces = cv2.CascadeClassifier('/Users/tsars/PycharmProjects/Diplom_django/detection_site/object_detection/haarcascade_profileface.xml')
        results = faces.detectMultiScale(gray, scaleFactor=1.40, minNeighbors=3)

        for (x, y, w, h) in results:  # x, y, w, h - размеры квадрата выделяющего лица
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

        cv2.imshow("Result", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    source.release()
# recognize_function()
