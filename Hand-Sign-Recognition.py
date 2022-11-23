import cv2
import time
import tensorflow as tf

# loading the model
model = tf.keras.models.load_model("./ASL-Model-1")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

def load_and_preprocess(img):
  img = tf.image.resize(img, (200, 200))

  return img


def fancyDraw(img, x, y, w, h, l=30, t=5, rt=1, color=(0, 255, 0)):
    x1, y1 = x + w, y + h

    cv2.rectangle(img, (0, 220), (200, 440), color, rt)
    # Top left x, y
    cv2.line(img, (x, y), (x + l, y), color, t)
    cv2.line(img, (x, y), (x, y + l), color, t)
    # Top right x1, y
    cv2.line(img, (x1, y), (x1 - l, y), color, t)
    cv2.line(img, (x1, y), (x1, y + l), color, t)
    # bottom left x, y1
    cv2.line(img, (x, y1), (x + l, y1), color, t)
    cv2.line(img, (x, y1), (x, y1 - l), color, t)
    # bottom right x1, y1
    cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

    return img

def itch(img, x, y, w, h, color=(0, 0, 0)):
    cv2.rectangle(img, (x, y), (w, h), thickness=cv2.FILLED, color=color)

    return img

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 130)

pTIme = 0
previous = time.time()
delta = 0
pred_show = ""
message = ""
while True:
    _, img = cap.read()

    # Get the current time, increase delta and update the previous variable
    current = time.time()
    delta += current - previous
    previous = current

    # Check if 5 seconds passed
    if delta > 5:
        # Operations on image
        # Reset the time counter
        delta = 0

        x, y, w, h = 0, 220, 200, 440
        _img_ = img.copy()
        # img = fancyDraw(img, x, y, w, h)


        filename = str(int(previous))
        img_pred = _img_[y:y + h, x:x + w]
        img_pred = cv2.resize(img_pred, (200, 200))
        img = itch(img, x, y, w, h)
        # cv2.imwrite(f"./RealTimeDetections/{filename}.jpg", img_pred)

        img_ = load_and_preprocess(img_pred)
        pred_prob = model.predict(tf.expand_dims(img_, axis=0))
        pred_class = class_names[pred_prob.argmax()]
        pred_prob = f"{pred_prob.max():.2f}"
        pred_show = f"Pred: {pred_class}, Prob: {pred_prob}%"

        if len(pred_class) == 1:
            message += pred_class
        elif pred_class == 'space':
            message += " "
        elif pred_class == 'del':
            message = message[:-1]

    if len(pred_show):
        cv2.putText(img, text=pred_show, org=(300, 33), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.8, color=(255, 0, 0), thickness=2)
        cv2.putText(img, text=message, org=(300, 73), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.8, color=(255, 0, 255), thickness=2)

    # FPS
    cTime = time.time()
    if (cTime - pTIme) != 0:
        fps = 1 / (cTime - pTIme)
        pTIme = cTime

        cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 33), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=(0, 0, 0), thickness=2)

    cv2.rectangle(img, (0, 220), (200, 440), color=(0, 255, 0), thickness=2, lineType=1)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
