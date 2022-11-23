import cv2
import time
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

cap = cv2.VideoCapture(0)

# loading the model
model = tf.keras.models.load_model("./ASL-Model-1")

def load_and_preprocess(img):
	img = tf.image.resize(img, (200, 200))

	return img

def itch(img, x, y, w, h, color=(0, 0, 0)):
	cv2.rectangle(img, (x, y), (w, h), thickness=cv2.FILLED, color=color)

	return img
# Prediction Function
def Pred(img, previous, message, save=False):
	x, y, w, h = 0, 220, 200, 440
	_img_ = img.copy()

	img_pred = _img_[y:y + h, x:x + w]
	# img_pred = cv2.resize(img_pred, (200, 200))
	img = itch(img, x, y, w, h)
	if save:
		filename = str(int(previous))
		cv2.imwrite(f"./RealTimeDetections/{filename}.jpg", img_pred)

	img_ = load_and_preprocess(img_pred)
	with tf.device('/cpu:0'):
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
	
	
	# if len(pred_show):
	# 	cv2.putText(img, text=pred_show, org=(300, 33), fontFace=cv2.FONT_HERSHEY_PLAIN,
	# 				fontScale=1.8, color=(255, 0, 0), thickness=2)
	# 	cv2.putText(img, text=message, org=(300, 73), fontFace=cv2.FONT_HERSHEY_PLAIN,
	# 				fontScale=1.8, color=(255, 0, 255), thickness=2)

	return img, message, pred_show
# Image Transformation Function
def transform(img):
	cv2.rectangle(img, (0, 220), (200, 440), color=(0, 255, 0), thickness=2, lineType=1)

	return img

def FPS(img, fps):
	cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 33), fontFace=cv2.FONT_HERSHEY_PLAIN,
				fontScale=2, color=(0, 0, 0), thickness=2)

	return img

def gen_frames():
	pTime=0
	previous = time.time()
	delta = 0
	message = ""
	pred_show = ""
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
			img, message, pred_show = Pred(img, previous, message)
		
		cv2.putText(img, text=pred_show, org=(300, 33), fontFace=cv2.FONT_HERSHEY_PLAIN,
					fontScale=1.8, color=(255, 0, 0), thickness=2)
		cv2.putText(img, text=message, org=(300, 73), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.8, color=(255, 0, 255), thickness=2)

		# FPS
		cTime = time.time()
		if (cTime - pTime) != 0:
			fps = 1 / (cTime - pTime)
			pTime = cTime

			img = FPS(img, fps)

		img = transform(img)
		ret, buffer = cv2.imencode('.jpg', img)
		img = buffer.tobytes()
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
	return render_template('index.html')

# video feed

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
	app.run(debug=True)
