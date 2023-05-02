from flask import Flask, render_template
import numpy as np
import requests
import cv2
from edge_impulse_linux.image import ImageImpulseRunner
app = Flask(__name__,
            static_url_path='', 
            static_folder='./assets',
            template_folder='./templates')

runner = None
modelfile = './models/ashan_modelfile.eim'

async def detectFace(f_img):
    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()    
            labels = model_info['model_parameters']['labels']
            image = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
            features, cropped = runner.get_features_from_image(image)
            res = runner.classify(features)        
            if "classification" in res["result"].keys():
                for label in labels:
                    score = res['result']['classification'][label]
                print('', flush=True)
            elif "bounding_boxes" in res["result"].keys():
                detected_labels = {}      
                for bb in res["result"]["bounding_boxes"]:
                    if bb['label'] in detected_labels:
                        detected_labels[bb['label']]['count'] += 1
                        detected_labels[bb['label']]['score'] += bb['value']
                    else:
                        detected_labels[bb['label']] = {'count': 1, 'score': bb['value']}
                    cropped = cv2.rectangle(
                        cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

            mean_valued_labels = {}
            for label in detected_labels:
                mean_valued_labels[label] = detected_labels[label]['score'] / detected_labels[label]['count']

            return mean_valued_labels

        finally:
            if (runner):
                runner.stop()

@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    return render_template('face_detection.html')


@app.route('/detect', methods=['GET'])
async def capture():
    img_url = "http://192.168.1.100/capture"
    detected_labels = {}
    try:
        img_resp = requests.get(img_url)
    except requests.exceptions.Timeout:
        return 'Request Timeout'     
    except requests.exceptions.ConnectionError:
        return 'Camera Connection Error'
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    det_labels = await detectFace(img)
    for label in det_labels:
        if label in detected_labels:
            detected_labels[label]['count'] += 1
            detected_labels[label]['score'] += det_labels[label]
        else:
            detected_labels[label] = {'count': 1, 'score': det_labels[label]}
    mean_valued_labels = {}
    for label in detected_labels:
        mean_valued_labels[label] = detected_labels[label]['score'] / detected_labels[label]['count']
    mean_valued_labels = {k: v for k, v in mean_valued_labels.items() if v > 0.8}
    print (mean_valued_labels)
    if len(mean_valued_labels) > 0:
         unlock_door = requests.get('http://192.168.1.181/unlock')
         if unlock_door.status_code == 200:
            return {'success': 'true' }
    else :
      return {'success': 'false' }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
