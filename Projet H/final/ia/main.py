import classify
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import os

UPLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_ECG_EXTENSIONS = {'txt', 'csv', 'tsv', 'npy'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)

validation_image_map = [
  "pictures/tiger.jpg",
  "pictures/bus.jpg",
  "pictures/cat.jpg",
  "pictures/piano.jpg",
  "pictures/whale.jpg",
]

@app.route ('/config', methods=['POST'])
def doConfiguration():
   print ("configuring")
   model_name = "efficientnetv2-s"
   if 'graph' in request.args:
      model_name = request.args['graph']
   elif 'graph' in request.form:
      model_name = request.form['graph']
   try:
      classify.config(model_name);
      print ("done")
      return { "status" : "ok", "data": "configuration is done" }
   except Exception as e:
      print(f"Error in config: {e}")
      return { "status" : "error", "data": str(e) }, 500
   
def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_ecg_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_ECG_EXTENSIONS

@app.route ('/classify', methods=['POST'])
def doClassification():
   print ("classifying")
   if 'picture' not in request.files:
      return ("bad url", 400)
   picture = request.files['picture']
   if not picture :
      return ("picture is wrong", 400)
   if picture.filename == '':
      return ("picture is empty", 400)
   if not allowed_file(picture.filename):
      return ("picture is empty", 400)
   filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(picture.filename))
   picture.save(filename)
   result = classify.classify(filename);
   print (result)
   print ("done")
   return { "status" : "ok", "data": result, "filename": filename }

@app.route ('/classify_ecg', methods=['POST'])
def doEcgClassification():
   print ("classifying ecg")
   if request.is_json:
      payload = request.get_json(silent=True) or {}
      signal = payload.get("signal")
      if signal is None:
         return ("signal is missing", 400)
      result = classify.classify_ecg_from_array(signal)
      print (result)
      print ("done")
      return { "status" : "ok", "data": result }

   if 'signal' not in request.files:
      return ("bad url", 400)
   signal_file = request.files['signal']
   if not signal_file :
      return ("signal is wrong", 400)
   if signal_file.filename == '':
      return ("signal is empty", 400)
   if not allowed_ecg_file(signal_file.filename):
      return ("signal is empty", 400)
   filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(signal_file.filename))
   signal_file.save(filename)
   result = classify.classify_ecg_file(filename)
   print (result)
   print ("done")
   return { "status" : "ok", "data": result, "filename": filename }

@app.route ('/validate', methods=['POST'])
def doValidation():
   print ("validating")
   picture_name = "picture.jpg"
   if 'id' in request.args:
      index = int(request.args['id'])
      if index >= 0 and index < len(validation_image_map):
         picture_name = validation_image_map[index]
   result = classify.classify(picture_name);
   print (result)
   print ("done")
   return { "status" : "ok", "data": result }
  
@app.route ('/', methods=['GET'])
def doHome():
   return '''
   <!doctype html>
   <html>
	   <head>
	      <title>Classifying</title>
              <meta charset="UTF-8" />
	   </head>
	   <body>
	      <h1>Classifying</h1>
	      <h2>Configure a classifier</h2>
	      <form method=post action="/config" enctype="application/x-www-form-urlencoded" name="config">
		 <label for="graph">Graph name ?</label>
		 <select name="graph" id="graph">
		    <option value="efficientnetv2-s" selected="">efficientnetv2-s</option>
		    <option value="efficientnetv2-m">efficientnetv2-m</option>
          <option value="MLP.keras">MLP.keras (ECG)</option>
          <option value="CNN1D.keras">CNN1D.keras (ECG)</option>
          <option value="RNN_LSTM.keras">RNN_LSTM.keras (ECG)</option>
		 </select>
		 <input type="submit" value="Config">
	      </form>
	      <h2>Classify a picture</h2>
	      <form method=post action="/classify" enctype="multipart/form-data" name="classify">
		 <input type="file" name="picture" onchange="loadFile(event)">
		 <input type="submit" value="Classify">
		 <img id="preview" src="" alt="">
	      </form>
         <h2>Classify an ECG signal</h2>
         <form method=post action="/classify_ecg" enctype="multipart/form-data" name="classify_ecg">
       <input type="file" name="signal">
       <input type="submit" value="Classify ECG">
         </form>
	      <script>
		 var loadFile = function(event) {
		    var preview = document.getElementById('preview');
		    preview.src = URL.createObjectURL(event.target.files[0]);
		    preview.onload = function() {
		       URL.revokeObjectURL(preview.src); // free memory
		    };
		 };
	      </script>
	   </body>
   </html>
   '''

if __name__ == '__main__' :
   print ("starting")
   app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
   print ("done")

