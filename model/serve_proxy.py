import flask
from flask import Flask, request
import json
import jsonschema
from flask_cors import CORS, cross_origin
from serve_utils.ServeProxyUtilsFacadeSingleton import ServeProxyUtilsFacadeSingleton

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.post('/get_commentary')
@cross_origin()
def get_commentary_execution():
    instance = ServeProxyUtilsFacadeSingleton()
    try:
        instance.validate_request(request.data)
    except json.JSONDecodeError:
        return flask.jsonify({"error": "JSON payload is malformed"}), 400
    except jsonschema.ValidationError:
        return flask.jsonify({"error": "JSON payload does not respect schema specification"}), 400
    except ValueError as e:
        return flask.jsonify({"error": str(e)}), 400

    return app.response_class(instance.get_commentary(request.data), mimetype='plain/text')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
