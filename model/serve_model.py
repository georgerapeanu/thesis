import flask
from flask import Flask, request
import json
import jsonschema
from serve_utils.ServeModelUtilsFacadeSingleton import ServeModelUtilsFacadeSingleton

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.post('/get_commentary_execution')
def get_commentary_execution():
    instance = ServeModelUtilsFacadeSingleton()
    try:
        instance.validate_commentary_request(request.data)
    except json.JSONDecodeError:
        return flask.jsonify({"error": "JSON payload is malformed"}), 400
    except jsonschema.ValidationError:
        return flask.jsonify({"error": "JSON payload does not respect schema specification"}), 400
    except ValueError as e:
        return flask.jsonify({"error": str(e)}), 400

    return app.response_class(instance.get_commentary_probabilities(request.data), mimetype='application/octet-stream')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
