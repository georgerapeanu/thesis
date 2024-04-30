import flask
from flask import Flask, request
import json
import jsonschema
from serve_utils.ServeModelUtilsFacadeSingleton import ServeModelUtilsFacadeSingleton
import logging

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('artifacts/model.log')
handler = logging.StreamHandler()
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
))
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


@app.post('/get_commentary_execution')
def get_commentary_execution():
    instance = ServeModelUtilsFacadeSingleton()
    try:
        logger.info(f"Commentary execution: Request request {request.data}")
        instance.validate_commentary_request(request.data)
    except json.JSONDecodeError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": "JSON payload is malformed"}), 400
    except jsonschema.ValidationError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": "JSON payload does not respect schema specification"}), 400
    except ValueError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": str(e)}), 400
    except Exception as e:
        return flask.Response("Unknown error has occurred", status=500)
    return app.response_class(instance.get_commentary_probabilities(request.data), mimetype='application/octet-stream')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
