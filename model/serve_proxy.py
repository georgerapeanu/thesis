import flask
from flask import Flask, request
import json
import jsonschema
from flask_cors import CORS, cross_origin
from serve_utils.ServeProxyUtilsFacadeSingleton import ServeProxyUtilsFacadeSingleton
import logging

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('artifacts/proxy.log')
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


@app.post('/get_commentary')
@cross_origin()
def get_commentary():
    instance = ServeProxyUtilsFacadeSingleton()
    try:
        logger.info(f"Get commentary: Received request {request.data} ")
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

    return app.response_class(instance.get_commentary(request.data), mimetype='plain/text')


@app.post('/topk')
@cross_origin()
def get_topk():
    instance = ServeProxyUtilsFacadeSingleton()
    try:
        logger.info(f"Get TopK: Received request {request.data} ")
        topk = instance.get_topk(request.data)
        return flask.jsonify(topk)
    except json.JSONDecodeError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": "JSON payload is malformed"}), 400
    except jsonschema.ValidationError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": "JSON payload does not respect schema specification"}), 400
    except ValueError as e:
        logger.warning("Error in payload: " + str(e))
        return flask.jsonify({"error": str(e)}), 200
    except Exception as e:
        return flask.Response("Unknown error has occurred", status=500)


@app.get("/info")
@cross_origin()
def get_info():
    logger.info(f"Get Info")
    instance = ServeProxyUtilsFacadeSingleton()
    return flask.jsonify(instance.get_info()), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
