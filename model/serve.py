import flask
from flask import Flask, request
import json
import jsonschema

import torch

from data.ActualBoardCommentaryDataset import ActualBoardCommentaryDataset
from model.commentary_models import ActualBoardTransformerMultipleHeadsModel
from serve_utils.ServeUtilsFacadeSingleton import ServeUtilsFacadeSingleton

app = Flask(__name__)



@app.get('/annotate')
def process():
    instance = ServeUtilsFacadeSingleton()
    try:
        instance.validate_request(request.data)
    except json.JSONDecodeError:
        return flask.jsonify({"error": "JSON payload is malformed"}), 400
    except jsonschema.ValidationError:
        return flask.jsonify({"error": "JSON payload does not respect schema specification"}), 400
    except ValueError as e:
        return flask.jsonify({"error": str(e)}), 400

    return app.response_class(instance.get_commentary(request.data), mimetype='text')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
