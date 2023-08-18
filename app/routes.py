import os
import numpy as np
from flask import render_template, jsonify, make_response
from flask import request as flask_request
from flask_restful import Resource, Api, reqparse

from app import app, api
from app.forms import SearchForm

from jinja2 import BaseLoader, TemplateNotFound, ChoiceLoader
from urllib import request, parse
from ElMD import ElMD
import pickle as pk
import pandas as pd

from logging.config import dictConfig

from crab.kingcrab import CrabNet
from crab.model import Model
from utils.get_compute_device import get_compute_device
from utils.utils import EDM_CsvLoader

def load_crabnet(path, classification=False):
    model = Model(CrabNet(compute_device=get_compute_device()).to(get_compute_device()),
                  model_name=path,
                  verbose=False,
                  classification=classification)

    model.load_network(f"app/trained_models/{path}")
    return model

parser = reqparse.RequestParser()
parser.add_argument('compositions', action="append")

crabnet_reg = [load_crabnet(f"TransferFinalModel_Reg.pth")]
crabnet_cls = [load_crabnet(f"TransferFinalModel_Clf.pth", classification=True)]

class UrlLoader(BaseLoader):
    def __init__(self, url_prefix):
        self.url_prefix = url_prefix

    def get_source(self, environment, template):
        url = parse.urljoin(self.url_prefix, template)
        try:
            t = request.urlopen(url)
            if t.getcode() is None or t.getcode() ==200:
                return t.read().decode('utf-8'), None, None
        except IOError:
            pass
        
        raise TemplateNotFound(template)

# Add this to ensure can access static files dynamically
app.jinja_loader = ChoiceLoader([app.jinja_loader, UrlLoader('https://lmds.liverpool.ac.uk/')])


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', composition="NaCl", prediction=5)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form = SearchForm()

    if form.validate_on_submit():
        comps = [comp.strip() for comp in str(form.search_term.data).split(",") if comp != ""]
        app.logger.debug(comps)

        queries = [ElMD(comp).pretty_formula for comp in str(form.search_term.data).split(",")]

        X = np.array([ElMD(query, metric="mat2vec").feature_vector for query in queries])

        df = pd.DataFrame({"formula": queries, "target": np.ones(len(queries))})
        
        if len(X) == 1:
            X.reshape(1, -1)
        
        for i in range(len(crabnet_reg)):
            crabnet_reg[i].load_data(df, train=False)
        
        for i in range(len(crabnet_cls)):
            crabnet_cls[i].load_data(df, train=False)

        crab_regs = [crabnet_reg[i].predict(crabnet_reg[i].data_loader, app)[1] for i in range(len(crabnet_reg))]

        crab_clfs = [crabnet_cls[i].predict(crabnet_cls[i].data_loader, app)[1] for i in range(len(crabnet_cls))]
        
        crab_reg_means = np.mean(crab_regs, axis=0)
        crab_clf_means = np.mean(crab_clfs, axis=0).astype(int)
        
        results = [(query, crab_clf_means[i], crab_reg_means[i]) for i, query in enumerate(comps)]
    
        return render_template("ionics_ml.html", form=form, results=results)

    return render_template("ionics_ml.html", form=form)


@app.route("/API_info", methods=['GET', 'POST'])
@app.route("/predict/API_info", methods=['GET', 'POST'])
def api_info():
    return render_template("api_info.html")

# Define the parser for the API

# Define the API class and associate it with flask
class CrabApiEndpoint(Resource):
    def get(self):
        return {"Hello": "world"}
    
    def put(self):
        
        args = parser.parse_args()
        if ("compositions" not in args):
            return make_response(jsonify({"Error":"Failed to process input, check both compositions are provided."}), 400)
        
        try:
            if isinstance(args["compositions"],str):
                comps = [args["compositions"].strip()]
            else:
                comps = [x.strip() for x in args["compositions"] if x.strip() != ""]

            queries = [ElMD(comp).pretty_formula for comp in comps]
            X = np.array([ElMD(query, metric="mat2vec").feature_vector for query in queries])

            df = pd.DataFrame({"formula": queries, "target": np.ones(len(queries))})
        
            if len(X) == 1:
                X.reshape(1, -1)
        
            for i in range(len(crabnet_reg)):
                crabnet_reg[i].load_data(df, train=False)
        
            for i in range(len(crabnet_cls)):
               crabnet_cls[i].load_data(df, train=False)


            crab_regs = [crabnet_reg[i].predict(crabnet_reg[i].data_loader)[1] for i in range(len(crabnet_reg))]
            crab_clfs = [crabnet_cls[i].predict(crabnet_cls[i].data_loader)[1] for i in range(len(crabnet_cls))]
    
            crab_reg_means = np.mean(crab_regs, axis=0)
            crab_clf_means = np.mean(crab_clfs, axis=0).astype(int)
        
            results = [(query.replace("\'", "").replace('\"', "").strip(), str(crab_clf_means[i]), str(crab_reg_means[i])) for i, query in enumerate(comps)]
    
            return jsonify({"Ionics ML Results": results})

        except Exception as e:
            app.logger.debug(e)
            return make_response(jsonify({"Error": "Failed to process result please check input"}), 500)

    def post(self):
        return self.put()

api.add_resource(CrabApiEndpoint, "/predict/API")
