## Problem
When surveys are sent out to people to collect feedback in the form of text, it is difficult to parse through this text to obtain fast and actionable insights.

## Solution
Provide a Juptyer Notebook and corresponding documentation that will walk someone through the process of uploading their survey to a Juptyer Notebook and Python code to provide a high-level summary of the textual data so it can be easily understood. But Juptyer Notebooks, while convenient for developers, do not help non-developers as it is not user friendly.

To make this even simpler for non-technologists, this project should also consider an extremely simple front-end to allow for CSV upload and the display of summary information using basic front-end technologies while using the same Python code for the backend.


## Running the app
### Without Docker
```
pip install -r requirements.txt
python -m spacy download en
python run_app.py
```

### Docker
```
docker build -t succinct-python .
docker run -d -p 4040:4040 -v /var/uploads/:/tmp/succint_upload/ succinct-python
```


## Running the Huey Consumer
To process tasks that are long running, we use Huey
```
python ~/opt/miniconda3/bin/huey_consumer.py --verbose app.tasks.get_similar.huey
```

# References
These sources were helpful to get everything up and running
- https://github.com/mbr/flask-bootstrap/tree/master/sample_app
- http://flask.pocoo.org/docs/1.0/quickstart/#rendering-templates
- https://pythonhosted.org/Flask-Uploads/#flaskext.uploads.UploadConfiguration
