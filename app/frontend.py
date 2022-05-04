# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

from flask import Blueprint, render_template, flash, redirect, url_for, request, current_app
from flask_uploads import (UploadSet, configure_uploads, DATA,
                              UploadNotAllowed)

import uuid

from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from markupsafe import escape

from os import listdir
from os.path import isfile, join, exists

from .forms import SignupForm
from .nav import nav
from .text_summarization import text_summarization
import pandas as pd
import csv, sys

csv.field_size_limit(sys.maxsize)

frontend = Blueprint('frontend', __name__)
uploaded_files = UploadSet('files', DATA)


def unique_id():
    return hex(uuid.uuid4().time)[2:-1]

def to_index():
    return redirect(url_for('.index'))

def to_listfiles():
    return redirect(url_for('.list_files'))

# We're adding a navbar as well through flask-navbar. In our example, the
# navbar has an usual amount of Link-Elements, more commonly you will have a
# lot more View instances.
nav.register_element('frontend_top', Navbar(
    View('Succinct', '.index'),
    View("Upload CSVs", ".new"),
    View("Find Insights", ".list_files"),

    #View('Debug-Info', 'debug.debug_root'),
    Subgroup(
        'Docs',
        Link('Flask-Bootstrap', 'http://pythonhosted.org/Flask-Bootstrap'),
        Link('Flask-AppConfig', 'https://github.com/mbr/flask-appconfig'),
        Link('Flask-Debug', 'https://github.com/mbr/flask-debug'),
        Separator(),
        Text('Bootstrap'),
        Link('Getting started', 'http://getbootstrap.com/getting-started/'),
        Link('CSS', 'http://getbootstrap.com/css/'),
        Link('Components', 'http://getbootstrap.com/components/'),
        Link('Javascript', 'http://getbootstrap.com/javascript/'),
        Link('Customize', 'http://getbootstrap.com/customize/'), ),
    Text('Using Flask-Bootstrap {}'.format(FLASK_BOOTSTRAP_VERSION)), ))


# Our index-page just shows a quick explanation. Check out the template
# "templates/index.html" documentation for more details.
@frontend.route('/')
def index():
    return render_template('index.html')

class Post():
    doc_type = 'post'
    title = ""
    filename = ""
    caption = ""
    published = ""

    @property
    def imgsrc(self):
        return uploaded_files.url(self.filename)

    def store(self):
        print("*** Storing!!")


@frontend.route('/new', methods=['GET', 'POST'])
def new():
    if request.method == 'POST':
        data = request.files.get('data')
        #title = request.form.get('title')
        #caption = request.form.get('caption')
        #if not (data and title and caption):
        #    flash("You must fill in all the fields")
        #else:
        #print("Title: %s | Caption: %s" % (title, caption))

        try:
            filename = uploaded_files.save(data)
        except UploadNotAllowed as una:
            print(una)
            flash("The upload was not allowed.  Make sure it's a .csv pretty please.")
        else:
            post = Post()
            post.id = unique_id()
            post.store()
            flash("Upload successful! ")
            return to_listfiles()

    return render_template('new.html')

def file_head(file, N_lines=3):
    pd.set_option('display.width',1000)
    pd.set_option('display.max_columns',500)
    return pd.read_csv(file, engine='python').head(N_lines)

    #from itertools import islice
    #with open(file) as file:
    #    head_arr = list(islice(file, N_lines))
    #head = "".join(head_arr)
    #return head

def get_column_names(file):
    df = pd.read_csv(file, engine='python')
    return df.columns

@frontend.route('/list-files/')
def list_files():
    file_dir = current_app.config["UPLOADED_FILES_DEST"]

    if exists(file_dir):
        onlyfiles = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

        return render_template('listfiles.html', files=sorted(onlyfiles))
    else:
        return render_template('error.html', error="No file has been uploaded yet.")


@frontend.route('/summarize/', methods=['POST'])
def summarize():
    file_dir = current_app.config["UPLOADED_FILES_DEST"]
    file_to_process = request.form.get('file_to_process')
    column_to_process = request.form.get('column_to_process')
    filter_text = request.form.get('filter_text')
    column_to_filter = request.form.get('column_to_filter')
    ngram_start = request.form.get('ngram_start')
    ngram_end = request.form.get('ngram_end')
    min_df = request.form.get('min_df')
    max_df = request.form.get('max_df')



    print("Processing file: %s" % file_to_process)

    full_path = join(file_dir, file_to_process)
    first_three_lines = file_head(full_path,5)
    column_names = get_column_names(full_path)

    print("Path: ", full_path)

    t = text_summarization()
    t.ngram_start = 2 if not ngram_start else int(ngram_start)
    t.ngram_end = 3 if not ngram_end else int(ngram_end)
    t.min_df = 0.0025 if not min_df else float(min_df)
    t.max_df = 1.0 if not max_df else float(max_df)

    if not column_to_process:
        return render_template('summarize.html',
                file_to_process=file_to_process,
                first_three_lines=first_three_lines,
                column_names=column_names,
                ngram_start=t.ngram_start,
                ngram_end=t.ngram_end,
                min_df=t.min_df,
                max_df=t.max_df)
    else:
        csv_data = pd.read_csv(full_path, engine='python')
        if filter_text != "" and column_to_filter!="":
            csv_data = csv_data.loc[csv_data[column_to_filter].astype(str) == str(filter_text)]

        number_of_rows = len(csv_data.index)

        stats = t.get_text_stats(column_to_process, csv_data)
        print(stats)

        return render_template('summarize.html',
                file_to_process=file_to_process,
                first_three_lines=first_three_lines,
                column_names=column_names,
                selected_column_name=column_to_process,
                column_to_filter=column_to_filter,
                filter_text=filter_text,
                num_rows = number_of_rows,
                num_ngrams=t.num_ngrams,
                sparsity=t.sparsity,
                common_terms=t.common_terms,
                tf_idf_results=t.tf_idf_results,
                term_examples=t.term_examples,
                tfidf_plot=t.tfidf_plot,
                term_plot=t.term_plot,
                ngram_start=t.ngram_start,
                ngram_end=t.ngram_end,
                min_df=t.min_df,
                max_df=t.max_df)


# Shows a long signup form, demonstrating form rendering.
@frontend.route('/example-form/', methods=('GET', 'POST'))
def example_form():
    form = SignupForm()

    if form.validate_on_submit():
        # We don't have anything fancy in our application, so we are just
        # flashing a message when a user completes the form successfully.
        #
        # Note that the default flashed messages rendering allows HTML, so
        # we need to escape things if we input user values:
        flash('Hello, {}. You have successfully signed up'
              .format(escape(form.name.data)))

        # In a real application, you may wish to avoid this tedious redirect.
        return redirect(url_for('.index'))

    return render_template('signup.html', form=form)
