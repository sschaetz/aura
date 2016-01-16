'''Alang web interface.'''
from flask import Flask
from flask import render_template

APP = Flask(__name__)

@APP.route('/')
def index():
    '''Render index page.'''
    return render_template('index.html')

if __name__ == '__main__':
    APP.run()
