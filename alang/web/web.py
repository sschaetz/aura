'''Alang web interface.'''
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

APP = Flask(__name__)

@APP.route('/')
def index():
    '''Render index page.'''
    return render_template('index.html')

@APP.route('/compile', methods=['POST'])
def compile():
    '''Compile the source code.'''
    print("Compile method called!")
    if request.method == 'POST':
        print(request.form['code'])
        return jsonify({'input': request.form['code']})

if __name__ == '__main__':
    APP.debug = True
    APP.run()
