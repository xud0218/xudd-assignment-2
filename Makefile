# Install required dependencies
install:
	pip install -r requirements.txt

# Run the Flask application locally
run:
	FLASK_APP=app.py FLASK_ENV=development flask run --host=127.0.0.1 --port=3000

# Clean any cached files
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
