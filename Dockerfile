FROM python:3.9

# Upgrade pip
RUN pip install -U pip

ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Create a new directory for app (keep it in its own directory)
ADD . /app
WORKDIR /app

## App engine stuff
# Expose port you want your app on
EXPOSE 8080

# Run
#CMD ["gunicorn", "-b", ":8080", "app:app", "--timeout"]
CMD gunicorn --bind :8080 main:app --timeout 0
