FROM python:3.9

## App engine stuff
# Expose port you want your app on
EXPOSE 8080

# Upgrade pip
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Create a new directory for app (keep it in its own directory)
COPY . /app
WORKDIR app

# Run
#CMD ["gunicorn", "-b", ":8080", "app:app", "--timeout", "300"]
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app --timeout 0
