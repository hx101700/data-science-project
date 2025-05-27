docker build -t datascience-app:latest .
docker save -o datascience-app.tar datascience-app:latest
# docker run -p 8501:8501 datascience-app:latest # run the container if needed