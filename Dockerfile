FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD sh -c "until python manage.py migrate --noinput; do echo 'DB not ready, retrying in 5s...'; sleep 5; done && gunicorn ChaiScript.wsgi --bind 0.0.0.0:$PORT"