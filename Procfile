heroku ps:scale web=0
heroku ps:scale web=1
web: uvicorn app/server:app --host=0.0.0.0 --port=${PORT:-5000}
