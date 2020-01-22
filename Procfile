#heroku ps:scale web=1
web: uvicorn eye_disease_server:app --host=0.0.0.0 --port=${PORT:-5000}
