app tested with python3.10.16

A. Ollama setup
  1. download ollama from [here](https://ollama.com/download) and install it on your machine
  2. open command prompt
  3. ollama run llama2:13b #this will download and run the model on your machine)

B. rag-pipeline
  1. open a new command prompt (make sure you have python3.10.16)
  2. Clone the git repo to your machine
  3. cd rag-pipeline
  4. pip install -r requirements.txt #requirements.txt might have some irrelevant dependencies which are not required
  5. uvicorn main:app --reload
  6. wait for the console to print "INFO:     Application startup complete."
  7. go to http://localhost:8000/docs from your browser
  8. click on /Ask Ask Question > Try it out > replace string in example value | schema with the query > Hit execute under it
  9. You should get the answer based on faqs.txt in Response body under Responses
