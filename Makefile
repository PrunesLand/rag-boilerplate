PY := .venv/bin/python

.PHONY: setup ingest chat ui eval clean

setup:            ## Create venv, install deps, pull models, seed config
	./setup.sh

ingest:           ## Crawl sources.txt and (re)build the index
	$(PY) -m ingest.build_index

chat:             ## Chat in the terminal
	$(PY) cli.py

ui:               ## Launch the Streamlit web UI
	.venv/bin/streamlit run app.py

eval:             ## Score the pipeline with RAGAS over eval/testset.jsonl
	$(PY) -m eval.run_ragas

clean:            ## Delete the built stores (keeps models + venv)
	rm -rf chroma_db docstore
