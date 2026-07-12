# AutoDefect Radar

AutoDefect Radar is a local retrieval-augmented generation (RAG) CLI for
exploring vehicle recalls and consumer complaints published by the National
Highway Traffic Safety Administration (NHTSA).

The application fetches public NHTSA records for a vehicle, redacts VINs from
complaint text, builds a local Chroma vector index, and answers questions using
a local Ollama model. Answers use retrieved records as evidence and include
NHTSA campaign or ODI complaint identifiers.

AutoDefect Radar is an independent open-source project. It is not affiliated
with NHTSA, vehicle manufacturers, or dealerships, and it does not provide
safety, legal, repair, or diagnostic advice.

## Architecture

The CLI runs a local pipeline:

1. Fetch recall and complaint records from the public NHTSA API.
2. Normalize records into VIN-redacted Markdown documents.
3. Split documents and embed them with `all-MiniLM-L6-v2`.
4. Persist the embeddings in a local Chroma index.
5. Retrieve similar chunks and send them to a local Ollama model.

Raw responses, normalized documents, and the vector index are generated under
`data/` and deliberately excluded from Git.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

Install the default model before asking questions:

```bash
ollama pull gemma3:270m
```

## Setup

```bash
uv sync --dev
```

## Build the index

The default corpus is the 2012 Honda Accord, including all recalls and the 100
most recently filed complaints.

```bash
uv run autodefect-radar ingest
```

Choose another vehicle or complaint limit with CLI options:

```bash
uv run autodefect-radar ingest \
  --make TOYOTA \
  --model CAMRY \
  --year 2020 \
  --complaints-limit 200
```

Use `--refresh` to bypass the local raw-response cache. Pass
`--complaints-limit -1` to index all returned complaints.

## Ask questions

```bash
uv run autodefect-radar ask "What airbag issues appear?"
uv run autodefect-radar ask "Are there ABS module complaints?" --k 6
```

The `ask` command requires an index created by `ingest` and a running Ollama
service.

By default, the answer prompt is relaxed so the model can synthesize from
retrieved evidence without over-refusing. Set `STRICT_ANSWER_PROMPT = True` in
`src/config.py` to restore strict context-only answering.

## Run the Gradio UI

After building the index, launch the browser UI:

```bash
uv run autodefect-radar-ui
```

The UI calls the same answer path as the `ask` command and shows the request
payload, answer, retrieved sources, and retrieved context.

## Data and privacy

NHTSA complaint records can contain vehicle identification numbers and
free-form consumer narratives. AutoDefect Radar redacts VIN-shaped values when
creating the normalized knowledge base. The original API cache is not modified,
so the complete `data/raw_cache/` directory remains local and ignored by Git.

The following generated directories are never committed:

- `data/raw_cache/`
- `data/knowledge-base/`
- `data/vector_db/`

Delete `data/` and run `ingest` again to rebuild all local artifacts.

## Tests

```bash
uv run pytest
```

The test suite uses temporary files, mocked HTTP responses, and deterministic
fake embeddings. It does not require network access, Ollama, or downloaded
embedding models.

## Limitations

- Retrieval uses vector similarity followed by a basic term-overlap guard. A
  weakly related question can still pass the guard when it shares one important
  term with retrieved text.
- Citation fallback reports retrieved evidence identifiers; it does not prove
  that every retrieved record contributed to the generated wording.
- NHTSA complaints are consumer reports and are not independently verified by
  this project.
- Generated indexes are local to the embedding model configured in the source.

## License

Licensed under the [MIT License](LICENSE).
