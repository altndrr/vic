# Retrieval

The container provides utilities to execute the retrieval pipeline. It includes two primary scripts for downloading the faiss indices from Google Drive and starting the server. Since `clip-retrieval` depends on multiple libraries that are incompatible with the rest of the codebase, the code has been separated from the repository.

Note that the `clip-retrieval back` command does not support batching, so we have created a script that extends the original one, which we use instead.

To start the server, run the following command:

```bash
# build the Docker images
docker compose build

# start the server
docker compose up --profile retrieval-server
```
