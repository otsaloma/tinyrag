#!/usr/bin/env python3

import click
import dataiter as di
import hashlib
import numpy as np
import openai

from jinja2 import Template
from pathlib import Path

chat_model = "gpt-4o"
embedding_model = "text-embedding-3-small"
oai = openai.OpenAI()

def get_db_path(name):
    return Path(__file__).with_name(f"{name}.npz")

def read_db(name):
    path = get_db_path(name)
    print(f"Reading {path.name}...")
    return di.read_npz(path)

def read_or_create_db(name):
    if get_db_path(name).exists():
        return read_db(name)
    print("Creating new database...")
    return di.DataFrame()

def write_db(db, name):
    path = get_db_path(name)
    print(f"Writing {db.nrow} rows to {path.name}...")
    db.write_npz(path)

def load_chunks(path):
    if path.suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            print(f"... Page {i+1:3d}/{len(reader.pages)}: ", end="")
            yield page.extract_text().strip()

def populate_db(path):
    blob = path.read_bytes()
    id = hashlib.sha1(blob).hexdigest()[:8]
    db = read_or_create_db("chunks")
    nprior = db.nrow
    print(f"Adding {path.name!r}...")
    for i, text in enumerate(load_chunks(path)):
        if db and db.filter(document_id=id, chunk=i+1).nrow:
            print("Already in database")
            continue
        response = oai.embeddings.create(input=text, model=embedding_model)
        print(f"{response.usage.total_tokens:4d} tokens")
        embedding = np.array(response.data[0].embedding)
        new = di.DataFrame(document_id=id,
                           document_name=path.name,
                           chunk=i+1,
                           text=text,
                           embedding=None)

        new.text = new.text.as_object()
        new.embedding[0] = embedding
        db = db.rbind(new)
    nadded = db.nrow - nprior
    print(f"{nadded} rows added to database.")
    if nadded: write_db(db, "chunks")

def cosine_similarity(a, b):
    # https://stackoverflow.com/a/43043160
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def title_print(title, text):
    print(f"\n{'='*25} {title.upper()} {'='*25}\n\n{text}")

def query_db(query, ntop=5):
    db = read_db("chunks")
    assert di.count_unique(db.embedding.map(len)) == 1, "db broke?"
    response = oai.embeddings.create(input=query, model=embedding_model)
    query_embedding = np.array(response.data[0].embedding)
    db.similarity = db.embedding.map(lambda x: cosine_similarity(x, query_embedding))
    db = db.sort(similarity=-1).head(ntop)
    prompt = Template("""
Use the following sources to answer the subsequent question.
Do not use any other information.
Do not make any assumptions.
List the sources you used below your answer in format "{DOCUMENT-NAME}:{CHUNK}".
If an answer cannot be found in the sources provided, reply "idk" and nothing else.
{% for c in chunks %}
<SOURCE DOCUMENT-NAME="{{c.document_name}}" CHUNK="{{c.chunk|int}}" CONFIDENCE="{{c.similarity|round(3)}}">
{{c.text}}
</SOURCE>
{% endfor %}
Question: {{query}}
""".strip())
    chunk_dicts = db.unselect("embedding").to_list_of_dicts()
    prompt = prompt.render(chunks=chunk_dicts, query=query)
    title_print("prompt", prompt)
    response = oai.chat.completions.create(model=chat_model, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ])
    message = response.choices[0].message.content
    title_print("response", message)

@click.command(no_args_is_help=True)
@click.option("-p", "--populate", is_flag=True, help="Populate database; ARGS: PDF-files")
@click.option("-q", "--query", is_flag=True, help="Query database; ARGS: Query string")
@click.argument("args", nargs=-1)
def main(populate, query, args):
    if populate:
        for path in map(Path, args):
            populate_db(path)
    elif query and args:
        query_db(args[0])

if __name__ == "__main__":
    main()
