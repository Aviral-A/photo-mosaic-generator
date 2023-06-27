import os
import numpy
import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


class Database:
    def __init__(self):
        self.connection = psycopg2.connect(DATABASE_URL)
        self.create_pgvector_extension()
        self.create_images_table()

    def create_pgvector_extension(self):
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.connection.commit()

    def create_images_table(self):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id bigserial PRIMARY KEY,
                    filename text,
                    output_filename text,
                    embedding vector(128)
                );
                """
            )
            self.connection.commit()

    def add_image(self, filename, output_filename, embeddings):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO images (filename, output_filename, embedding)
                VALUES (%s, %s, %s);
                """,
                (filename, output_filename, embeddings.tolist()),
            )
            self.connection.commit()