from supabase import Client, create_client


from app.core.config import settings


def create_headers():
    return {
        "apiKey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
    }


class SupabaseClient:
    def __init__(self):
        self.client: Client = create_client(
            supabase_url=settings.SUPABASE_URL, supabase_key=settings.SUPABASE_KEY
        )

    def get_data(self, table_name):
        return self.client.from_(table_name).select("*").execute()

    def insert_data(self, table_name, data):
        return self.client.from_(table_name).insert(data).execute()

    def update_data(self, table_name, data, id):
        return self.client.from_(table_name).update(data).eq("id", id).execute()


class SupabaseStorage:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )

    def list_buckets(self):
        return self.client.storage().list_buckets()

    def list_files(self, bucket_name):
        return self.client.storage().from_(bucket_name).list()

    def get_bucket(self, bucket_name):
        return self.client.storage().get_bucket(bucket_name)

    def create_bucket(self, bucket_name):
        return self.client.storage().create_bucket(bucket_name)

    def delete_bucket(self, bucket_name):
        return self.client.storage().delete_bucket(bucket_name)

    def upload(self, bucket_name, file_name, file_path):
        with open(file_path, "rb+") as file:
            return (
                self.client.storage()
                .from_(bucket_name)
                .upload(path=file_name, file=file_path)
            )

    def download(self, bucket_name, file_name, file_path):
        with open(file_path, "wb+") as file:
            response = self.client.storage().from_(bucket_name).download(path=file_name)
            file.write(response)
