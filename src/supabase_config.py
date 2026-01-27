import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()


def get_supabase_client() -> Client:
    """
    Initialize and return a Supabase client using environment variables.

    Required .env variables:
    - SUPABASE_URL
    - SUPABASE_KEY (service role or anon, depending on your security model)
    """
    url: Optional[str] = os.getenv("SUPABASE_URL")
    key: Optional[str] = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment")

    return create_client(url, key)


def upload_to_supabase(
    client: Client,
    file_path: str,
    bucket: str = "documents",
    destination_path: Optional[str] = None,
) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    object_path = destination_path or file_name

    with open(file_path, "rb") as f:
        # التعديل هنا: نرفع الملف بدون تمرير خيارات الـ Headers التي تسبب المشكلة
        client.storage.from_(bucket).upload(
            path=object_path,
            file=f,
            file_options={"cacheControl": "3600", "upsert": "true"} # لاحظ استخدام "true" كنص
        )

    public_url = client.storage.from_(bucket).get_public_url(object_path)
    return public_url