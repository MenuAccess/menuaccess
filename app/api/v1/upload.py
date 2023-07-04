from typing import List, Optional, Any

from pathlib import Path
import json

from fastapi import APIRouter, Depends, Query, File, UploadFile, BackgroundTasks

from app.services.index import LlamaIndex
from app.schemas.output import FormattedOutput

from app.core.config import settings

router = APIRouter()

index_name = "./index.json"
documents_folder = "./documents"

@router.post(
    "/upload/{user_id}",
    response_description="Upload a file",
    tags=["file"],
)
async def file_upload(
    user_id: str,
    background_tasks: BackgroundTasks, file: UploadFile = File(...), restrictions: Optional[List[str]] = Query(None)
):
    """Handle a file upload, the file should have the user's ID in the filename."""

    # Create Temporary Directory and File to store the file

    if file.filename:

        llama_index = LlamaIndex(user_id=user_id, file_name=file.filename)

        background_tasks.add_task(llama_index.upload_file, file)

        background_tasks.add_task(
            llama_index.query_index,
            user_id=user_id,
            file_name=file.filename,
            restrictions=''.join(restrictions))

    return {"message": "File uploaded successfully"}
