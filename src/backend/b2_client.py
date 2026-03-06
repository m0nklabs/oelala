import os
import io
import boto3
import logging
from botocore.client import Config
from typing import Union, BinaryIO, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class B2Client:
    def __init__(self):
        self.endpoint = os.environ.get("B2_ENDPOINT")
        self.key_id = os.environ.get("B2_KEY_ID")
        self.app_key = os.environ.get("B2_APPLICATION_KEY")
        self.bucket_name = os.environ.get("B2_BUCKET_NAME", "oelala-media-eu")
        
        if not self.endpoint or not self.key_id or not self.app_key:
            self._client = None
        else:
            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                config=Config(signature_version='s3v4')
            )
            
    def is_configured(self) -> bool:
        return self._client is not None

    def put(self, bucket: str, key: str, data: Union[bytes, BinaryIO], content_type: str = "application/octet-stream") -> bool:
        if not self.is_configured():
            return False
            
        b2_key = f"{bucket}/{key}"
        kwargs = {}
        if content_type:
            kwargs['ContentType'] = content_type
            
        try:
            if isinstance(data, bytes):
                self._client.put_object(Bucket=self.bucket_name, Key=b2_key, Body=data, **kwargs)
            else:
                self._client.upload_fileobj(data, self.bucket_name, b2_key, ExtraArgs=kwargs)
            return True
        except Exception as e:
            logger.error(f"B2 upload error: {e}")
            return False

    def get(self, bucket: str, key: str) -> Optional[bytes]:
        if not self.is_configured():
            return None
            
        b2_key = f"{bucket}/{key}"
        try:
            response = self._client.get_object(Bucket=self.bucket_name, Key=b2_key)
            return response['Body'].read()
        except Exception:
            return None

    def get_with_metadata(self, bucket: str, key: str) -> Optional[Tuple[bytes, str, int]]:
        if not self.is_configured():
            return None
            
        b2_key = f"{bucket}/{key}"
        try:
            response = self._client.get_object(Bucket=self.bucket_name, Key=b2_key)
            content = response['Body'].read()
            return content, response.get('ContentType', 'application/octet-stream'), len(content)
        except Exception:
            return None

    def delete(self, bucket: str, key: str) -> bool:
        if not self.is_configured():
            return False
        b2_key = f"{bucket}/{key}"
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=b2_key)
            return True
        except Exception:
            return False

    def get_presigned_url(self, bucket: str, key: str, expires_in: int = 3600) -> Optional[str]:
        if not self.is_configured():
            return None
            
        b2_key = f"{bucket}/{key}"
        try:
            url = self._client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.bucket_name, 'Key': b2_key},
                ExpiresIn=expires_in
            )
            return url
        except Exception:
            return None

b2_client = B2Client()
