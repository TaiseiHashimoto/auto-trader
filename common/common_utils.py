import numpy as np
import pandas as pd
from typing import Optional
# from dataclasses import dataclass
from google.cloud import (storage, secretmanager)


# @dataclass
# class Order:
#     position_type: str
#     entry_timestamp: pd.Timestamp
#     exit_timestamp: pd.Timestamp
#     entry_rate: float
#     exit_rate: float

class Order():
    def __init__(
        self,
        position_type: str,
        entry_timestamp: pd.Timestamp,
        entry_rate: float,
    ):
        self.position_type = position_type
        self.entry_timestamp = entry_timestamp
        self.entry_rate = entry_rate

    def exit(
        self,
        exit_timestamp: pd.Timestamp,
        exit_rate: float,
    ):
        self.exit_timestamp = exit_timestamp
        self.exit_rate = exit_rate

    @property
    def gain(self):
        rate_diff = self.exit_rate - self.entry_rate
        if self.position_type == "long":
            return rate_diff
        elif self.position_type == "short":
            return -rate_diff


class OrderManager():
    def __init__(self):
        # self.prev_timestamp = None
        # self.position_type = None
        self.order_history = []
        self.open_position = None

    def entry(self, position_type: str, timestamp: pd.Timestamp, rate: float):
        assert not self.has_position()

        self.open_position = Order(position_type, timestamp, rate)

    def exit(self, position_type: str, timestamp: pd.Timestamp, rate: float):
        assert self.has_position(position_type)

        self.open_position.exit(timestamp, rate)
        self.order_history.append(self.open_position)
        self.open_position = None

    def has_position(self, position_type: Optional[str] = None) -> bool:
        if position_type is None:
            return self.open_position is not None
        else:
            return (
                self.open_position is not None and
                self.open_position.position_type == position_type
            )

    def entry_rate_of_open_position(self) -> float:
        assert self.has_position()
        return self.open_position.entry_rate


class GCSWrapper:
    def __init__(self, project_id: str, bucket_id: str):
        """GCS のラッパークラス
        Arguments:
            project_id -- GoogleCloudPlatform Project ID
            bucket_id -- GoogleCloudStorage Bucket ID
        """
        self._project_id = project_id
        self._bucket_id = bucket_id
        self._client = storage.Client(project_id)
        self._bucket = self._client.get_bucket(self._bucket_id)

    def list_bucket_names(self):
        """バケット名の一覧を表示
        """
        return [bucket.name for bucket in self._client.list_buckets()]

    def list_file_names(self):
        """バケット内のファイル一覧を表示
        """
        return [file.name for file in self._client.list_blobs(self._bucket)]

    def upload_file(self, local_path: str, gcs_path: str):
        """GCSにローカルファイルをアップロード

        Arguments:
            local_path -- local file path
            gcs_path -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def download_file(self, local_path: str, gcs_path: str):
        """GCSのファイルをファイルとしてダウンロード

        Arguments:
            local_path -- local file path
            gcs_path -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.download_to_filename(local_path)


class SecretManagerWrapper:
    def __init__(self, project_id: str):
        """Secret Manager のラッパークラス
        Arguments:
            project_id -- GoogleCloudPlatform Project ID
        """
        self._project_id = project_id
        self._client = secretmanager.SecretManagerServiceClient()

    def fetch_secret(self, secret_id: str, secret_version: Optional[str] = "latest"):
        """Secret 取得
        Arguments:
            secret_id -- GoogleSecretManager Secret ID
            secret_version -- GoogleSecretManager Secret Version
        """
        name = self._client.secret_version_path(self._project_id, secret_id, secret_version)
        response = self._client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF8")


def calc_year_month_offset(year: int, month: int, month_offset: int):
    """
    ある年月からnヶ月後/前の年月を求める
    """
    month = month + month_offset
    year = year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year, month


