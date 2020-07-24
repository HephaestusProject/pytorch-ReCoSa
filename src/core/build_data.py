# Copyright (c) HephaestusProject
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import hashlib
import os
import shutil
import time
from os import stat
from typing import Optional

import requests
import tqdm
import yaml


class Config:
    """
    yaml parser
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def parse(cpath: str) -> dict:
        with open(cpath) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.
    Reference: https://github.com/facebookresearch/ParlAI/blob/ea366da91c93f2cec8fe29b6d93b37ed96ed45bd/parlai/core/build_data.py
    """

    def __init__(self, url: str, file_name: str, hashcode: str) -> None:
        self.url = url
        self.file_name = file_name
        self.hashcode = hashcode

    def download_file(self, dpath: str) -> None:
        download(self.url, dpath, self.file_name)
        self.checksum(dpath)

    def checksum(self, dpath: str) -> None:
        """
        Checksum on a given file.
        :param dpath: path to the downloaded file.
        """
        sha256_hash = hashlib.sha256()
        with open(os.path.join(dpath, self.file_name), "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self.hashcode:
                # remove_dir(dpath)
                raise AssertionError(
                    f"[ Checksum for {self.file_name} from \n{self.url}\n"
                    "does not match the expected checksum. Please try again. ]"
                )
            else:
                print("[ Checksum Successful ]")


def download(url, path: str, fname: str, redownload: str = False) -> None:
    """
    Download file using `requests`.
    If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``True``).
    """
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload
    print("[ downloading: " + url + " to " + outfile + " ]")
    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit="B", unit_scale=True, desc="Downloading {}".format(fname))

    while download and retry >= 0:
        resume_file = outfile + ".part"
        resume = os.path.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = "ab"
        else:
            resume_pos = 0
            mode = "wb"
        response = None

        with requests.Session() as session:
            try:
                header = (
                    {"Range": "bytes=%d-" % resume_pos, "Accept-Encoding": "identity"}
                    if resume
                    else {}
                )
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get("Accept-Ranges", "none") == "none":
                    resume_pos = 0
                    mode = "wb"

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get("Content-Length", -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos

                with open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except requests.exceptions.ConnectionError:
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print("Connection error, retrying. (%d retries left)" % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print("Retried too many times, stopped retrying.")
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning("Connection broken too many times. Stopped retrying.")

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning(
                "Received less data than specified in "
                + "Content-Length header for "
                + url
                + "."
                + " There may be a download problem."
            )
        move(resume_file, outfile)

    pbar.close()


def move(path1: str, path2: str) -> None:
    """Rename the given file."""
    shutil.move(path1, path2)


def built(path: str, version_string: str = Optional[None]) -> bool:
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version
    is regarded as not built.
    """
    if version_string:
        fname = os.path.join(path, ".built")
        if not os.path.isfile(fname):
            return False
        else:
            with open(fname, "r") as read:
                text = read.read().split("\n")
            return len(text) > 1 and text[1] == version_string
    else:
        return os.path.isfile(os.path.join(path, ".built"))


def make_dir(path: str) -> None:
    """Make the directory and any nonexistent parent directories (`mkdir -p`)."""
    # the current working directory is a fine path
    if path != "":
        os.makedirs(path, exist_ok=True)


def mark_done(path: str, version_string: str = Optional[None]) -> None:
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.
    :param str path:
        The file path to mark as built.
    :param str version_string:
        The version of this dataset.
    """
    with open(os.path.join(path, ".built"), "w") as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write("\n" + version_string)


def untar(path: str, fname: str, deleteTar: bool = True) -> None:
    """
    Unpack the given archive file to the same directory.
    :param str path:
        The folder containing the archive. Will contain the contents.
    :param str fname:
        The filename of the archive file.
    :param bool deleteTar:
        If true, the archive will be deleted after extraction.
    """
    print("unpacking " + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)


if __name__ == "__main__":
    pass
