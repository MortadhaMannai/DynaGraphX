import gc
import os
import pathlib
import pickle
import subprocess
from datetime import datetime
from typing import Any, Dict, Union

import msgpack
import msgpack_numpy as m
from scipy.sparse import csr_matrix

m.patch()  # monkey patch msgpack for numpy support


def file_exists(file: Union[str, os.PathLike]) -> bool:
    return os.path.exists(file)


def gz_file_exists(file: Union[str, os.PathLike]) -> bool:
    return os.path.exists(file + '.gz')


def check_dir_exists(dir: Union[str, os.PathLike]):
    if not os.path.exists(dir):
        os.makedirs(dir)


def pickle_save(file: Union[str, os.PathLike], data: Any):
    with open(file, 'wb') as output:
        pickle.dump(data, output)


def pickle_load(file: Union[str, os.PathLike]) -> Any:
    with open(file, 'rb') as f:
        return pickle.load(f)


def msgpack_save(file: Union[str, os.PathLike], data: Any):
    # Write msgpack file
    with open(file, 'wb') as output:
        packed = msgpack.packb(data, default=encode_this)  # use_bin_type=True by default
        output.write(packed)
        output.close()
    # Compress msgpack with gzip keeping existing file
    with open(file + '.gz', 'wb') as output_compressed:
        subprocess.run(['gzip', '-c', '-f', file], stdout=output_compressed)
    # subprocess.run(['rm', file])


def msgpack_load(file: Union[str, os.PathLike]) -> Any:
    # If decompressed file doesn't exist, decompress first
    if not file_exists(file) and gz_file_exists(file):
        with open(file, 'wb') as msg_file:
            subprocess.run(['gunzip', '-c', file + '.gz'], stdout=msg_file)

    with open(file, 'rb') as f:
        byte_data = f.read()
        # CPython's GC starts when growing allocated object.
        # This means unpacking may cause useless GC.
        # You can use gc.disable() when unpacking large message.
        gc.disable()
        try:
            result = msgpack.unpackb(byte_data, object_hook=decode_this)  # raw=True by default
        except ValueError:
            result = msgpack.unpackb(byte_data, object_hook=decode_this, strict_map_key=False)  # raw=True by default
        # subprocess.run(['rm', file])
        gc.enable()
        return result


def encode_this(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, pathlib.PosixPath):
        return {'__path__': True, 'as_str': str(obj)}
    if isinstance(obj, frozenset):
        return {'__frozenset__': True, 'as_tuple': tuple(obj)}
    if isinstance(obj, set):
        return {'__set__': True, 'as_list': list(obj)}
    if isinstance(obj, csr_matrix):
        return {'__csr_matrix__': True, 'as_csr_tuple': (obj.data, obj.indices, obj.indptr)}
    if isinstance(obj, datetime):
        return {'__datetime__': True, 'as_datetime_str': obj.isoformat()}
    return obj


def decode_this(obj: Dict[str, Any]) -> Any:
    if '__path__' in obj:
        obj = str(obj['as_str'])  # just keep as string
    if '__frozenset__' in obj:
        obj = frozenset(obj['as_tuple'])
    if '__set__' in obj:
        obj = set(obj['as_list'])
    if '__csr_matrix__' in obj:
        assert len(obj['as_csr_tuple']) == 3
        obj = csr_matrix((obj['as_csr_tuple'][0],
                          obj['as_csr_tuple'][1],
                          obj['as_csr_tuple'][2]))
    if '__datetime__' in obj:
        print('__datetime__')
        if obj['as_datetime_str']:
            print(obj['as_datetime_str'])
            obj = datetime.fromisoformat(obj['as_datetime_str'])
        else:
            print('no as_datetime_str ?')
    return obj
