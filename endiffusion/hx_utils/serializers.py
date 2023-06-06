"""
LMDB Embeddings - Fast word vectors with little memory usage in Python.
dom.hudson@thoughtriver.com
Copyright (C) 2018 ThoughtRiver Limited
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pickle
import pickletools

import msgpack


class PickleSerializer:
    @staticmethod
    def serialize(vector):
        return pickletools.optimize(
            pickle.dumps(vector, pickle.HIGHEST_PROTOCOL)
        )

    @staticmethod
    def unserialize(serialized_vector):
        return pickle.loads(serialized_vector)


class MsgpackSerializer:
    def __init__(self, raw=False):
        self._raw = raw

    @staticmethod
    def serialize(vector):
        return msgpack.packb(vector)

    def unserialize(self, serialized_vector):
        return msgpack.unpackb(serialized_vector, raw=self._raw)
