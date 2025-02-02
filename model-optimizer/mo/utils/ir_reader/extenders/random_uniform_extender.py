# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.middle.passes.convert_data_type import destination_type_to_np_data_type

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class RandomUniformExtender(Extender):
    op = 'RandomUniform'

    @staticmethod
    def extend(op: Node):
        if op.has_valid('output_type'):
            op['output_type'] = destination_type_to_np_data_type(op.output_type)
