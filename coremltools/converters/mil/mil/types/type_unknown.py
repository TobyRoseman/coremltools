# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from .type_spec import Type


class unknown:
    """
    unknown is basically Any type.
    """

    @classmethod
    def __type_info__(cls):
        return Type("unknown", python_class=cls)

    def __init__(self, val=None):
        self.val = val
