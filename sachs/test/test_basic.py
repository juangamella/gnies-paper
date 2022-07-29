# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Tests for the research.utils module
"""

import unittest

# ---------------------------------------------------------------------
# Tests


class BasicTests(unittest.TestCase):
    """Test that importing and loading the data works"""
    
    def test_wang_2017_load(self):
        import sachs.wang_2017 as sachs_wang
        import sachs.wang_2017.main
        data = sachs_wang.load_data(normalize=True)
        nobs = [len(s) for s in data]
        self.assertEqual(nobs, sachs.wang_2017.main._expected_observations)
        self.assertEqual(data[0].shape[1], 11)
        data = sachs_wang.load_data(normalize=False)
        nobs = [len(s) for s in data]
        self.assertEqual(nobs, sachs.wang_2017.main._expected_observations)
        self.assertEqual(data[0].shape[1], 11)

    def test_wang_2017_directory(self):
        import sachs.wang_2017 as sachs_wang
        path = '/tmp'
        graph = 'consensus'
        sachs_wang.prepare_experiments_directory(path, graph, normalize=False)
        sachs_wang.prepare_experiments_directory(path, graph, normalize=True)

    def test_taeb_2022(self):
        import sachs.taeb_2022 as sachs_taeb
        data = sachs_taeb.load_data(normalize=True)
        self.assertEqual(data[0].shape[1], 11)
        self.assertEqual(len(data), 8)
        data = sachs_taeb.load_data(normalize=False)
        self.assertEqual(data[0].shape[1], 11)
        self.assertEqual(len(data), 8)

    def test_taeb_2022_directory(self):
        import sachs.taeb_2022 as sachs_taeb
        path = '/tmp'
        graph = 'consensus'
        sachs_taeb.prepare_experiments_directory(path, graph, normalize=False)
        sachs_taeb.prepare_experiments_directory(path, graph, normalize=True)
