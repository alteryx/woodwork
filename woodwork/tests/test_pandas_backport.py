"""
All code in this module was copied from pandas and is subject to the following
license:

BSD 3-Clause License

Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
All rights reserved.

Copyright (c) 2011-2021, Open source contributors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pytest

from woodwork.pandas_backport import guess_datetime_format


# TODO: Remove this test and also the
# `woodwork.pandas_backport.guess_datetime_format` backport when this fix is
# released in pandas 1.4:
# https://github.com/pandas-dev/pandas/pull/43900
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("20111230", "%Y%m%d"),
        ("2011-12-30", "%Y-%m-%d"),
        ("30-12-2011", "%d-%m-%Y"),
        ("2011-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ("2011-12-30T00:00:00", "%Y-%m-%dT%H:%M:%S"),
        ("2011-12-30T00:00:00UTC", "%Y-%m-%dT%H:%M:%S%Z"),
        ("2011-12-30T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+9", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+090", None),
        ("2011-12-30T00:00:00+0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00-0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:00", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:000", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+9:0", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:", None),
        ("2011-12-30T00:00:00.000000UTC", "%Y-%m-%dT%H:%M:%S.%f%Z"),
        ("2011-12-30T00:00:00.000000Z", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+9", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+090", None),
        ("2011-12-30T00:00:00.000000+0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000-0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:000", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+9:0", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:", None),
        ("2011-12-30 00:00:00.000000", "%Y-%m-%d %H:%M:%S.%f"),
        ("Tue 24 Aug 2021 01:30:48 AM", "%a %d %b %Y %H:%M:%S %p"),
        ("Tuesday 24 Aug 2021 01:30:48 AM", "%A %d %b %Y %H:%M:%S %p"),
    ],
)
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    result = guess_datetime_format(string)
    assert result == fmt
