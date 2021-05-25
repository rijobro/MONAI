# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import Callable, List, Optional, Tuple

from parameterized import parameterized

from monai.data import create_test_image_2d
from monai.transforms import AddChanneld, Compose, LoadImaged, RandAdjustContrastd, RandAxisFlipd, RandFlipd
from monai.transforms.inverse import InvertibleTransform
from monai.utils import set_determinism
from monai.utils.enums import InverseKeys
from tests.utils import make_nifti_image

KEYS = ["image", "label"]

TESTS: List[Tuple] = []

# Remove any applied "RandAdjustContrastd" (only "label")
TESTS.append(
    (
        Compose(
            [LoadImaged(KEYS), AddChanneld(KEYS), RandAxisFlipd("image"), RandAdjustContrastd("label"), RandFlipd(KEYS)]
        ),
        (0, 1),
        None,
        lambda x: x[InverseKeys.CLASS_NAME] != "RandAdjustContrastd",
    )
)

# Remove first in list for both
TESTS.append(
    (
        Compose(
            [LoadImaged(KEYS), AddChanneld(KEYS), RandAxisFlipd("image"), RandAdjustContrastd("label"), RandFlipd(KEYS)]
        ),
        (1, 1),
        lambda x: x[1:],
        None,
    )
)

# per-element won't remove "RandAdjustContrastd" as it was the first Invertible applied to "label", not a change of 2
TESTS.append(
    (
        Compose(
            [LoadImaged(KEYS), AddChanneld(KEYS), RandAxisFlipd("image"), RandAdjustContrastd("label"), RandFlipd(KEYS)]
        ),
        (1, 1),
        lambda x: x[1:],
        lambda x: x[InverseKeys.CLASS_NAME] != "RandAdjustContrastd",
    )
)

# remove first transform from both "image" and "label", and then remove any "RandAxisFlipd" ("image")
TESTS.append(
    (
        Compose(
            [LoadImaged(KEYS), AddChanneld(KEYS), RandAxisFlipd("image"), RandAdjustContrastd("label"), RandFlipd(KEYS)]
        ),
        (2, 1),
        lambda x: x[:-1],
        lambda x: x[InverseKeys.CLASS_NAME] != "RandAxisFlipd",
    )
)

# remove all RandAxisFlipd (2 for "image", 1 for "label")
TESTS.append(
    (
        Compose([LoadImaged(KEYS), AddChanneld(KEYS), RandAxisFlipd("image"), RandAxisFlipd(KEYS)]),
        (2, 1),
        None,
        lambda x: x[InverseKeys.CLASS_NAME] != "RandAxisFlipd",
    )
)


class TestInverseSubset(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)

        im_fnames = [make_nifti_image(i) for i in create_test_image_2d(101, 100)]
        self.data = {k: v for k, v in zip(KEYS, im_fnames)}

    def tearDown(self):
        set_determinism(seed=None)

    def test_error(self):
        with self.assertRaises(ValueError):
            Compose.remove_applied_transforms(self.data, KEYS)

    @parameterized.expand(TESTS)
    def test_inverse_subset(
        self,
        transforms: InvertibleTransform,
        expected_num_removed: Tuple[int],
        list_selection_fn: Optional[Callable] = None,
        per_element_select_fn: Optional[Callable] = None,
    ) -> None:
        d = transforms(self.data)
        # print(d)
        d2 = transforms.remove_applied_transforms(
            d, KEYS, list_selection_fn=list_selection_fn, per_element_selection_fn=per_element_select_fn
        )

        for k, n in zip(KEYS, expected_num_removed):
            self.assertEqual(len(d[k + InverseKeys.KEY_SUFFIX]), len(d2[k + InverseKeys.KEY_SUFFIX]) + n)


if __name__ == "__main__":
    unittest.main()
