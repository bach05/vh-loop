"""Sample registry helpers.

Defines a Pydantic discriminated union that can be used to parse
heterogeneous sample records. Add additional sample types to the
Union as they are implemented.
"""

from .single_image import SISimpleDataSample

from typing import Union, Annotated
from pydantic import Field

SampleUnion = Annotated[
    Union[
        SISimpleDataSample,
        # add more sample types here
    ],
    Field(discriminator="sample_type"),
]
