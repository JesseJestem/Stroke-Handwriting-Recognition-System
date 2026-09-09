from typing import NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
StrokeArray: TypeAlias = NDArray[np.float32]

class StrokePoint(TypedDict):
    x: float
    y: float
    t: float
    pressure: NotRequired[float]
    pen_down: NotRequired[bool]


class StrokeData(TypedDict):
    strokes: list[StrokePoint]
    label: NotRequired[str]
    canvas_width: NotRequired[int]
    canvas_height: NotRequired[int]