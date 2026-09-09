import json
from pathlib import Path
from typing import cast

import numpy as np

from handwriting.core.exceptions import StrokeDataError
from handwriting.core.types import IntArray, StrokeArray, StrokeData

#~~~~~~~~~~~~~~~~~~
#Load stroke JSON -> dict
#~~~~~~~~~~~~~~~~~~

def load_stroke_json(stroke_path: str | Path) -> StrokeData:
    stroke_path = Path(stroke_path)

    try:
        with open(stroke_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except FileNotFoundError as exc:
        raise StrokeDataError(
            f"Stroke file not found: {stroke_path}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise StrokeDataError(
            f"Invalid stroke JSON: {stroke_path}"
        ) from exc

    return cast(StrokeData, data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Normalized strokes in range 0-1 in formate [num_points, 6]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def normalize_strokes (data: StrokeData) -> StrokeArray:

    strokes = data.get("strokes", []) #if not - empty list
    
    if len(strokes) == 0:
        return np.zeros((0, 6), dtype=np.float32) #emtpy list - zeros
    
    #take stroke data and convert it into np.array
    x_values = np.array(
        [p["x"] for p in strokes],
        dtype=np.float32,
    )
    y_values = np.array(
        [p["y"] for p in strokes],
        dtype=np.float32,
    )
    t_values = np.array(
        [p["t"] for p in strokes],
        dtype=np.float32,
    )
    pressure_values = np.array(
        [p.get("pressure", 0.5) for p in strokes],
        dtype=np.float32,
    )
    pen_down_values = np.array(
        [1.0 if p.get("pen_down", True) else 0.0 for p in strokes],
        dtype=np.float32,
    )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Add aditional stroke start feature
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    stroke_start_values = np.zeros_like(pen_down_values)
    previous_pen_down = 0.0

    # Detect the beginning of each stroke
    # pen_down:    [1, 1, 1, 0, 1, 1]
    # stroke_start:[1, 0, 0, 0, 1, 0]
    for i, pen_down in enumerate(pen_down_values):
        if pen_down == 1.0 and previous_pen_down == 0.0:
            stroke_start_values[i] = 1.0
        previous_pen_down = pen_down

    #~~~~~~~~~~~~~~~~~~~~
    #Normalize coordinate
    #~~~~~~~~~~~~~~~~~~~~

    #take borders of letter
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    #take width, height, scale - proportion
    width = x_max - x_min
    height = y_max - y_min
    scale = max(width, height)

    #if scale too low - set zeros to not divide by 0
    if scale < 1e-6:
        x_norm = np.zeros_like(x_values)
        y_norm = np.zeros_like(y_values)

    #transform coordinate to 0-1 range
    else:
        x_norm = (x_values - x_min) / scale
        y_norm = (y_values - y_min) / scale 

        #set letter on center
        if width < scale:
            x_norm += (1.0 - width / scale) / 2.0

        if height < scale:
            y_norm += (1.0 - height / scale) / 2.0

    #cut over or under
    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)

    #~~~~~~~~~~~~~~
    #Normalize time
    #~~~~~~~~~~~~~~

    t_min, t_max = t_values.min(), t_values.max()

    #transform time to 0-1 range
    if t_max - t_min < 1e-6:
        t_norm = np.zeros_like(t_values)
    
    else:
        t_norm = (t_values - t_min) / (t_max - t_min)

    #~~~~~~~~~~~~~~
    #Normalize pressure
    #~~~~~~~~~~~~~~

    pressure_values = np.clip(pressure_values, 0.0, 1.0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Collect all normalize features to np.array and return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    features = np.stack(
        [
            x_norm,
            y_norm,
            t_norm,
            pressure_values,
            pen_down_values,
            stroke_start_values,
        ],
        axis=1
    )

    return features.astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Split sequence separate strokes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_strokes(strokes: StrokeArray) -> list[StrokeArray]:

    segments = []
    current_segment = []

    #add point to segment and merge segments if not divided
    #result = segments = [stroke_1,stroke_2,stroke_3]
    for point in strokes:
        pen_down = point[4] >=0.5

        if pen_down:
            current_segment.append(point)
        else:
            if len(current_segment) > 0:
                segments.append(np.array(current_segment,dtype=np.float32))
                current_segment = []
    
    if len(current_segment) > 0:
        segments.append(np.array(current_segment, dtype=np.float32))

    return segments

#~~~~~~~~~~~~~~~~~~~~~~~~
#Stroke length
#~~~~~~~~~~~~~~~~~~~~~~~~

#check lenght to set propotrion if every stroke in range 0~100
def stroke_length(stroke: StrokeArray) -> float:

    #if only one point = 1
    if len(stroke) < 2:
        return 1.0
    
    #calculate diffetent between points
    dx = np.diff(stroke[:, 0])
    dy = np.diff(stroke[:, 1])

    #calculate distance between points
    distances = np.sqrt(dx ** 2 + dy ** 2)
    #calculate length of every stroke
    length = float(np.sum(distances))

    return max(length, 1e-6)

#~~~~~~~~~~~~~~~~~~~
#Resample one stroke
#~~~~~~~~~~~~~~~~~~~

def resample_single_stroke(
        stroke: StrokeArray,
        target_points: int,
) -> StrokeArray:

    #--------------------
    #if 0 target -> zeros
    if target_points <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    
    #--------------------
    #if stroke empty -> zeros
    if len(stroke) == 0:
        return np.zeros((target_points, 6), dtype=np.float32)
    
    #--------------------
    #if one point -> repeat target_points times
    if len(stroke) == 1:
        repeated = np.repeat(stroke, target_points, axis=0)

        repeated[:, 4] = 1.0 #pen_down
        repeated[:, 5] = 0.0 #stroke_start
        repeated[0, 5] = 1.0 #first point = start of stroke

        return repeated.astype(np.float32)
    
    #--------------------
    #calculate distance along the stroke for resampling
    dx = np.diff(stroke[:, 0])
    dy = np.diff(stroke[:, 1])

    distances = np.sqrt(dx ** 2 + dy ** 2)
    #add line based on sum of all distance from 0.0
    #[0.2, 0.5, 0.3] -> [0.0, 0.2, 0.7, 1.0]
    old_positions = np.concatenate([[0.0], np.cumsum(distances)])

    total_length = old_positions[-1]

    #--------------------
    #if too small = one point -> repeat one point
    if total_length < 1e-6:
        repeated = np.repeat(stroke[:1], target_points, axis=0)

        repeated[:, 4] = 1.0 #pen_down
        repeated[:, 5] = 0.0 #stroke_start
        repeated[0, 5] = 1.0 #first point = start of stroke

        return repeated.astype(np.float32)

    #--------------------
    #remove duplicate distance position - one point
    #same logic as only one point in stroke
    unique_positions, unique_indicates = np.unique(old_positions, return_index=True)

    if len(unique_indicates) < 2:
        repeated = np.repeat(stroke, target_points, axis=0)

        repeated[:, 4] = 1.0 #pen_down
        repeated[:, 5] = 0.0 #stroke_start
        repeated[0, 5] = 1.0 #first point = start of stroke

        return repeated.astype(np.float32)

    #~~~~~~~~~~~~~~~~~~~~
    #Main Single Stroke Resample
    #~~~~~~~~~~~~~~~~~~~~

    #take clear stroke
    unique_stroke = stroke[unique_indicates]

    #create new smooth line based on unique pofitions
    #[0.0, 1.0] target_points = 5 -> [0.0, 0.25, 0.5, 0.75, 1.0]
    new_positions = np.linspace(0.0, unique_positions[-1], num=target_points)
    #create empty result array
    resampled = np.zeros((target_points, 6), dtype=np.float32)

    #interpolate only x, y, t, pressure
    #np.interp(x, xp, fp)

    #x - where recive new position [0.0, 0.25, 0.5, 0.75, 1.0]
    #xp - based on old positions where known points [0.0, 0.4, 1.0]
    #fp - known points from ond positions [:, 0] = [10, 20, 50]
    #resampled - [10.0, 16.25, 25.0, 37.5, 50.0]
    for feature_index in range(4):
        resampled[:, feature_index] = np.interp(
            new_positions,
            unique_positions,
            unique_stroke[:, feature_index],
        )
    
    #inside one stroke pen is down
    resampled[:, 4] = 1.0

    #only first point is stroke_start
    resampled[:, 5] = 0.0
    resampled[0, 5] = 1.0

    return resampled.astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Distribute points between strokes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def distribute_points_between_strokes(
        segments: list[StrokeArray],
        max_points: int,
        separator_points_per_gap: int = 2,
) -> tuple[IntArray, int]:
    
    #set all points(100) between each stroke with -2 points for gap
    #100 opints, 3 stroke, 2 gaps, separator points = 2 gaps*2 =4
    #result points = 100-4 = 96

    num_segments = len(segments)

    #if no segments, empty list + 0
    if num_segments == 0:
        return np.array([], dtype=np.int32), 0
    
    #if gaps = -1, take 0
    gaps = max(num_segments -1, 0)
    total_sep_points = gaps * separator_points_per_gap

    drawing_points = max_points - total_sep_points

    #if sepatarors take too much plase -> disable separators, take all points to stroke
    if num_segments > drawing_points:
        points_per_segment = np.zeros(num_segments, dtype=np.int32)
        points_per_segment[:drawing_points] = 1
        return points_per_segment, separator_points_per_gap
    
    #calculate lenght of every stroke
    lengths = np.array(
        [stroke_length(segment) for segment in segments],
        dtype=np.float32,
    )

    #give atleast 1 point to stroke
    points_per_segment = np.ones(num_segments, dtype=np.int32)

    remaining_points = drawing_points - num_segments

    #give remaining_points to all stroke by proportion
    if remaining_points > 0:
        #how many % of point to every stroke
        weights = lengths / lengths.sum()

        raw_extra_points = weights * remaining_points
        extra_points = np.floor(raw_extra_points).astype(np.int32)

        points_per_segment += extra_points

        #floor can lose points, so give missing points to lagest fraction
        missing_points = drawing_points - points_per_segment.sum()

        fraction_parts = raw_extra_points - extra_points
        order = np.argsort(fraction_parts)[::-1]

        for i in order[:missing_points]:
            points_per_segment[i] += 1
        
    return points_per_segment, separator_points_per_gap

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Resample all strokes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resample_strokes(
        strokes: StrokeArray,
        max_points: int = 100,
) -> StrokeArray:

    #-------------------------
    #if no points return zeros (100, 6)
    if len(strokes) == 0:
        return np.zeros((max_points, 6), dtype=np.float32)
    
    segments = split_strokes(strokes)

    #---------------------
    #if aster segmentation = 0, return zeros
    if len(segments) == 0:
        return np.zeros((max_points, 6), dtype=np.float32)
    
    #---------------------
    #give points for every segment
    points_per_segment, separator_points_per_gap = distribute_points_between_strokes(
        segments=segments,
        max_points=max_points,
        separator_points_per_gap=2,
    )

    #------------------------
    #temp storage of segments + aading resampled_segments to result
    result_parts = []

    for segment_index, segment in enumerate(segments):
        target_points = int(points_per_segment[segment_index])
        if target_points <= 0:
            continue

        #resample for one stroke
        resampled_segment = resample_single_stroke(
            stroke=segment,
            target_points=target_points,
        )

        result_parts.append(resampled_segment)

        #----------------------
        #checking if need to add separator points between strokes
        #(but not last one and we have separators point)
        if segment_index < len(segments) - 1 and separator_points_per_gap > 0:
            current_end = resampled_segment[-1].copy()
            next_start = segments[segment_index + 1][0].copy() #take first point of last segment

            #---------------------
            #pen is up in the end of current segment, create separete point
            current_end[4] = 0.0 #pen_down
            current_end[5] = 0.0 #stroke_start

            #-------------------
            #move to start of next stroke while pen is up
            next_start[4] = 0.0 #pen_down
            next_start[5] = 0.0 #stroke_start

            #append + reshape to merge same shapes arrays
            result_parts.append(current_end.reshape(1, 6))
            result_parts.append(next_start.reshape(1, 6))
    
    #if nothing - add 0
    if len(result_parts) == 0:
        return np.zeros((max_points, 6), dtype=np.float32)
    
    #----------------------
    #merge all parts in result in order
    result = np.concatenate(result_parts, axis=0)

    #--------------------
    #if too many points - cut
    if len(result) > max_points:
        result = result[:max_points]

    #-----------------------
    #if too few points - add padding with 0
    if len(result) < max_points:
        padding = np.zeros((max_points - len(result), 6), dtype=np.float32)
        result = np.concatenate([result, padding], axis=0)
    
    return result.astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main pipeline: json -> load data -> norm -> split by strokes -> resample -> complete
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def preprocess_strokes(
        stroke_path: str | Path,
        max_points: int = 100,
) -> StrokeArray:

    data = load_stroke_json(stroke_path)
    normalized = normalize_strokes(data)
    resampled = resample_strokes(normalized, max_points=max_points)

    return resampled
