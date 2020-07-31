from typing import Tuple, List, Dict, Callable
from pandas import DataFrame

Query = Tuple[str, str]
Location = Tuple[float, float, float]  # start, end, length
Instance = Tuple[Query, Location]
Rating = List[float]
Prediction = Tuple[Query, List[Location], Rating]
Result = Tuple[Query, List[Location], Rating, dict]


def ActivityNetCap2Instances(raw_data: dict) -> List[Instance]:
    instances: List[Instance] = []
    for video_id, anno in raw_data.items():
        for sentence, timestamp in zip(anno["sentences"], anno["timestamps"]):
            query = (video_id, sentence)
            location = (*timestamp, anno["duration"])
            instance = (query, location)
            instances.append(instance)
    return instances


def CharadesSTA2Instances(raw_data: DataFrame) -> List[Instance]:
    instances = []
    for _, row in raw_data.iterrows():
        query = (row["id"], row["description"])
        location = tuple(row[["start (sec)", "end (sec)", "length"]].tolist())
        instance = (query, location)
        instances.append(instance)
    return instances
