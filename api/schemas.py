from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    service: str
    model_name: str
    model_family: str
    feature_count: int
    uptime_status: str

class PredictTripRequest(BaseModel):
    trip_id: str = Field(..., description="The unique trip identifier (e.g. '00000000')")
    sigma_z: float = Field(4.07, gt=0, description="GPS measurement noise standard deviation in meters")
    beta: float = Field(3.0, gt=0, description="HMM transition network distance variance parameter")
    alpha_ml: float = Field(0.70, ge=0, le=1, description="Weight blending ML classifier score with spatial likelihood [0, 1]")

class PredictRawRequest(BaseModel):
    track: List[List[float]] = Field(..., description="List of [lon, lat, timestamp]")
    nodes: List[List[float]] = Field(..., description="List of [lon, lat] for 0-indexed nodes")
    arcs: List[List[int]] = Field(..., description="List of [u, v] directed node pairs for 0-indexed arcs")
    sigma_z: float = Field(4.07, gt=0, description="GPS measurement noise std (meters)")
    beta: float = Field(3.0, gt=0, description="HMM transition parameter")
    alpha_ml: float = Field(0.70, ge=0, le=1, description="ML weight parameter")

class ReconstructedRouteResponse(BaseModel):
    trip_id: Optional[str]
    status: str
    num_gps_observations: int
    num_reconstructed_arcs: int
    reconstructed_route: List[int] = Field(..., description="Ordered list of traversed road arc IDs")
    route_continuity_pct: float = Field(..., description="Percentage of continuous network transitions")
    arc_set_f1_pct: Optional[float] = Field(None, description="Arc-set F1 overlap; ignores order and repeated traversals")
    sequence_precision_pct: Optional[float] = Field(None, description="Arc-set precision vs ground-truth route")
    sequence_recall_pct: Optional[float] = Field(None, description="Arc-set recall vs ground-truth route")
    has_ground_truth: bool = Field(False, description="Whether reference ground-truth route was available for scoring")
    disconnected_transition_count: int = 0
    route_segments: List[List[int]] = Field(default_factory=list, description="Connected route sections; never draw across section boundaries")
    processing_time_sec: float
    map_url: Optional[str] = Field(None, description="URL of the dynamically generated interactive Folium map")
    route_coordinates: Optional[List[List[float]]] = Field(None, description="Latitude/longitude points of the reconstructed route")
    message: str
