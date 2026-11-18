import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app
from src.config import TEST_DIR, OUTPUT_DIR
from src.data_loader import load_trip
from src.hmm_decoder import ViterbiMapMatcher
from src.metrics import compute_classification_metrics, compute_observation_top1_accuracy
from src.visualizer import generate_interactive_folium_map


def test_trip_grouping_and_macro_f1():
    result = compute_observation_top1_accuracy(
        np.array([0, 0, 0, 0]), np.arange(4), np.array([1, 0, 0, 1]),
        np.array([.9, .1, .2, .8]), trip_ids=np.array(['a', 'a', 'b', 'b']))
    assert result['total_obs'] == 2 and result['top1_accuracy'] == 1
    metrics = compute_classification_metrics(np.array([0, 0, 0, 1]), np.array([.1, .1, .1, .1]))
    assert metrics['f1'] == 0
    assert metrics['macro_f1'] == pytest.approx(3 / 7)


def test_reference_not_read_for_blind_loader():
    with patch('src.data_loader.parse_route', side_effect=AssertionError('Labels read')):
        assert not len(load_trip(TEST_DIR, '00000000', include_ground_truth=False)['route'])


def test_map_omits_reference_geometry(tmp_path):
    trip = load_trip(TEST_DIR, '00000000')
    trip['route'] = np.array([len(trip['arcs']) + 999])  # Would fail if accessed for drawing.
    path = tmp_path / 'blind.html'
    generate_interactive_folium_map(trip, pred_hmm_arcs=[0], save_path=path)
    html = path.read_text(encoding='utf-8')
    assert 'Reference Ground Truth' not in html
    assert 'Reference Route (Comparison)' not in html


def test_disconnected_and_empty_candidates_rejected():
    matcher = ViterbiMapMatcher(np.array([[0, 1], [2, 3]]), np.array([0., 1., 10., 11.]), np.zeros(4))
    kwargs = dict(candidate_arcs_per_obs=[np.array([0]), np.array([1])],
                  candidate_probs_per_obs=[np.ones(1), np.ones(1)],
                  candidate_dists_per_obs=[np.zeros(1), np.zeros(1)],
                  candidate_proj_t_per_obs=[np.zeros(1), np.zeros(1)],
                  gps_timestamps=np.array([0., 1.]), gps_x=np.array([0., 10.]), gps_y=np.zeros(2))
    _, route = matcher.decode_trajectory(**kwargs)
    assert route == [0, 1]
    assert matcher.disconnected_transitions == [(0, 1)]
    kwargs['candidate_arcs_per_obs'][1] = np.array([], dtype=int)
    with pytest.raises(ValueError, match='No candidates'):
        matcher.decode_trajectory(**kwargs)


def test_reserved_list_and_validation():
    with TestClient(app) as client:
        trips = client.get('/trips').json()
        assert sum(t['split'] == 'evaluator_test' for t in trips) == 10
        assert client.post('/predict/trip/00000015?sigma_z=0').status_code == 422
        assert client.post('/predict/trip/00000015?alpha_ml=2').status_code == 422
        assert client.post('/predict/trip/bogus').status_code == 404
        with patch('api.main.get_loaded_model', return_value=({'model': None}, None)):
            assert client.post('/predict/raw', json=dict(track=[], nodes=[], arcs=[])).status_code == 422
            assert client.post('/predict/raw', json=dict(track=[[0, 0, 0]], nodes=[[0, 0]], arcs=[[0, 2]])).status_code == 422


def test_old_analysis_maps_not_public():
    with TestClient(app) as client:
        assert client.get('/eda/folium_map_test_00000000.html').status_code == 404


def test_live_evaluator_has_no_reference_or_scores():
    with TestClient(app) as client:
        with patch('api.main.parse_route', side_effect=AssertionError('Evaluator labels read')):
            response = client.post('/predict/trip/00000023')
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['has_ground_truth'] is False and data['arc_set_f1_pct'] is None
        html = client.get(data['map_url']).text
        assert 'Reference Ground Truth' not in html and 'Reference Route (Comparison)' not in html


def test_student_network_gaps_are_explicit():
    with TestClient(app) as client:
        response = client.post('/predict/trip/00000000')
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['disconnected_transition_count'] > 0
        assert data['status'] == 'partial'
        assert data['route_coordinates'] is None
        assert len(data['route_segments']) == data['disconnected_transition_count'] + 1
        assert data['route_continuity_pct'] < 100
