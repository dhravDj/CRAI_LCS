from typing import Dict, Any, Optional

# Verified Sensitivity Reward Profiles from Nature Scientific Reports (Table 10-11)
REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "RW-1": {
        "alpha": 0.50,
        "beta": 0.20,
        "gamma": 0.20,
        "eta": 0.10,
        "description": "Latency-Focused Profile: Heavily penalizes transmission and queue delay."
    },
    "RW-2": {
        "alpha": 0.35,
        "beta": 0.25,
        "gamma": 0.30,
        "eta": 0.10,
        "description": "Balanced Profile (Default): Equalized latency, clinical risk, and miss avoidance."
    },
    "RW-3": {
        "alpha": 0.25,
        "beta": 0.20,
        "gamma": 0.45,
        "eta": 0.10,
        "description": "Clinical-Risk-Focused Profile: Maximally incentivizes on-time emergency tasks."
    },
    "RW-4": {
        "alpha": 0.25,
        "beta": 0.20,
        "gamma": 0.25,
        "eta": 0.30,
        "description": "Resource-Utilization-Focused Profile: Prioritizes edge core load balancing."
    }
}


def compute_reward(
    latency_ms: float,
    deadline_ms: float,
    predicted_miss_prob: float,
    clinical_risk: float,
    mean_edge_util: float,
    profile: str = "RW-2",
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    eta: Optional[float] = None
) -> float:
    """
    Computes Normalized Multi-Objective Reward Function (Equation 29, FR-014, ADR-002).

    R_t = -[ alpha * (L_i / D_i) + beta * P_{miss,i} - gamma * (R_i * 1_{L_i <= D_i}) ] + eta * U_edge

    Parameters:
    - latency_ms: Total end-to-end task turnaround time (communication + queue + execution).
    - deadline_ms: Task deadline budget in milliseconds.
    - predicted_miss_prob: Pre-execution deadline violation probability in [0, 1].
    - clinical_risk: Patient arrhythmia urgency index in [0, 1].
    - mean_edge_util: Average processor utilization across 25 edge gateways in [0, 1].
    - profile: One of 'RW-1', 'RW-2', 'RW-3', 'RW-4'.
    - alpha, beta, gamma, eta: Explicit overriding weights.
    """
    # Select weights
    cfg = REWARD_PROFILES.get(profile, REWARD_PROFILES["RW-2"])
    w_alpha = float(alpha if alpha is not None else cfg["alpha"])
    w_beta = float(beta if beta is not None else cfg["beta"])
    w_gamma = float(gamma if gamma is not None else cfg["gamma"])
    w_eta = float(eta if eta is not None else cfg["eta"])

    # Normalized Latency: L_i / D_i (~1.0 scale prevents RSK-007 gradient explosion)
    safe_deadline = max(float(deadline_ms), 1e-3)
    norm_latency = float(latency_ms) / safe_deadline

    # Indicator function 1_{L_i <= D_i}
    is_on_time = 1.0 if float(latency_ms) <= float(deadline_ms) else 0.0

    # Risk bonus is only granted if the task actually fulfilled its clinical deadline
    risk_term = float(clinical_risk) * is_on_time

    # Multi-objective synthesis
    reward = -(
        (w_alpha * norm_latency)
        + (w_beta * float(predicted_miss_prob))
        - (w_gamma * risk_term)
    ) + (w_eta * float(mean_edge_util))

    return float(reward)
