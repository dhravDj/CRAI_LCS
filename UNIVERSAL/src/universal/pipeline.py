"""
U-CRAI-LCS: End-to-End Universal Multi-Disease Clinical Continuum Pipeline
Integrates Ingestion -> Gating -> Multi-Branch Embedding -> Triage Calibration -> Two-Tier Scheduling -> Execution.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from src.universal.schema import ModalityType, UniversalTelemetryTask
from src.universal.adapters import UniversalTelemetryIngestor
from src.universal.asymmetric_gate import AsymmetricSensorGate, GatingDecision
from src.universal.models import UniversalBiosignalEncoder, PluggableDiseaseRiskHead
from src.universal.triage import ClinicalUrgencyCalibrator, DynamicSLAPredictor
from src.universal.two_tier_scheduler import TwoTierUniversalScheduler
from src.universal.simulator import UniversalHealthcareContinuumEnv
from src.universal.audit_logger import ClinicalAuditLogger


class UniversalHealthcarePipeline:
    """
    Unified Orchestrator for Multi-Disease Healthcare IoT Continuum.
    """
    def __init__(self, seed: int = 42):
        self.ingestor = UniversalTelemetryIngestor()
        self.sensor_gate = AsymmetricSensorGate()
        self.encoder = UniversalBiosignalEncoder(embedding_dim=128)
        self.risk_head = PluggableDiseaseRiskHead(embedding_dim=128)
        self.encoder.eval()
        self.risk_head.eval()

        self.calibrator = ClinicalUrgencyCalibrator()
        self.sla_predictor = DynamicSLAPredictor()
        self.scheduler = TwoTierUniversalScheduler(num_edge_nodes=25)
        self.env = UniversalHealthcareContinuumEnv(num_edge_nodes=25, seed=seed)
        self.audit_logger = ClinicalAuditLogger()

    def process_raw_telemetry(
        self,
        task_id: int,
        modality: ModalityType,
        patient_id: int,
        raw_signal: np.ndarray,
        arrival_time: float,
        channel_snr_db: float = 20.0,
        has_pacemaker: bool = False
    ) -> Tuple[UniversalTelemetryTask, GatingDecision]:
        # 1. Condition & Ingest
        task = self.ingestor.ingest(task_id, modality, patient_id, raw_signal, arrival_time)

        # 2. Asymmetric Sensor-Side Gating with SQI, Duty Cycle & Pacemaker guard
        gate_decision = self.sensor_gate.screen_window(
            signal=task.raw_signal,
            modality=modality,
            patient_id=patient_id,
            has_pacemaker=has_pacemaker
        )

        # 3. AI Risk Inference
        sig = task.raw_signal
        if modality == ModalityType.NEURO_EEG:
            # Shape (1, 16, 1280)
            tensor_x = torch.from_numpy(sig).unsqueeze(0).float()
        else:
            # Shape (1, 1, Samples)
            tensor_x = torch.from_numpy(sig).view(1, 1, -1).float()

        with torch.no_grad():
            z = self.encoder(tensor_x, modality)
            r = float(self.risk_head(z).item())

        task.clinical_risk = r

        # 4. Clinical Time-to-Harm Calibration
        u = self.calibrator.calibrate(r, modality)
        task.clinical_urgency = u

        # 5. Dynamic SLA Deadline & P_miss
        deadline_ms, p_miss = self.sla_predictor.compute_deadline_and_miss_risk(
            urgency=u,
            payload_bytes=task.payload_bytes,
            channel_snr_db=channel_snr_db
        )
        task.deadline_ms = deadline_ms
        task.predicted_miss_prob = p_miss

        return task, gate_decision

    def run_multi_disease_streaming_benchmark(
        self,
        tasks: List[UniversalTelemetryTask]
    ) -> Dict[str, Any]:
        """Executes a streaming benchmark of mixed clinical tasks across the continuum."""
        self.env.set_workload(tasks)
        obs, info = self.env.reset()

        fail_safe_count = 0
        latencies = []

        for task in tasks:
            action, is_fail_safe, target_desc = self.scheduler.select_action(
                task, obs, current_time=self.env.current_time
            )
            if is_fail_safe:
                fail_safe_count += 1

            next_obs, reward, term, trunc, info = self.env.step(action)
            obs = next_obs
            latencies.append(task.actual_latency_ms)
            self.audit_logger.log_task_event(task, action, is_fail_safe, target_desc)

        info["fail_safe_count"] = fail_safe_count
        info["p95_latency_ms"] = float(np.percentile(latencies, 95))
        info["audit_critical_events"] = len(self.audit_logger.critical_events_ring)
        return info
