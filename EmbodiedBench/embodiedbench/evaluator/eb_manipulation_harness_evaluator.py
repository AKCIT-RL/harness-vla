"""Evaluator for the Harness VLA beta on EB-Manipulation.

Mirrors :class:`EB_ManipulationEvaluator` but drives the environment with the
Harness architecture: the :class:`HarnessPlanner` emits ONE primitive invocation
per turn, the :class:`PrimitiveLibrary` compiles it into a short burst of discrete
actions, and the loop closes by re-perceiving object coordinates and the
end-effector pose before the next turn. A per-episode JSONL audit trace records
every primitive, its compiled actions, and the environment feedback.

Beta scope (see ``docs/HARNESS_VLA_BETA_REPORT.md``):
* language-only perception (object coordinate table as text, no images);
* optional explicitly selected Task Specific Memory structural prior;
* ``vla_act`` defaults to a mock scripted contact primitive; the optional
    OpenVLA HTTP backend is a beta-only frozen alternative, not paper reproduction.
"""

import os
import copy
import hashlib
import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv, ValidEvalSets
from embodiedbench.envs.eb_manipulation.eb_man_utils import (
    form_harness_grounding_artifact_for_input,
)
from embodiedbench.envs.eb_manipulation.rgbd_grounding import (
    compute_oracle_metrics,
    summarize_oracle_frames,
)
from harness_vla.memory.global_memory import (
    GlobalMemory,
    GlobalMemoryLedger,
)
from harness_vla.planner.harness_planner import HarnessPlanner
from harness_vla.memory.task_memory import (
    load_task_memory,
    resolve_task_memory_commands,
)
from harness_vla.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    load_complete_jsonl,
    resolve_git_commit,
    summarize_trace_records,
    write_json_atomic,
)
from harness_vla.evaluation_guards import (
    NoProgressGuard,
    validate_vla_semantics,
)
from harness_vla.primitives.primitives import (
    PoseState,
    PrimitiveError,
    PrimitiveLibrary,
    _physical_object_name,
    classify_grasp_outcome,
    classify_spatial_postcondition,
    pose_from_observation,
    primitive_termination,
    reconcile_held_object,
    summarize_physical_state,
)
from harness_vla.backends.openvla_backend import (
    OpenVLABackendError,
    OpenVLAHTTPBackend,
    OpenVLAObservation,
    convert_libero_delta_to_eb,
)
from harness_vla.backends.pirlinf_backend import (
    PiRLinfBackendError,
    PiRLinfObservation,
    PiRLinfWebsocketBackend,
)
from harness_vla.planner.phase_policy import (
    Phase,
    PhaseOperation,
    build_phase_manifest,
    validate_phase_manifest,
)
from harness_vla.backends.vla_runtime import VLARuntime
from embodiedbench.main import logger


class EB_ManipulationHarnessEvaluator:
    """Harness VLA beta evaluator (one primitive per planner turn)."""

    def __init__(self, config):
        self.model_name = config['model_name']
        self.config = config
        self.eval_set = ValidEvalSets[0]
        self.env = None
        self.planner = None
        self.task_memory_path = config.get('task_memory_path', '') or ''
        self.task_memory_commands = ()
        self._task_memory_payload = None
        self.task_memory_audit = {
            'path': self.task_memory_path,
            'hash': None,
            'decision': 'not_configured',
            'stage': 'selection',
        }
        if self.task_memory_path:
            try:
                audit, self.task_memory_commands = load_task_memory(
                    self.task_memory_path
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.task_memory_audit.update({
                    'decision': 'rejected',
                    'stage': 'load',
                    'reason': str(error),
                })
            else:
                self._task_memory_payload = {
                    'audit': audit,
                    'commands': list(self.task_memory_commands),
                }
                self.task_memory_audit.update({
                    'hash': audit['source']['commands_sha256'],
                    'decision': 'used',
                    'stage': 'loaded',
                })
        self.task_memory_episode_audit = dict(self.task_memory_audit)
        self.library = PrimitiveLibrary(
            approach_dz=config.get('approach_dz', 8),
            lift_dz=config.get('lift_dz', 6),
        )
        # Turns (primitive invocations), not env steps, cap per episode.
        self.max_turns = config.get('max_turns', 12)
        # vla_act consumes several env steps, so allow more than the default 15.
        self.max_env_steps = config.get('max_env_steps', 30)
        self.configured_max_env_steps = self.max_env_steps
        self.phase_manifest = None
        self.phase_policy = None
        self.episode_protocol_seeds = ()
        self._protocol_memory_before = None
        self._protocol_memory_after = None
        self.global_memory_ledger_path = (
            config.get('global_memory_ledger_path', '') or ''
        )
        self.global_memory_include_seed = config.get(
            'global_memory_include_seed', True
        )
        if not isinstance(self.global_memory_include_seed, bool):
            raise ValueError('global_memory_include_seed must be boolean')
        self.global_memory_audit = {
            'path': self.global_memory_ledger_path,
            'decision': (
                'not_configured' if not self.global_memory_ledger_path else 'configured'
            ),
        }
        self._configure_phase_protocol()
        if self.global_memory_ledger_path and self.phase_policy is None:
            raise ValueError(
                'global_memory_ledger_path requires bootstrap or deployment phase'
            )
        self.grasp_thresholds = {
            'object_lift_threshold': config.get('grasp_object_lift_threshold', 3.0),
            'max_gripper_object_distance': config.get('grasp_max_distance', 8.0),
            'empty_object_motion_threshold': config.get('grasp_empty_motion_threshold', 1.0),
            'min_gripper_lift': config.get('grasp_min_gripper_lift', 3.0),
            'max_comotion_residual': config.get('grasp_max_comotion_residual', 2.0),
        }
        # Beta-only tolerances in the environment's 100^3 voxel grid.
        self.move_to_tolerance = float(config.get('move_to_tolerance', 2.0))
        self.place_tolerance = float(config.get('place_tolerance', 12.0))
        self.vla_backend_name = config.get('vla_backend', 'mock')
        self.openvla_backend = None
        self.pirlinf_backend = None
        self.openvla_max_chunks = int(config.get('openvla_max_chunks', 8))
        self.pirlinf_max_chunks = int(config.get('pirlinf_max_chunks', 8))
        self.pirlinf_replan_steps = int(config.get('pirlinf_replan_steps', 5))
        self.openvla_conversion = {
            'max_delta_xyz': config.get('openvla_max_delta_xyz', 0.05),
            'max_delta_rotation': config.get('openvla_max_delta_rotation', 0.5),
            'workspace_bounds': config.get(
                'openvla_workspace_bounds', [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
            ),
            'gripper_convention': config.get(
                'openvla_gripper_convention', 'libero_minus_open_plus_close'
            ),
            'rotation_frame': config.get('openvla_rotation_frame', 'local'),
        }
        if self.vla_backend_name == 'openvla_http':
            self.openvla_backend = OpenVLAHTTPBackend(
                config.get('openvla_url', ''),
                timeout=config.get('openvla_timeout', 120.0),
                **self.openvla_conversion,
                expected_unnorm_key=config.get('openvla_unnorm_key', 'libero_object'),
            )
        elif self.vla_backend_name == 'pirlinf_websocket':
            self.pirlinf_backend = PiRLinfWebsocketBackend(
                config.get('pirlinf_host', '127.0.0.1'),
                int(config.get('pirlinf_port', 8010)),
                replan_steps=self.pirlinf_replan_steps,
                timeout=config.get('pirlinf_timeout', 120.0),
            )
        elif self.vla_backend_name != 'mock':
            raise ValueError(f"Unsupported vla_backend: {self.vla_backend_name!r}")

    @staticmethod
    def _structured_sha256(value):
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _configure_phase_protocol(self):
        phase = self.config.get('protocol_phase', '') or ''
        if not phase:
            self.protocol_phase = 'unspecified'
            return
        self.phase_manifest = build_phase_manifest(
            bootstrap_seed=self.config['bootstrap_seed'],
            evaluation_seeds=self.config['evaluation_seeds'],
            bootstrap_budget=self.config['bootstrap_budget'],
            deployment_budget=self.config['deployment_budget'],
        )
        validate_phase_manifest(self.phase_manifest)
        self.phase_policy = self.phase_manifest.policy_for(phase)
        self.protocol_phase = self.phase_policy.phase.value

        selected_indexes = self.config.get('selected_indexes')
        protocol_seeds = self.config.get('episode_protocol_seeds')
        if not isinstance(selected_indexes, (list, tuple)) or not selected_indexes:
            raise ValueError('opt-in phase protocol requires explicit selected_indexes')
        if not isinstance(protocol_seeds, (list, tuple)):
            raise ValueError('opt-in phase protocol requires episode_protocol_seeds')
        if len(protocol_seeds) != len(selected_indexes):
            raise ValueError(
                'episode_protocol_seeds must align 1:1 with selected_indexes'
            )
        for seed in protocol_seeds:
            self.phase_manifest.guard_operation(
                self.phase_policy.phase, seed, PhaseOperation.READ_MEMORY
            )
        self.episode_protocol_seeds = tuple(protocol_seeds)
        # Local budget unit: benchmark environment steps per episode.
        self.max_env_steps = min(self.configured_max_env_steps, self.phase_policy.budget)

    def _protocol_audit(self, episode_offset=None):
        seed = None
        if episode_offset is not None and self.episode_protocol_seeds:
            seed = self.episode_protocol_seeds[episode_offset]
        if self.phase_policy is None:
            return {
                'phase': 'unspecified',
                'protocol_seed': None,
                'reportable': True,
                'reset_allowed': None,
                'memory_write_allowed': None,
                'budget': self.max_env_steps,
                'budget_unit': 'environment_steps_per_episode',
                'initialization_reset': True,
                'exploratory_reset_used': False,
            }
        return {
            'phase': self.protocol_phase,
            'protocol_seed': seed,
            'episode_protocol_seeds': list(self.episode_protocol_seeds),
            'reportable': self.phase_policy.reportable,
            'reset_allowed': self.phase_policy.reset_allowed,
            'memory_write_allowed': self.phase_policy.memory_write_allowed,
            'budget': self.max_env_steps,
            'phase_budget': self.phase_policy.budget,
            'configured_max_env_steps': self.configured_max_env_steps,
            'budget_unit': 'environment_steps_per_episode',
            'initialization_reset': True,
            'exploratory_reset_used': False,
        }

    def write_memory(self, protocol_seed, writer, *args, **kwargs):
        """Run a memory writer only after validating phase and seed."""
        if self.phase_manifest is not None:
            self.phase_manifest.guard_operation(
                self.phase_policy.phase, protocol_seed, PhaseOperation.WRITE_MEMORY
            )
        return writer(*args, **kwargs)

    def _load_global_memory(self):
        base_memory = GlobalMemory.load(self.config.get('global_memory_path', ''))
        if not self.global_memory_ledger_path:
            return base_memory
        memory = GlobalMemory.from_ledger(
            self.global_memory_ledger_path,
            include_seed=self.global_memory_include_seed,
            base_memory=base_memory,
        )
        ledger = GlobalMemoryLedger.load(
            self.global_memory_ledger_path,
            read_only=self.phase_policy.phase is Phase.DEPLOYMENT,
        )
        ledger_sha256 = None
        if os.path.exists(self.global_memory_ledger_path):
            with open(self.global_memory_ledger_path, 'rb') as ledger_file:
                ledger_sha256 = hashlib.sha256(ledger_file.read()).hexdigest()
        self.global_memory_audit = ledger.audit()
        self.global_memory_audit.update({
            'path': self.global_memory_ledger_path,
            'decision': 'loaded_validated_ledger',
            'read_only': ledger.read_only,
            'ledger_sha256': ledger_sha256,
            'rendered_sha256': self._memory_hashes(memory)[
                'global_memory_rendered_sha256'
            ],
        })
        return memory

    def process_global_memory_trace(
        self, protocol_seed, trace_path, *, run_status
    ):
        """Process one complete bootstrap trace behind the phase writer guard."""
        if not self.global_memory_ledger_path:
            raise ValueError('global_memory_ledger_path is not configured')

        def process():
            ledger = GlobalMemoryLedger.load(self.global_memory_ledger_path)
            audit = ledger.process_trace(
                trace_path,
                run_status=run_status,
                ledger_path=self.global_memory_ledger_path,
            )
            memory = self._load_global_memory()
            memory_setter = getattr(self.planner, 'set_global_memory', None)
            if callable(memory_setter):
                memory_setter(memory)
            else:
                self.planner.global_memory = memory
            audit['rendered_sha256'] = self._memory_hashes(memory)[
                'global_memory_rendered_sha256'
            ]
            self.global_memory_audit = audit
            return audit

        return self.write_memory(protocol_seed, process)

    def _memory_hashes(self, global_memory):
        ledger_sha256 = None
        if self.global_memory_ledger_path:
            ledger_path = Path(self.global_memory_ledger_path)
            if ledger_path.is_file():
                ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        return {
            'task_memory_sha256': (
                self._structured_sha256(self._task_memory_payload)
                if self._task_memory_payload is not None else None
            ),
            'global_memory_rendered_sha256': self._structured_sha256(
                {'rendered': global_memory.render()}
            ),
            'global_memory_ledger_sha256': ledger_sha256,
        }

    def _verify_deployment_memory_unchanged(self):
        if self.phase_policy is None:
            return
        self._protocol_memory_after = self._memory_hashes(self.planner.global_memory)
        if (
            self.phase_policy.phase is Phase.DEPLOYMENT
            and self._protocol_memory_after != self._protocol_memory_before
        ):
            raise RuntimeError('deployment memory changed during evaluation')

    # -- persistence ------------------------------------------------------

    def _results_dir(self):
        res_path = os.path.join(self.env.log_path, 'results')
        os.makedirs(res_path, exist_ok=True)
        return res_path

    def save_episode_metric(self, episode_info):
        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
        with open(os.path.join(self._results_dir(), filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def save_trace(self, trace):
        trace_path = self._trace_path()
        initialize_jsonl(trace_path)
        for record in trace:
            append_jsonl_record(trace_path, record)

    def _trace_path(self):
        filename = 'trace_episode_{}.jsonl'.format(self.env._current_episode_num)
        return os.path.join(self._results_dir(), filename)

    def _record_trace(self, trace, record):
        record['protocol'] = dict(self.current_episode_protocol)
        trace.append(record)
        append_jsonl_record(self._trace_path(), record)

    def save_trace_summary(self):
        summary = summarize_trace_records(load_complete_jsonl(self._trace_path()))
        summary['global_memory'] = dict(self.global_memory_audit)
        filename = 'trace_summary_episode_{}.json'.format(self.env._current_episode_num)
        with open(os.path.join(self._results_dir(), filename), 'w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False)

    def _write_run_manifest(self, status, error=None):
        secret_keys = {'api_key', 'password', 'access_token', 'auth_token', 'secret'}
        redacted_config = {
            key: ('<redacted>' if key.lower() in secret_keys else value)
            for key, value in self.config.items()
        }
        manifest = {
            'status': status,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'commit': resolve_git_commit(os.getcwd()),
            'eval_set': self.eval_set,
            'selected_indexes': list(self.config.get('selected_indexes', []) or []),
            'completed_episodes': int(getattr(self.env, '_current_episode_num', 0)),
            'config': redacted_config,
            'task_memory': dict(self.task_memory_episode_audit),
            'protocol': self._protocol_audit(),
            'phase_manifest': (
                self.phase_manifest.to_dict() if self.phase_manifest is not None else None
            ),
            'phase_manifest_sha256': (
                hashlib.sha256(self.phase_manifest.to_json().encode('utf-8')).hexdigest()
                if self.phase_manifest is not None else None
            ),
            'memory_hashes_before': self._protocol_memory_before,
            'memory_hashes_after': self._protocol_memory_after,
            'global_memory': dict(self.global_memory_audit),
        }
        if error is not None:
            manifest['error'] = {
                'type': type(error).__name__,
                'message': str(error),
            }
        write_json_atomic(os.path.join(self.log_path, 'run_manifest.json'), manifest)

    def print_task_eval_results(self, filename):
        folder_path = self._results_dir()
        total, success, planner_steps, output_error = 0, 0, 0, 0
        semantic_rejects, no_progress_rejected = 0, 0
        grounding_observations = 0
        grounding_weighted_error = 0.0
        grounding_max_error = None
        postconditions_met, postconditions_failed = 0, 0
        termination_reasons = {}
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".json") and file_name.startswith("episode"):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                if not data.get('protocol', {}).get('reportable', True):
                    continue
                if data.get("planner_output_error", 0) > 0:
                    output_error += 1
                if data.get("task_success") == 1:
                    success += 1
                planner_steps += data.get("planner_steps", 0)
                semantic_rejects += data.get("semantic_rejects", 0)
                no_progress_rejected += data.get("no_progress_rejected", 0)
                grounding = data.get('grounding_metrics', {})
                postconditions_met += data.get('primitive_postconditions_met', 0)
                postconditions_failed += data.get('primitive_postconditions_failed', 0)
                for reason, count in data.get('termination_reasons', {}).items():
                    termination_reasons[reason] = termination_reasons.get(reason, 0) + count
                object_count = grounding.get('object_observation_count', 0)
                mean_error = grounding.get('mean_error_m')
                if object_count and mean_error is not None:
                    grounding_observations += object_count
                    grounding_weighted_error += object_count * mean_error
                episode_max = grounding.get('max_error_m')
                if episode_max is not None:
                    grounding_max_error = (
                        episode_max if grounding_max_error is None
                        else max(grounding_max_error, episode_max)
                    )
                total += 1
        task_log = {
            'save_path': self.log_path,
            'total_num_tasks': total,
            'num_success': success,
            'success_rate': success / total if total else 0.0,
            'avg_planner_steps': planner_steps / total if total else 0.0,
            'output_format_error': output_error,
            'semantic_rejects': semantic_rejects,
            'no_progress_rejected': no_progress_rejected,
            'grounding_object_observation_count': grounding_observations,
            'grounding_mean_surface_to_origin_error_m': (
                grounding_weighted_error / grounding_observations
                if grounding_observations else None
            ),
            'grounding_max_surface_to_origin_error_m': grounding_max_error,
            'primitive_postconditions_met': postconditions_met,
            'primitive_postconditions_failed': postconditions_failed,
            'termination_reasons': termination_reasons,
        }
        with open(os.path.join(folder_path, filename), 'w', encoding='utf-8') as f:
            json.dump(task_log, f, ensure_ascii=False)

    # -- perception helpers ----------------------------------------------

    def _perceive_grounding(self, obs):
        artifact = form_harness_grounding_artifact_for_input(
            copy.deepcopy(obs), self.env.task_class, ['front_rgb']
        )
        artifact['planner_coords'] = {
            key: [int(round(value)) for value in coord]
            for key, coord in artifact['planner_coords'].items()
        }
        metrics = compute_oracle_metrics(
            artifact, obs.get('object_informations', {})
        )
        return artifact, metrics

    def _save_grounding_audit(self, obs, artifact, metrics):
        if not self.config.get('save_grounding_audit', True):
            return None
        episode_dir = os.path.join(
            self.env.log_path, 'grounding',
            f'episode_{self.env._current_episode_num}',
        )
        os.makedirs(episode_dir, exist_ok=True)
        frame_id = artifact['frame_id']
        sidecar_path = os.path.join(episode_dir, f'frame_{frame_id:05d}.json')
        with open(sidecar_path, 'w', encoding='utf-8') as file:
            json.dump({'grounding': artifact, 'oracle_metrics': metrics}, file)

        rgb = obs.get('front_rgb')
        if rgb is None:
            return sidecar_path
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(image)
        for object_id, estimate in artifact.get('objects', {}).items():
            sample = next(
                (item for item in estimate.get('samples', []) if item['camera'] == 'front'),
                None,
            )
            if sample is None:
                continue
            x, y = sample['pixel_uv']
            radius = 4
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 32, 32), width=2,
            )
            draw.text((x + 6, y - 6), object_id, fill=(255, 255, 0))
        overlay_path = os.path.join(episode_dir, f'frame_{frame_id:05d}_front.png')
        image.save(overlay_path)
        return sidecar_path

    def _planner_act(
        self, instruction, coords, pose, history, roles, labels,
        held_object_id=None, attachment_evidence_available=False,
        resolved_task_memory=None,
    ):
        try:
            return self.planner.act(
                instruction, coords, pose, history,
                object_roles=roles, object_labels=labels,
                held_object_id=held_object_id,
                attachment_evidence_available=attachment_evidence_available,
                resolved_task_memory=resolved_task_memory,
            )
        except TypeError as exc:
            if 'unexpected keyword argument' not in str(exc):
                raise
            try:
                return self.planner.act(
                    instruction, coords, pose, history,
                    object_roles=roles, object_labels=labels,
                )
            except TypeError as fallback_exc:
                if 'unexpected keyword argument' not in str(fallback_exc):
                    raise
                return self.planner.act(instruction, coords, pose, history)

    def _current_pose(self, obs):
        pose = pose_from_observation(obs)
        return pose if pose is not None else PoseState()

    def _grasp_attachment_evidence(self, info=None):
        evidence_getter = getattr(self.env, 'get_grasp_attachment_evidence', None)
        if callable(evidence_getter):
            try:
                names, available = evidence_getter()
                return list(names), bool(available)
            except Exception:
                pass
        getter = getattr(self.env, 'get_grasped_object_names', None)
        if callable(getter):
            try:
                return list(getter()), True
            except Exception:
                pass
        if 'grasped_objects' in (info or {}):
            return list(info.get('grasped_objects', []) or []), True
        return [], False

    def _grasped_object_names(self, info=None):
        return self._grasp_attachment_evidence(info)[0]

    def _execute_openvla(
        self, obs_dict, instruction, invocation, id_to_sim_name, grounding_frames, remaining
    ):
        """Infer, execute, and reobserve one OpenVLA action at a time."""
        if remaining <= 0:
            raise OpenVLABackendError('no environment-step budget remains for vla_act')
        mode = invocation.get('mode')
        object_id = invocation.get('object') or invocation.get('target')
        destination_id = invocation.get('destination')
        state = {
            'obs_dict': obs_dict,
            'done': False,
            'info': {'task_success': 0},
            'grounding': None,
            'grounding_metrics': None,
            'attachments': [],
            'attachment_available': False,
            'step_results': [],
            'tau_reason': None,
        }

        def policy_observation(current_obs):
            front_rgb = current_obs.get('front_rgb')
            gripper_pose = current_obs.get('gripper_pose')
            if front_rgb is None or gripper_pose is None:
                raise OpenVLABackendError(
                    'OpenVLA requires live front_rgb and gripper_pose observations'
                )
            return OpenVLAObservation(front_rgb, gripper_pose, mode)

        def execute(action):
            obs, reward, done, info = self.env.step(list(action.converted_action))
            if self.config.get('save_images', False):
                self.env.save_image(['front_rgb'])
            current_obs = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
            grounding, metrics = self._perceive_grounding(current_obs)
            grounding_frames.append(metrics)
            self._save_grounding_audit(current_obs, grounding, metrics)
            attachments, attachment_available = self._grasp_attachment_evidence(info)
            state.update({
                'obs_dict': current_obs,
                'done': bool(done),
                'info': info,
                'grounding': grounding,
                'grounding_metrics': metrics,
                'attachments': attachments,
                'attachment_available': attachment_available,
            })
            state['step_results'].append({
                'action': list(action.converted_action),
                'raw_delta': list(action.raw_delta),
                'inference_duration_s': action.inference_duration_s,
                'reward': reward,
                'action_success': info['action_success'],
                'task_success': info['task_success'],
                'env_done': bool(done),
                'env_feedback': info.get('env_feedback', ''),
                'grasped_objects': attachments,
                'attachment_evidence_available': attachment_available,
            })
            return policy_observation(current_obs)

        def stop_after_step(_observation):
            if state['done']:
                state['tau_reason'] = 'environment_done'
                return True
            if mode == 'grasp' and state['attachment_available']:
                target_name = id_to_sim_name.get(object_id)
                attached = {_physical_object_name(n) for n in state['attachments']}
                if target_name and _physical_object_name(target_name) in attached:
                    state['tau_reason'] = 'target_attached'
                    return True
            if mode == 'place' and state['attachment_available'] and not state['attachments']:
                coords = state['grounding']['planner_coords']
                spatial = classify_spatial_postcondition(
                    coords.get(destination_id), coords.get(object_id), self.place_tolerance
                )
                if spatial['postcondition_met']:
                    state['tau_reason'] = 'released_at_destination'
                    return True
            return False

        prompt = f"{instruction.strip()}\nMode: {mode}."
        runtime_result = VLARuntime(self.openvla_backend).run(
            initial_observation=policy_observation(obs_dict),
            prompt=prompt,
            max_chunks=min(self.openvla_max_chunks, remaining),
            tau=stop_after_step,
            executor=execute,
        )
        state['stop_reason'] = state['tau_reason'] or runtime_result.termination_reason
        state['chunks_executed'] = runtime_result.chunks_executed
        return state

    def _execute_pirlinf(
        self, obs_dict, instruction, invocation, id_to_sim_name, grounding_frames, remaining
    ):
        """Infer PiRLinf chunks and execute each delta against the live pose."""
        if remaining <= 0:
            raise PiRLinfBackendError('no environment-step budget remains for vla_act')
        mode = invocation.get('mode')
        object_id = invocation.get('object') or invocation.get('target')
        destination_id = invocation.get('destination')
        state = {
            'obs_dict': obs_dict,
            'done': False,
            'info': {'task_success': 0},
            'grounding': None,
            'grounding_metrics': None,
            'attachments': [],
            'attachment_available': False,
            'step_results': [],
            'tau_reason': None,
            'gripper_qpos_source': None,
        }

        def policy_observation(current_obs):
            front_rgb = current_obs.get('front_rgb')
            wrist_rgb = current_obs.get('wrist_rgb')
            gripper_pose = current_obs.get('gripper_pose')
            if front_rgb is None or wrist_rgb is None or gripper_pose is None:
                raise PiRLinfBackendError(
                    'PiRLinf requires live front_rgb, wrist_rgb, and gripper_pose observations'
                )
            gripper_qpos = current_obs.get('gripper_joint_positions')
            if gripper_qpos is None:
                gripper_open = current_obs.get('gripper_open')
                if gripper_open is None:
                    raise PiRLinfBackendError(
                        'PiRLinf requires gripper_joint_positions or gripper_open'
                    )
                gripper_qpos = [0.04, 0.04] if float(gripper_open) > 0.5 else [0.0, 0.0]
                qpos_source = 'gripper_open_fallback'
            else:
                qpos_source = 'gripper_joint_positions'
            state['gripper_qpos_source'] = qpos_source
            return PiRLinfObservation(
                front_rgb, wrist_rgb, gripper_pose, gripper_qpos, mode
            )

        def execute(chunk):
            qpos_source = state['gripper_qpos_source']
            for raw_delta in chunk.raw_deltas:
                if state['done'] or len(state['step_results']) >= remaining:
                    break
                gripper_pose = state['obs_dict'].get('gripper_pose')
                if gripper_pose is None:
                    raise PiRLinfBackendError(
                        'PiRLinf requires live gripper_pose for action conversion'
                    )
                try:
                    action = convert_libero_delta_to_eb(
                        raw_delta, gripper_pose, **self.openvla_conversion
                    )
                except OpenVLABackendError as exc:
                    raise PiRLinfBackendError(
                        f'PiRLinf action conversion failed: {exc}'
                    ) from exc
                obs, reward, done, info = self.env.step(action)
                if self.config.get('save_images', False):
                    self.env.save_image(['front_rgb'])
                current_obs = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
                grounding, metrics = self._perceive_grounding(current_obs)
                grounding_frames.append(metrics)
                self._save_grounding_audit(current_obs, grounding, metrics)
                attachments, attachment_available = self._grasp_attachment_evidence(info)
                state.update({
                    'obs_dict': current_obs,
                    'done': bool(done),
                    'info': info,
                    'grounding': grounding,
                    'grounding_metrics': metrics,
                    'attachments': attachments,
                    'attachment_available': attachment_available,
                })
                state['step_results'].append({
                    'action': list(action),
                    'raw_delta': list(raw_delta),
                    'inference_duration_s': chunk.inference_duration_s,
                    'full_chunk_length': chunk.full_chunk_length,
                    'gripper_qpos_source': qpos_source,
                    'reward': reward,
                    'action_success': info['action_success'],
                    'task_success': info['task_success'],
                    'env_done': bool(done),
                    'env_feedback': info.get('env_feedback', ''),
                    'grasped_objects': attachments,
                    'attachment_evidence_available': attachment_available,
                })
            return policy_observation(state['obs_dict'])

        def stop_after_chunk(_observation):
            if state['done']:
                state['tau_reason'] = 'environment_done'
                return True
            if mode == 'grasp' and state['attachment_available']:
                target_name = id_to_sim_name.get(object_id)
                attached = {_physical_object_name(n) for n in state['attachments']}
                if target_name and _physical_object_name(target_name) in attached:
                    state['tau_reason'] = 'target_attached'
                    return True
            if mode == 'place' and state['attachment_available'] and not state['attachments']:
                coords = state['grounding']['planner_coords']
                spatial = classify_spatial_postcondition(
                    coords.get(destination_id), coords.get(object_id), self.place_tolerance
                )
                if spatial['postcondition_met']:
                    state['tau_reason'] = 'released_at_destination'
                    return True
            return False

        prompt = f"{instruction.strip()}\nMode: {mode}."
        runtime_result = VLARuntime(self.pirlinf_backend).run(
            initial_observation=policy_observation(obs_dict),
            prompt=prompt,
            max_chunks=min(
                self.pirlinf_max_chunks,
                math.ceil(remaining / self.pirlinf_replan_steps),
            ),
            tau=stop_after_chunk,
            executor=execute,
        )
        state['stop_reason'] = state['tau_reason'] or runtime_result.termination_reason
        state['chunks_executed'] = runtime_result.chunks_executed
        return state

    # -- main loop --------------------------------------------------------

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {
                'reward': [], 'action_success': [],
                'semantic_rejects': 0, 'no_progress_rejected': 0,
                'primitive_postconditions_met': 0,
                'primitive_postconditions_failed': 0,
                'termination_reasons': {},
            }
            trace = []
            grounding_frames = []
            self.task_memory_episode_audit = dict(self.task_memory_audit)

            episode_offset = self.env._current_episode_num
            _, obs = self.env.reset()
            self.current_episode_protocol = self._protocol_audit(episode_offset)
            initialize_jsonl(self._trace_path())
            if self.phase_manifest is not None:
                self._record_trace(trace, {
                    'turn': 0,
                    'status': 'initialization_reset',
                    'execution_status': 'initialization',
                    'invocation': None,
                })
            obs_dict = vars(copy.deepcopy(obs))
            if self.config.get('save_images', False):
                self.env.save_image(['front_rgb'])
            grounding, grounding_metrics = self._perceive_grounding(obs_dict)
            grounding_frames.append(grounding_metrics)
            self._save_grounding_audit(obs_dict, grounding, grounding_metrics)
            avg_obj_coord = grounding['planner_coords']
            object_roles = grounding['roles']
            object_labels = grounding['labels']
            id_to_sim_name = grounding['id_to_sim_name']
            pose = self._current_pose(obs_dict)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            history = []
            held_object_id = None
            held_evidence_available = False
            placed_object_ids = set()
            no_progress_guard = NoProgressGuard(limit=3)
            done = False
            info = {'task_success': 0, 'episode_elapsed_seconds': 0}
            turn = 0

            while not done and turn < self.max_turns:
                turn += 1
                resolved_task_memory = None
                if self.task_memory_episode_audit['decision'] == 'rejected':
                    feedback = (
                        'Configured Task Specific Memory rejected before action: '
                        + self.task_memory_episode_audit['reason']
                    )
                    self._record_trace(trace, {
                        'turn': turn,
                        'status': 'task_memory_rejected',
                        'execution_status': 'not_executed',
                        'feedback': feedback,
                        'task_memory': dict(self.task_memory_episode_audit),
                        'invocation': None,
                    })
                    break
                if self.task_memory_commands:
                    try:
                        resolved_task_memory = resolve_task_memory_commands(
                            self.task_memory_commands,
                            avg_obj_coord,
                            object_labels,
                            object_roles,
                        )
                    except ValueError as error:
                        self.task_memory_episode_audit.update({
                            'decision': 'rejected',
                            'stage': 'grounding',
                            'rejection_turn': turn,
                            'reason': str(error),
                        })
                        feedback = (
                            'Configured Task Specific Memory rejected before action: '
                            + str(error)
                        )
                        self._record_trace(trace, {
                            'turn': turn,
                            'status': 'task_memory_rejected',
                            'execution_status': 'not_executed',
                            'feedback': feedback,
                            'task_memory': dict(self.task_memory_episode_audit),
                            'invocation': None,
                        })
                        break
                try:
                    invocation, raw_text = self._planner_act(
                        user_instruction, avg_obj_coord, pose.as_action(), history,
                        object_roles, object_labels,
                        held_object_id=held_object_id,
                        attachment_evidence_available=held_evidence_available,
                        resolved_task_memory=resolved_task_memory,
                    )
                except (socket.timeout, TimeoutError, OSError) as error:
                    feedback = f'Planner request failed ({type(error).__name__}: {error}). Retrying next turn.'
                    logger.warning(feedback)
                    self._record_trace(trace, {
                        'turn': turn,
                        'pose_before': pose.as_action(),
                        'status': 'planner_request_error',
                        'feedback': feedback,
                        'invocation': None,
                    })
                    history.append({
                        'action': None,
                        'status': 'planner_request_error',
                        'feedback': feedback,
                    })
                    continue
                record = {
                    'turn': turn,
                    'pose_before': pose.as_action(),
                    'object_coords': avg_obj_coord,
                    'object_roles': object_roles,
                    'object_labels': object_labels,
                    'id_to_sim_name': id_to_sim_name,
                    'raw_output': raw_text,
                    'planner_thinking': getattr(self.planner, 'last_thinking', None),
                    'invocation': invocation,
                    'grounding_frame_id': grounding['frame_id'],
                    'grounding_coordinate_source': grounding['coordinate_source'],
                    'grounding_objects': grounding['objects'],
                    'grounding_oracle_metrics': grounding_metrics,
                    'task_memory': dict(self.task_memory_episode_audit),
                }

                if invocation is None:
                    feedback = "Output was not valid JSON with an 'action'. Reply with one primitive as JSON."
                    record['status'] = 'parse_error'
                    record['feedback'] = feedback
                    history.append({'action': None, 'status': 'parse_error', 'feedback': feedback})
                    self._record_trace(trace, record)
                    continue

                semantic_rejection = validate_vla_semantics(
                    invocation, avg_obj_coord, object_roles, held_object_id, object_labels
                )
                if semantic_rejection is not None:
                    status, feedback = semantic_rejection
                    record.update({
                        'execution_status': 'not_executed',
                        'status': status,
                        'semantic_reject': True,
                        'feedback': feedback,
                        'held_object_id': held_object_id,
                    })
                    episode_info['semantic_rejects'] += 1
                    history.append({'action': invocation, 'status': status, 'feedback': feedback})
                    self._record_trace(trace, record)
                    continue

                if no_progress_guard.should_reject(invocation):
                    feedback = (
                        "Repeated action rejected before execution after 3 identical zero-reward, "
                        "zero-task-progress executions. Choose a different action or re-stage; the "
                        "episode remains active."
                    )
                    record.update({
                        'execution_status': 'not_executed',
                        'status': 'no_progress_rejected',
                        'no_progress_rejected': True,
                        'feedback': feedback,
                        'held_object_id': held_object_id,
                    })
                    episode_info['no_progress_rejected'] += 1
                    history.append({
                        'action': invocation, 'status': 'no_progress_rejected', 'feedback': feedback
                    })
                    self._record_trace(trace, record)
                    continue

                if invocation.get('action') == 'vla_act' and (
                    self.openvla_backend is not None or self.pirlinf_backend is not None
                ):
                    remaining = self.env._max_episode_steps - self.env._current_step
                    backend_name = self.vla_backend_name
                    try:
                        if self.pirlinf_backend is not None:
                            vla_execution = self._execute_pirlinf(
                                obs_dict, user_instruction, invocation, id_to_sim_name,
                                grounding_frames, remaining,
                            )
                        else:
                            vla_execution = self._execute_openvla(
                                obs_dict, user_instruction, invocation, id_to_sim_name,
                                grounding_frames, remaining,
                            )
                    except (OpenVLABackendError, PiRLinfBackendError) as error:
                        feedback = f'{backend_name} vla_act failed: {error}'
                        record.update({
                            'backend': backend_name,
                            'execution_status': 'failed',
                            'status': 'vla_backend_error',
                            'feedback': feedback,
                            'held_object_id': held_object_id,
                            'termination_reason': 'backend_error',
                        })
                        history.append({
                            'action': invocation,
                            'status': 'vla_backend_error',
                            'feedback': feedback,
                        })
                        self._record_trace(trace, record)
                        continue

                    step_results = vla_execution['step_results']
                    obs_dict = vla_execution['obs_dict']
                    done = vla_execution['done']
                    info = vla_execution['info']
                    grounding = vla_execution['grounding']
                    grounding_metrics = vla_execution['grounding_metrics']
                    avg_obj_coord = grounding['planner_coords']
                    object_roles = grounding['roles']
                    object_labels = grounding['labels']
                    id_to_sim_name = grounding['id_to_sim_name']
                    pose = self._current_pose(obs_dict)
                    for step in step_results:
                        episode_info['reward'].append(step['reward'])
                        episode_info['action_success'].append(step['action_success'])

                    mode = invocation.get('mode')
                    object_id = invocation.get('object') or invocation.get('target')
                    destination_id = invocation.get('destination')
                    attachments = vla_execution['attachments']
                    attachment_available = vla_execution['attachment_available']
                    held_object_id, held_evidence_available = reconcile_held_object(
                        held_object_id, attachments, attachment_available, id_to_sim_name
                    )
                    spatial_postcondition = None
                    grasp_outcome = None
                    release_executed = bool(
                        mode == 'place'
                        and any(step['action'][6] == 1 for step in step_results)
                    )
                    if mode == 'grasp':
                        grasp_outcome = classify_grasp_outcome(
                            target_object_id=object_id,
                            target_sim_name=id_to_sim_name.get(object_id),
                            grasped_object_names=attachments if attachment_available else [],
                        )
                        if grasp_outcome['outcome'] == 'grasp_verified':
                            held_object_id = object_id
                            placed_object_ids.discard(object_id)
                    elif mode == 'place':
                        spatial_postcondition = classify_spatial_postcondition(
                            avg_obj_coord.get(destination_id),
                            avg_obj_coord.get(object_id),
                            self.place_tolerance,
                        )
                    termination_reason, primitive_postcondition_met = primitive_termination(
                        mode=mode,
                        grasp_outcome=grasp_outcome['outcome'] if grasp_outcome else None,
                        env_done=bool(done),
                        release_executed=release_executed,
                        attachment_evidence_available=attachment_available,
                        grasped_object_names=attachments,
                        primitive_name='vla_act',
                        spatial_postcondition_met=(
                            spatial_postcondition['postcondition_met']
                            if spatial_postcondition else None
                        ),
                    )
                    if mode == 'place' and primitive_postcondition_met:
                        placed_object_ids.add(object_id)
                        held_object_id = None
                    episode_info[
                        'primitive_postconditions_met'
                        if primitive_postcondition_met
                        else 'primitive_postconditions_failed'
                    ] += 1
                    episode_info['termination_reasons'][termination_reason] = (
                        episode_info['termination_reasons'].get(termination_reason, 0) + 1
                    )
                    execution_status = (
                        'success'
                        if step_results and step_results[-1]['action_success'] == 1.0
                        else 'failed'
                    )
                    feedback = (
                        step_results[-1]['env_feedback']
                        if step_results else f'{backend_name} executed no actions'
                    )
                    if mode == 'place' and primitive_postcondition_met and not done:
                        feedback += (
                            " Benchmark task signal: NOT successful yet - the placement "
                            "did not complete the task. The task is unfinished; keep acting."
                        )
                    no_progress_guard.observe_execution(invocation, step_results)
                    record.update({
                        'backend': backend_name,
                        'primitive': 'vla_act',
                        'is_contact': True,
                        'meta': {
                            'mode': mode,
                            'object_id': object_id,
                            'destination_id': destination_id,
                        },
                        'compiled_actions': [step['action'] for step in step_results],
                        'raw_deltas': [step['raw_delta'] for step in step_results],
                        'inference_durations_s': [
                            step['inference_duration_s'] for step in step_results
                        ],
                        'step_results': step_results,
                        'pose_after': pose.as_action(),
                        'execution_status': execution_status,
                        'status': (
                            grasp_outcome['outcome'] if grasp_outcome
                            else 'postcondition_met' if primitive_postcondition_met
                            else termination_reason
                        ),
                        'feedback': feedback,
                        'held_object_id': held_object_id,
                        'env_done': bool(done),
                        'termination_reason': termination_reason,
                        'gripper_qpos_sources': [
                            step.get('gripper_qpos_source') for step in step_results
                            if step.get('gripper_qpos_source') is not None
                        ],
                        'vla_stop_reason': vla_execution['stop_reason'],
                        'vla_chunks_executed': vla_execution['chunks_executed'],
                        'primitive_postcondition_met': primitive_postcondition_met,
                        'spatial_postcondition': spatial_postcondition,
                        'physical_state': summarize_physical_state(
                            object_roles, held_object_id, placed_object_ids
                        ),
                    })
                    if grasp_outcome:
                        record['grasp_outcome'] = grasp_outcome
                    history.append({
                        'action': invocation,
                        'status': record['status'],
                        'feedback': feedback,
                    })
                    self._record_trace(trace, record)
                    continue

                # Compile the primitive to discrete actions.
                try:
                    result = self.library.compile(invocation, pose, avg_obj_coord)
                except PrimitiveError as e:
                    feedback = f"Primitive could not be executed: {e}"
                    record['status'] = 'compile_error'
                    record['feedback'] = feedback
                    history.append({'action': invocation, 'status': 'compile_error', 'feedback': feedback})
                    self._record_trace(trace, record)
                    continue

                mode = result.meta.get('mode')

                # Execute the compiled actions (bounded by remaining env steps).
                remaining = self.env._max_episode_steps - self.env._current_step
                step_results = []
                last_feedback = ''
                object_id = result.meta.get('object_id')
                position_pre_grasp = avg_obj_coord.get(object_id) if mode == 'grasp' else None
                gripper_pre_grasp = pose.as_action()[:3] if mode == 'grasp' else None
                position_at_close = None
                gripper_at_close = None
                attachment_at_close = []
                attachment_evidence_available = False
                for subaction_index, action_single in enumerate(result.actions[:max(0, remaining)]):
                    obs, reward, done, info = self.env.step(action_single)
                    if self.config.get('save_images', False):
                        self.env.save_image(['front_rgb'])
                    last_feedback = info.get('env_feedback', '')
                    sub_obs_dict = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
                    sub_grounding, sub_grounding_metrics = self._perceive_grounding(
                        sub_obs_dict
                    )
                    grounding_frames.append(sub_grounding_metrics)
                    self._save_grounding_audit(
                        sub_obs_dict, sub_grounding, sub_grounding_metrics
                    )
                    sub_coords = sub_grounding['planner_coords']
                    sub_pose = self._current_pose(sub_obs_dict)
                    attachments, attachment_available = self._grasp_attachment_evidence(info)
                    attachment_evidence_available = (
                        attachment_evidence_available or attachment_available
                    )
                    step_results.append({
                        'action': list(action_single),
                        'reward': reward,
                        'action_success': info['action_success'],
                        'task_success': info['task_success'],
                        'env_done': bool(done),
                        'env_feedback': last_feedback,
                        'grasped_objects': attachments,
                        'attachment_evidence_available': attachment_available,
                    })
                    if mode == 'grasp' and subaction_index == 2:
                        position_at_close = sub_coords.get(object_id)
                        gripper_at_close = sub_pose.as_action()[:3]
                        attachment_at_close = attachments
                    episode_info['reward'].append(reward)
                    episode_info['action_success'].append(info['action_success'])
                    print(f"Executed {result.name} sub-action {action_single}, task_success={info['task_success']}")
                    if done:
                        break

                obs_dict = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
                grounding, grounding_metrics = self._perceive_grounding(obs_dict)
                grounding_frames.append(grounding_metrics)
                self._save_grounding_audit(obs_dict, grounding, grounding_metrics)
                avg_obj_coord = grounding['planner_coords']
                object_roles = grounding['roles']
                object_labels = grounding['labels']
                id_to_sim_name = grounding['id_to_sim_name']
                pose = self._current_pose(obs_dict)

                execution_status = 'success' if step_results and step_results[-1]['action_success'] == 1.0 else 'failed'
                feedback = last_feedback or 'no environment feedback'
                grasp_outcome = None
                release_executed = bool(
                    (mode == 'place' or result.name == 'release')
                    and any(step['action'][6] == 1 for step in step_results)
                )
                if mode == 'grasp':
                    final_attachments = (
                        step_results[-1]['grasped_objects'] if step_results else attachment_at_close
                    )
                    grasp_outcome = classify_grasp_outcome(
                        target_object_id=object_id,
                        target_sim_name=id_to_sim_name.get(object_id),
                        grasped_object_names=final_attachments,
                        object_position_at_close=position_at_close,
                        object_position_after_lift=avg_obj_coord.get(object_id),
                        gripper_position_at_close=gripper_at_close,
                        gripper_position_after_lift=pose.as_action()[:3],
                        **self.grasp_thresholds,
                    )
                    if grasp_outcome['outcome'] == 'grasp_verified':
                        held_object_id = object_id
                        placed_object_ids.discard(object_id)
                    else:
                        held_object_id = None
                    feedback = (
                        f"Grasp outcome: {grasp_outcome['outcome']} "
                        f"({grasp_outcome['reason']}). action execution: {execution_status}."
                    )
                spatial_postcondition = None
                if result.name == 'move_to':
                    spatial_postcondition = classify_spatial_postcondition(
                        result.end_pose.as_action()[:3],
                        pose.as_action()[:3],
                        self.move_to_tolerance,
                    )
                elif mode == 'place':
                    spatial_postcondition = classify_spatial_postcondition(
                        avg_obj_coord.get(result.meta.get('destination_id')),
                        avg_obj_coord.get(object_id),
                        self.place_tolerance,
                    )
                termination_reason, primitive_postcondition_met = primitive_termination(
                    mode=mode,
                    grasp_outcome=grasp_outcome['outcome'] if grasp_outcome else None,
                    env_done=bool(done),
                    release_executed=release_executed,
                    attachment_evidence_available=attachment_evidence_available,
                    grasped_object_names=(
                        step_results[-1]['grasped_objects'] if step_results else []
                    ),
                    primitive_name=result.name,
                    spatial_postcondition_met=(
                        spatial_postcondition['postcondition_met']
                        if spatial_postcondition else None
                    ),
                )
                if mode == 'place' and primitive_postcondition_met:
                    placed_object_ids.add(object_id)
                detached = bool(
                    release_executed
                    and attachment_evidence_available
                    and not (step_results[-1]['grasped_objects'] if step_results else [])
                )
                if detached:
                    held_object_id = None
                held_object_id, held_evidence_available = reconcile_held_object(
                    held_object_id,
                    step_results[-1]['grasped_objects'] if step_results else [],
                    attachment_evidence_available,
                    id_to_sim_name,
                )
                if spatial_postcondition is not None:
                    distance = spatial_postcondition['distance']
                    distance_text = 'unknown' if distance is None else f'{distance:.3f}'
                    feedback = (
                        f"{result.name} postcondition: {termination_reason}; "
                        f"spatial={spatial_postcondition['reason']} "
                        f"(distance={distance_text}, tolerance="
                        f"{spatial_postcondition['tolerance']:.3f} voxels). "
                        f"action execution: {execution_status}."
                    )
                elif result.name == 'release':
                    feedback = (
                        f"Release outcome: {termination_reason}; detached={detached}. "
                        f"action execution: {execution_status}."
                    )
                if mode == 'place' and primitive_postcondition_met and not done:
                    feedback += (
                        " Benchmark task signal: NOT successful yet - the placement "
                        "did not complete the task. The task is unfinished; keep acting "
                        "(e.g. verify the placement or continue with the next object)."
                    )
                physical_state = summarize_physical_state(
                    object_roles, held_object_id, placed_object_ids
                )
                episode_info[
                    'primitive_postconditions_met'
                    if primitive_postcondition_met
                    else 'primitive_postconditions_failed'
                ] += 1
                episode_info['termination_reasons'][termination_reason] = (
                    episode_info['termination_reasons'].get(termination_reason, 0) + 1
                )
                no_progress_guard.observe_execution(invocation, step_results)
                record.update({
                    'primitive': result.name,
                    'is_contact': result.is_contact,
                    'meta': result.meta,
                    'compiled_actions': [list(a) for a in result.actions],
                    'step_results': step_results,
                    'pose_after': pose.as_action(),
                    'execution_status': execution_status,
                    'status': (
                        grasp_outcome['outcome'] if grasp_outcome
                        else 'postcondition_met' if primitive_postcondition_met
                        else termination_reason
                    ),
                    'feedback': feedback,
                    'held_object_id': held_object_id,
                    'env_done': bool(done),
                    'termination_reason': termination_reason,
                    'primitive_postcondition_met': primitive_postcondition_met,
                    'spatial_postcondition': spatial_postcondition,
                    'physical_state': physical_state,
                })
                if grasp_outcome:
                    record['grasp_evidence'] = {
                        'object_position_pre_grasp': position_pre_grasp,
                        'gripper_position_pre_grasp': gripper_pre_grasp,
                        'object_position_at_close': position_at_close,
                        'object_position_after_lift': avg_obj_coord.get(object_id),
                        'gripper_position_at_close': gripper_at_close,
                        'gripper_position_after_lift': pose.as_action()[:3],
                        'attachment_at_close': attachment_at_close,
                    }
                    record['grasp_outcome'] = grasp_outcome
                history.append({'action': invocation, 'status': record['status'], 'feedback': feedback})
                self._record_trace(trace, record)

            # Episode metrics.
            episode_info['instruction'] = user_instruction
            episode_info['avg_reward'] = float(np.mean(episode_info['reward'])) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = self.env._current_step
            episode_info['num_turns'] = turn
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info['episode_elapsed_seconds'] = info.get('episode_elapsed_seconds', 0)
            episode_info['grounding_metrics'] = summarize_oracle_frames(
                grounding_frames
            )
            episode_info['physical_state_final'] = summarize_physical_state(
                object_roles, held_object_id, placed_object_ids
            )
            episode_info['task_memory'] = dict(self.task_memory_episode_audit)
            episode_info['protocol'] = dict(self.current_episode_protocol)
            episode_info['memory_hashes_before'] = self._protocol_memory_before
            if (
                self.global_memory_ledger_path
                and self.phase_policy is not None
                and self.phase_policy.phase is Phase.BOOTSTRAP
            ):
                for record in trace:
                    record['task_success'] = info['task_success']
                    record['episode_status'] = 'completed'
                self.save_trace(trace)
                self.process_global_memory_trace(
                    self.current_episode_protocol['protocol_seed'],
                    self._trace_path(),
                    run_status='completed',
                )
            self._verify_deployment_memory_unchanged()
            episode_info['memory_hashes_after'] = self._protocol_memory_after
            self.save_episode_metric(episode_info)
            self.save_trace_summary()
            progress_bar.update()

        self.print_task_eval_results(filename="summary.json")
        self.env.close()

    def evaluate_main(self):
        valid_eval_sets = list(self.config.get('eval_sets') or ValidEvalSets)
        if not valid_eval_sets:
            valid_eval_sets = ValidEvalSets

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            real_model_name = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
            real_model_name = real_model_name.replace(':', '_')
            exp = self.config.get('exp_name') or 'harness'
            run_root = self.config.get('run_root')
            if run_root:
                self.log_path = os.path.join(run_root, self.eval_set)
            else:
                output_root = self.config.get('output_root') or 'running/eb_manipulation_harness'
                self.log_path = os.path.join(
                    output_root, real_model_name, exp, self.eval_set
                )
            if self.phase_manifest is not None:
                os.makedirs(self.log_path, exist_ok=True)
                write_json_atomic(
                    os.path.join(self.log_path, 'phase_manifest.json'),
                    self.phase_manifest.to_dict(),
                )
            self.env = EBManEnv(
                eval_set=self.eval_set,
                render_mode=self.config.get('render_mode', 'human'),
                img_size=(self.config['resolution'], self.config['resolution']),
                down_sample_ratio=self.config['down_sample_ratio'],
                selected_indexes=list(self.config.get('selected_indexes', []) or []),
                headless=self.config.get('headless', True),
                log_path=self.log_path,
            )
            self.env._max_episode_steps = self.max_env_steps
            global_memory = self._load_global_memory()
            self._protocol_memory_before = self._memory_hashes(global_memory)
            self._protocol_memory_after = None
            self.planner = HarnessPlanner(
                model_name=self.model_name,
                base_url=self.config.get('base_url'),
                api_key=self.config.get('api_key'),
                global_memory=global_memory,
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 1024),
                num_ctx=self.config.get('num_ctx'),
                disable_thinking=self.config.get('disable_thinking', False),
                enable_thinking=self.config.get('enable_thinking', False),
                request_timeout=self.config.get('request_timeout', 600.0),
            )
            self._write_run_manifest('running')
            try:
                self.evaluate()
                self._verify_deployment_memory_unchanged()
            except BaseException as error:
                self._write_run_manifest('incomplete', error=error)
                raise
            else:
                self._write_run_manifest('completed')
            finally:
                with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                    f.write(str(self.config))

    def check_config_valid(self):
        # Beta is language-only by construction; warn if configured otherwise.
        if not self.config.get('language_only', 1):
            logger.warning("Harness beta is language-only; forcing language_only=1.")
            self.config['language_only'] = 1
