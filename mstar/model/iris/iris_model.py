import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    DEFAULT_PARTITION,
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import KVReqConfig, NodeResourceSpec, ResourceReqConfig, SamplingReqConfig
from mstar.graph.base import GraphEdge, GraphSection, TensorPointerInfo
from mstar.model.base import ForwardPassArgs, KVLabel, Model, ProcessPromptOutput
from mstar.model.iris.config import (
    ENDING_SAMPLER,
    HF_REPO,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGE_SHAPE,
    IMAGINE_WALK,
    INITIAL_DECODER,
    NUM_ACTIONS,
    OBSERVATION_SAMPLER,
    OUTPUT_MODALITIES,
    REWARD_SAMPLER,
    ROLLOUT_LOOP,
    TOKEN_LOOP,
    TOKENS_PER_FRAME,
    WORLD_KV,
    WORLD_MODEL,
    IrisConfig,
)
from mstar.model.iris.graph import imagination_graph
from mstar.model.iris.resources import context_epochs, iris_resources, request_kv_layout
from mstar.model.multimodal import PromptPart


class IrisModel(Model):
    """Bounded supplied-action imagination using the pinned Breakout networks."""

    PREPROCESS_ONLY_KWARGS = ("actions",)
    INTERNAL_KWARGS = ("iris_horizon",)
    # The supplied trajectory must fail admission rather than be shortened to fit.
    LENGTH_KWARGS = ("iris_horizon",)

    def __init__(
        self,
        model_path_hf: str = HF_REPO,
        cache_dir: str | None = None,
        game: str = "Breakout",
        max_rollout_horizon: int = 20,
        skip_weight_loading: bool = False,
        checkpoint_path: str | None = None,
        max_batch_size: int = 2,
    ):
        self.config = IrisConfig(game=game, max_rollout_horizon=max_rollout_horizon, max_batch_size=max_batch_size)
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self._checkpoint_path = checkpoint_path
        self._networks = {}
        self._submodules = {}

    @staticmethod
    def _validate_modalities(input_modalities, output_modalities):
        if input_modalities != ["image"]:
            raise ValueError("IRIS requires exactly one image input")
        if output_modalities != list(OUTPUT_MODALITIES):
            raise ValueError(f"IRIS requires output_modalities={list(OUTPUT_MODALITIES)!r}")

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        self._validate_modalities(input_modalities, output_modalities)
        if prompt not in (None, ""):
            raise ValueError("IRIS does not accept text conditioning")
        if prompt_parts is not None and (
            len(prompt_parts) != 1 or prompt_parts[0].modality != "image" or prompt_parts[0].index != 0
        ):
            raise ValueError("IRIS prompt_parts must describe the single initial image")
        if kwargs.get("max_output_tokens") is not None:
            raise ValueError("IRIS horizon is the number of supplied actions; omit max_output_tokens")

        tensors = tensors or {}
        images = tensors.get("image_inputs", [])
        if len(images) != 1 or any(items for key, items in tensors.items() if key != "image_inputs"):
            raise ValueError("IRIS requires one image_inputs tensor and no other media inputs")
        image = images[0]
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or image.layout != torch.strided
            or image.dtype != torch.float32
            or tuple(image.shape) != IMAGE_SHAPE
        ):
            raise ValueError("IRIS image must be a CPU float32 tensor shaped [3, 64, 64]")
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError("IRIS image values must be finite and in [0, 1]")

        actions = kwargs.get("actions")
        if isinstance(actions, list):
            if not 1 <= len(actions) <= self.config.max_rollout_horizon:
                raise ValueError(f"IRIS requires 1..{self.config.max_rollout_horizon} actions")
            if any(type(action) is not int or not 0 <= action < NUM_ACTIONS for action in actions):
                raise ValueError("IRIS action IDs must be integers in [0, 4); bool is not an action ID")
            actions = torch.tensor(actions, dtype=torch.int64)
        if (
            not isinstance(actions, torch.Tensor)
            or actions.device.type != "cpu"
            or actions.layout != torch.strided
            or actions.dtype != torch.int64
            or actions.ndim != 1
        ):
            raise ValueError("IRIS actions must be a list of IDs or a CPU int64 tensor shaped [H]")
        if not 1 <= actions.numel() <= self.config.max_rollout_horizon:
            raise ValueError(f"IRIS requires 1..{self.config.max_rollout_horizon} actions")
        if (actions < 0).any() or (actions >= NUM_ACTIONS).any():
            raise ValueError("IRIS action IDs must be integers in [0, 4)")

        return ProcessPromptOutput(
            new_input_tensors={
                "image_inputs": [],
                "initial_image": [image.contiguous()],
                "actions": [actions.contiguous()],
            },
            metadata={"iris_horizon": actions.numel(), "kv_layout": request_kv_layout(actions.numel())},
        )

    def _horizon(self, model_kwargs: dict | None) -> int:
        horizon = (model_kwargs or {}).get("iris_horizon")
        if type(horizon) is not int or not 1 <= horizon <= self.config.max_rollout_horizon:
            raise ValueError("IRIS requires a validated iris_horizon from process_prompt")
        return horizon

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if partition_name != DEFAULT_PARTITION:
            raise ValueError(f"Unknown IRIS partition: {partition_name!r}")
        self._validate_modalities(input_modalities, output_modalities)
        horizon = self._horizon(model_kwargs)
        inputs = []
        for name, node, shape, dtype in (
            ("initial_image", IMAGE_ENCODER, IMAGE_SHAPE, torch.float32),
            ("actions", WORLD_MODEL, (horizon,), torch.int64),
        ):
            infos = input_signals.get(name, [])
            if len(infos) != 1 or tuple(infos[0].dims) != shape or infos[0].dtype != dtype:
                raise ValueError(f"IRIS {name} needs one {dtype} descriptor shaped {shape}")
            inputs.append(GraphEdge(next_node=node, name=name, tensor_info=list(infos)))
        metadata = CurrentForwardConductorMetadata(
            graph_walk=IMAGINE_WALK,
            is_prefill=True,
            input_modalities=list(input_modalities),
            output_modalities=list(output_modalities),
            kwargs={"iris_horizon": horizon},
        )
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=[info for edge in inputs for info in edge.tensor_info],
            step_metadata={"iris_horizon": horizon},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name != DEFAULT_PARTITION or partition_metadata.graph_walk != IMAGINE_WALK:
            raise ValueError("IRIS can finish only its default partition's imagine walk")
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[],
            unpersist_tensors=[],
            request_done=True,
        )

    def get_max_output_tokens(self, **model_kwargs):
        return self._horizon(model_kwargs)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {ROLLOUT_LOOP: self._horizon(model_kwargs), TOKEN_LOOP: TOKENS_PER_FRAME + 1}

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"actions"})

    def get_autocast_dtype(self):
        return None

    def get_output_frame_rate(self, modality="video_frame", request_kwargs=None) -> float:
        if modality != "video_frame":
            raise ValueError(f"Unsupported IRIS frame-rate modality: {modality!r}")
        # Matches the official game's recorded-video playback rate.
        return 15.0

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs=None) -> bytes:
        output = output.detach().cpu()
        if modality == "video_frame":
            if output.dtype != torch.uint8 or tuple(output.shape) != (1, 64, 64, 3):
                raise ValueError("IRIS video_frame must be uint8 shaped [1, 64, 64, 3]")
        elif modality == "scalar":
            if output.dtype != torch.float32 or tuple(output.shape) != (1,):
                raise ValueError("IRIS reward must be float32 shaped [1]")
            if not ((output == -1) | (output == 0) | (output == 1)).all():
                raise ValueError("IRIS reward must be -1, 0 or 1")
            return output.numpy().astype("<f4", copy=False).tobytes()
        elif modality == "tensor":
            if output.dtype != torch.bool or tuple(output.shape) != (1,):
                raise ValueError("IRIS ending flag must be bool shaped [1]")
        else:
            raise ValueError(f"Unsupported IRIS output modality: {modality!r}")
        return output.contiguous().numpy().tobytes()

    def get_node_resources(self) -> list[NodeResourceSpec]:
        return iris_resources()

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {WORLD_KV: {epoch.label: KVLabel() for epoch in context_epochs(self.config.max_rollout_horizon)}}

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        horizon = self._horizon(model_kwargs)
        configs: dict[str, ResourceReqConfig] = {
            WORLD_KV: KVReqConfig(
                needed_labels=[epoch.label for epoch in context_epochs(horizon)],
                prefix_cache=False,
            ),
        }
        # The reference samples unfiltered Categorical(logits=...), at T=1.
        for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
            configs[key] = SamplingReqConfig(
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                ignore_eos=True,
                penalize_prompt=False,
            )
        return configs

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        return {IMAGINE_WALK: imagination_graph(self.config.max_rollout_horizon)}

    def get_submodule(
        self,
        node_name: str,
        device="cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
        moe_parallel=None,
    ) -> torch.nn.Module | None:
        if node_name not in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER):
            raise ValueError(f"Unknown IRIS node: {node_name!r}")
        if self.skip_weight_loading:
            return None
        from mstar.model.iris.checkpoint import load_networks, resolve_checkpoint
        from mstar.model.iris.submodules import ImageSubmodule, WorldSubmodule

        if autocast_dtype not in (None, torch.float32):
            raise ValueError("IRIS's reference path requires FP32")
        device = str(torch.device(device))
        key = (node_name, device)
        if key not in self._submodules:
            if device not in self._networks:
                path = resolve_checkpoint(self._checkpoint_path, self.model_path_hf, self.cache_dir)
                self._networks[device] = load_networks(path, device)
            networks = self._networks[device]
            if node_name == WORLD_MODEL:
                module = WorldSubmodule(networks.world, self.config.max_rollout_horizon, self.config.max_batch_size)
            else:
                module = ImageSubmodule(
                    networks.tokenizer,
                    decode=node_name != IMAGE_ENCODER,
                    initial=node_name == INITIAL_DECODER,
                    max_batch_size=self.config.max_batch_size,
                )
            self._submodules[key] = module.eval()
        return self._submodules[key]
