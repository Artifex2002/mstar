import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    DEFAULT_PARTITION,
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.model.base import ForwardPassArgs, Model, ProcessPromptOutput
from mstar.model.iris.config import HF_REPO, IMAGE_SHAPE, IRISConfig
from mstar.model.iris.resources import iris_resources
from mstar.model.multimodal import PromptPart
from mstar.utils.generation_kwargs import check_seed


class IRISModel(Model):
    """Supplied-action IRIS world-model request preparation."""

    PREPROCESS_ONLY_KWARGS = ("actions",)
    INTERNAL_KWARGS = ("iris_horizon",)
    LENGTH_KWARGS = ("iris_horizon",)

    def __init__(
        self,
        model_path_hf: str = HF_REPO,
        game: str = "Breakout",
        max_rollout_horizon: int = 20,
    ) -> None:
        self.config = IRISConfig(
            game=game,
            max_rollout_horizon=max_rollout_horizon,
        )
        self.model_path_hf = model_path_hf

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"actions"})

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        if input_modalities != ["image"]:
            raise ValueError("IRIS requires exactly one image input.")

        if output_modalities != ["video_frame", "scalar", "tensor"]:
            raise ValueError(
                "IRIS requires output_modalities="
                "['video_frame', 'scalar', 'tensor']."
            )

        if prompt not in (None, ""):
            raise ValueError("IRIS does not accept text conditioning.")

        if prompt_parts is not None and (
            len(prompt_parts) != 1
            or prompt_parts[0].modality != "image"
            or prompt_parts[0].index != 0
        ):
            raise ValueError("IRIS requires one image prompt part.")

        if kwargs.get("max_output_tokens") is not None:
            raise ValueError(
                "IRIS derives its horizon from actions; omit max_output_tokens."
            )

        check_seed(kwargs.get("seed"))

        tensors = tensors or {}
        images = tensors.get("image_inputs", [])

        if len(images) != 1 or any(
            items for key, items in tensors.items() if key != "image_inputs"
        ):
            raise ValueError("IRIS requires one image and no other media inputs.")

        image = images[0]
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or image.layout != torch.strided
            or image.dtype != torch.float32
            or tuple(image.shape) != IMAGE_SHAPE
        ):
            raise ValueError(
                "IRIS requires a CPU float32 image shaped [3, 64, 64]."
            )

        if (
            not torch.isfinite(image).all()
            or (image < 0).any()
            or (image > 1).any()
        ):
            raise ValueError("IRIS image values must be finite and in [0, 1].")

        actions = kwargs.get("actions")
        if not isinstance(actions, list):
            raise ValueError("IRIS actions must be a list of integer IDs.")

        horizon = len(actions)
        if not 1 <= horizon <= self.config.max_rollout_horizon:
            raise ValueError(
                "IRIS requires between 1 and "
                f"{self.config.max_rollout_horizon} actions."
            )

        if any(
            type(action) is not int
            or not 0 <= action < self.config.num_actions
            for action in actions
        ):
            raise ValueError(
                "IRIS action IDs must be integers in "
                f"[0, {self.config.num_actions})."
            )

        action_tensor = torch.tensor(actions, dtype=torch.int64)

        return ProcessPromptOutput(
            new_input_tensors={
                "image_inputs": [],
                "initial_image": [image.contiguous()],
                "actions": [action_tensor],
            },
            metadata={"iris_horizon": horizon},
        )

    def _horizon(self, model_kwargs: dict | None) -> int:
        horizon = (model_kwargs or {}).get("iris_horizon")
        if (
            type(horizon) is not int
            or not 1 <= horizon <= self.config.max_rollout_horizon
        ):
            raise ValueError(
                "IRIS requires a validated iris_horizon from process_prompt."
            )
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

        if input_modalities != ["image"]:
            raise ValueError("IRIS requires exactly one image input.")

        if output_modalities != ["video_frame", "scalar", "tensor"]:
            raise ValueError(
                "IRIS requires output_modalities="
                "['video_frame', 'scalar', 'tensor']."
            )

        horizon = self._horizon(model_kwargs)
        inputs = []

        for name, node, shape, dtype in (
            ("initial_image", "image_encoder", IMAGE_SHAPE, torch.float32),
            ("actions", "world_model", (horizon,), torch.int64),
        ):
            infos = input_signals.get(name, [])
            if (
                len(infos) != 1
                or tuple(infos[0].dims) != shape
                or infos[0].dtype != dtype
            ):
                raise ValueError(
                    f"IRIS {name} requires one {dtype} descriptor "
                    f"shaped {shape}."
                )

            inputs.append(
                GraphEdge(
                    next_node=node,
                    name=name,
                    tensor_info=list(infos),
                )
            )

        metadata = CurrentForwardConductorMetadata(
            graph_walk="imagine",
            is_prefill=True,
            input_modalities=list(input_modalities),
            output_modalities=list(output_modalities),
            kwargs={"iris_horizon": horizon},
        )

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=[
                info for edge in inputs for info in edge.tensor_info
            ],
            step_metadata={"iris_horizon": horizon},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if (
            partition_name != DEFAULT_PARTITION
            or partition_metadata.graph_walk != "imagine"
        ):
            raise ValueError(
                "IRIS can finish only its default partition's imagine walk."
            )

        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[],
            unpersist_tensors=[],
            request_done=True,
        )

    def get_max_output_tokens(self, **model_kwargs) -> int:
        return self._horizon(model_kwargs)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {"rollout_loop": self._horizon(model_kwargs)}

    def get_node_resources(self) -> list[NodeResourceSpec]:
        return iris_resources()
