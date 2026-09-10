import numpy as np
from onnxruntime import OrtValue

from homr.onnxruntime_utils import (
    create_inference_session,
    cuda_provider_options,
    should_accelerate_transformer,
    uses_fp16_models,
)
from homr.transformer.configs import Config
from homr.type_definitions import NDArray


class Encoder:
    def __init__(self, config: Config) -> None:
        if uses_fp16_models(config.use_gpu_inference):
            self.encoder, self.use_gpu = create_inference_session(
                config.filepaths.encoder_path_fp16,
                use_gpu_inference=True,
                component_name="TrOMR encoder",
                cuda_options=cuda_provider_options(cudnn_conv_algo_search="DEFAULT"),
            )
            self.fp16 = True

        else:
            self.encoder, self.use_gpu = create_inference_session(
                config.filepaths.encoder_path,
                use_gpu_inference=should_accelerate_transformer(
                    config.use_gpu_inference
                ),
                component_name="TrOMR encoder",
            )
            self.fp16 = False

        self.io_binding = self.encoder.io_binding()
        self.device_id = 0

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> list[OrtValue]:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", x.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", x.astype(np.float32))

        self.io_binding.bind_output("output", "cuda" if self.use_gpu else "cpu", self.device_id)
        self.encoder.run_with_iobinding(self.io_binding)
        return self.io_binding.get_outputs()[0].numpy()
