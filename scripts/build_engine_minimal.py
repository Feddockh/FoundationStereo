# build_engine_minimal.py
import tensorrt as trt, sys

onnx_path, engine_path = sys.argv[1], sys.argv[2]

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
flags   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser  = trt.OnnxParser(network, logger)
config  = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB workspace
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Parse ONNX
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

# Set fixed shape profile if any input is dynamic
profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    name  = network.get_input(i).name
    shape = tuple(network.get_input(i).shape)
    # Replace -1 (dynamic) with 1
    fixed = tuple(1 if d == -1 else d for d in shape)
    profile.set_shape(name, fixed, fixed, fixed)
config.add_optimization_profile(profile)

# Build engine
serialized = builder.build_serialized_network(network, config)
assert serialized, "Build failed"
with open(engine_path, "wb") as f:
    f.write(serialized)
print("Saved:", engine_path)
