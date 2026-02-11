import onnxruntime as ort

MODEL = "models/phi3_fp16/phi3-small-8k-instruct-cuda-fp16.onnx"

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(
    MODEL,
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Providers:", sess.get_providers())
print("Inputs:")
for x in sess.get_inputs():
    print(" ", x.name, x.type, x.shape)
print("Outputs:")
for y in sess.get_outputs():
    print(" ", y.name, y.type, y.shape)
