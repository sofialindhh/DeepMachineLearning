
import inspect
from tabicl import TabICLClassifier

model = TabICLClassifier(device="cpu")
model._load_model()

attn = model.model_.icl_predictor.tf_icl.blocks[0].attn
ssmax = attn.ssmax_layer

print("SSMax class:", type(ssmax))
print("SSMax forward signature:", inspect.signature(ssmax.forward))
print(inspect.getsource(type(ssmax).forward))
