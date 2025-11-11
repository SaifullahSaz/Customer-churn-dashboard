from utils import _load_artifacts

m, f, s = _load_artifacts()
print('MODEL_TYPE:', type(m))
print('HAS_FEATURE_LIST:', bool(f))
print('FEATURE_LIST_LEN:', len(f) if f else 0)
print('SCALER_TYPE:', type(s))
