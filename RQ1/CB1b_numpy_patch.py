# ============================================================
# BLOCK 1b: NumPy 2.0 Patch for imagecorruptions fog/frost
# np.float_ was removed in NumPy 2.0 — patch it back in
# ============================================================

import numpy as np

if not hasattr(np, 'float_'):
    np.float_ = np.float64
    print("✅ np.float_ patch applied")
else:
    print("✅ np.float_ already exists — no patch needed")

# Verify fog now works on a test image
from imagecorruptions import corrupt
test = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
try:
    _ = corrupt(test, corruption_name='fog', severity=1)
    print("✅ Fog corruption verified working")
except Exception as e:
    print(f"❌ Fog still failing: {e}")
