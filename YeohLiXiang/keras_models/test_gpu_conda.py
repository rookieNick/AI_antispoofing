import tensorflow as tf
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)

# Check GPU availability
print("\n=== GPU Detection ===")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices found: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

# Check CUDA availability
print("\n=== CUDA Detection ===")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")

# Test GPU computation
print("\n=== GPU Test ===")
if gpus:
    print("Testing GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"GPU computation result: {c}")
        print("GPU test successful!")
else:
    print("No GPU found for testing")

# List all physical devices
print("\n=== All Physical Devices ===")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"  {device.device_type}: {device.name}") 