import os

print("Current directory:", os.getcwd())
print("__file__:", __file__)
print("dirname(__file__):", os.path.dirname(__file__))
print("join(.., model, CNN_YeohLiXiang, cnn_pytorch.pth):", 
      os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang', 'cnn_pytorch.pth'))
print("abspath of above:", 
      os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang', 'cnn_pytorch.pth')))

# Check if the file exists
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang', 'cnn_pytorch.pth'))
print("File exists:", os.path.exists(model_path))

# Also check the actual model directory
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang'))
print("Model directory:", model_dir)
print("Model directory exists:", os.path.exists(model_dir))
if os.path.exists(model_dir):
    print("Contents of model directory:", os.listdir(model_dir))
