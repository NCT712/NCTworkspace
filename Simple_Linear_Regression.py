import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('data_training.csv')
X = df['x'].values
Y = df['y'].values
N = len(X)

# Predict function
def predict(x, w, b):
    return w * x + b
# Caculate MAE
def MAE(X, Y, w, b):
    error = 0
    for i in range(N):
        y_pred = predict(X[i], w, b)
        error += abs(y_pred - Y[i])
    return error / N
# Caculate MSE
def MSE(X, Y, w, b):
    error = 0
    for i in range(N):
        y_pred = predict(X[i], w, b)
        error += (y_pred - Y[i])**2
    return error / N
# Find the sign
def sign(val):
    if val > 0: return 1
    if val < 0: return -1
    return 0
# Compute gradient for MSE
def gradient_MSE(X, Y, w, b):
    dw = 0
    db = 0
    for i in range(N):
        y_pred = predict(X[i], w, b)
        dw += 2 * (y_pred - Y[i]) * X[i]
        db += 2 * (y_pred - Y[i])
    return dw / N, db / N
# Compute gradient for MAE
def gradient_MAE(X, Y, w, b):
    dw = 0
    db = 0
    for i in range(N):
        y_pred = predict(X[i], w, b)
        dw += sign(y_pred - Y[i]) * X[i]
        db += sign(y_pred - Y[i])
    return dw / N, db / N
# Update w and b
def update(w, b, dw, db, alpha):
    w_new = w - alpha * dw
    b_new = b - alpha * db
    return w_new, b_new
# Save history
def save(history, epoch, loss, w, b):
    history.append({
        'epoch': epoch,
        'loss': loss,
        'w': w,
        'b': b
    })
# Train function
def train(X, Y, alpha, epochs, type = ""):
    w = w_in
    b = b_in     
    history = []   

    for i in range(epochs):
        if type == "MSE":
            loss = MSE(X, Y, w, b)
            dw, db = gradient_MSE(X, Y, w, b)
        elif type == "MAE":
            loss = MAE(X, Y, w, b)
            dw, db = gradient_MAE(X, Y, w, b)
        
        w, b = update(w, b, dw, db, alpha)
        save(history, i, loss, w, b)       
    return w, b, history

# MAIN PROGRAM
w_in = 0
b_in = 0
t = input("Enter the training type (MSE/MAE): ")
a = float(input("Enter the learning rate (0.0005, 0.001, 0.005, 0.01): "))

w_final, b_final, history = train(X, Y, alpha = a, epochs = 10000, type = t)

# Summerizing
print("----- 5 Epoch đầu -----")
for i in history[:5]:
    print(f"Epoch {i['epoch'] + 1}: Loss = {i['loss']:.4f}, PT: y = {i['w']:.4f}x + {i['b']:.4f}")
        
print("\n----- 5 Epoch giữa -----")
mid = len(history) // 2
for i in history[mid-2:mid+3]:
    print(f"Epoch {i['epoch'] + 1}: Loss = {i['loss']:.4f}, PT: y = {i['w']:.4f}x + {i['b']:.4f}")
        
print("\n----- 5 Epoch cuối -----")
for i in history[-5:]:
    print(f"Epoch {i['epoch'] + 1}: Loss = {i['loss']:.4f}, PT: y = {i['w']:.4f}x + {i['b']:.4f}")


# Visualization
plt.scatter(X, Y, color='blue', label = 'Input Data')

y_final = [predict(x, w_final, b_final) for x in X]
plt.plot(X, y_final, color='red', label = 'Predicted Line')

plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
