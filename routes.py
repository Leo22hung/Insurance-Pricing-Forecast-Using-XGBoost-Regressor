from flask import Blueprint, request, jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Tạo một blueprint cho các routes mới
api = Blueprint('api', __name__)

# Xử lý các biến phân loại
def preprocess_data(df):
    categorical_cols = ['sex', 'smoker', 'region']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

# API phân tích EDA (Exploratory Data Analysis)
@api.route('/eda', methods=['POST'])
def eda():
    data = request.json  # Lấy dữ liệu JSON từ request
    df = pd.DataFrame(data)  # Chuyển đổi sang DataFrame

    # Phân tích dữ liệu cơ bản
    eda_results = {
        "columns": df.columns.tolist(),
        "description": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    return jsonify(eda_results)

# API huấn luyện mô hình
@api.route('/train', methods=['POST'])
def train():
    data = request.json  # Lấy dữ liệu JSON từ request
    df = pd.DataFrame(data)
    
    # Xử lý dữ liệu: Mã hóa các cột phân loại
    df_encoded = preprocess_data(df)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Linear Regression
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Lưu mô hình vào file pickle
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Lưu danh sách các cột đã mã hóa
    trained_columns = X_train.columns
    with open('models/trained_columns.pkl', 'wb') as f:
        pickle.dump(trained_columns, f)

    # Lưu tập kiểm tra để sử dụng cho đánh giá mô hình
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('models/test_data.csv', index=False)
    
    return jsonify({"message": "Model trained successfully!"})

# API dự đoán từ mô hình đã huấn luyện
@api.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Lấy dữ liệu JSON từ request
    df = pd.DataFrame([data])

    # Xử lý dữ liệu: Mã hóa các cột phân loại
    df_encoded = preprocess_data(df)

    # Tải mô hình từ file pickle
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Lấy các cột mà mô hình đã học
    with open('models/trained_columns.pkl', 'rb') as f:
        trained_columns = pickle.load(f)

    # Thêm các cột còn thiếu vào tập dữ liệu dự đoán với giá trị 0
    missing_cols = set(trained_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0

    # Đảm bảo rằng các cột trong tập dự đoán có thứ tự giống với tập huấn luyện
    df_encoded = df_encoded[trained_columns]

    # Dự đoán
    prediction = model.predict(df_encoded)
    return jsonify({'prediction': prediction.tolist()})

# API cập nhật mô hình (PUT)
@api.route('/update-model', methods=['PUT'])
def update():
    data = request.json  # Lấy dữ liệu JSON từ request
    df = pd.DataFrame(data)
    
    # Xử lý dữ liệu: Mã hóa các cột phân loại
    df_encoded = preprocess_data(df)
    
    # Chia dữ liệu để cập nhật mô hình
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    # Tải mô hình hiện tại
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Cập nhật mô hình với dữ liệu mới
    model.fit(X, y)

    # Lưu lại mô hình sau khi cập nhật
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({"message": "Model updated successfully!"})

# API đánh giá mô hình
@api.route('/evaluate', methods=['GET'])
def evaluate():
    # Đọc tập kiểm tra từ file CSV
    test_data = pd.read_csv('models/test_data.csv')
    
    # Chia dữ liệu kiểm tra thành X và y
    X_test = test_data.drop('charges', axis=1)
    y_true = test_data['charges']

    # Tải mô hình đã huấn luyện từ file pickle
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Tải các cột đã được mã hóa trong quá trình huấn luyện
    with open('models/trained_columns.pkl', 'rb') as f:
        trained_columns = pickle.load(f)

    # Thêm các cột còn thiếu vào tập dữ liệu kiểm tra với giá trị mặc định là 0
    missing_cols = set(trained_columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    # Đảm bảo thứ tự cột của tập kiểm tra khớp với các cột đã huấn luyện
    X_test = X_test[trained_columns]

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính toán độ chính xác của mô hình (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    
    return jsonify({"mean_squared_error": mse})

# API thống kê mô hình (GET)
@api.route('/stats', methods=['GET'])
def stats():
    # Ví dụ thống kê về mô hình
    stats = {
        "model_type": "Linear Regression",
        "total_params": 10,
        "training_samples": 1338  # Dữ liệu insurance.csv có 1338 mẫu
    }
    return jsonify(stats)

# Đăng ký tất cả các routes này vào Flask app
def register_routes(app):
    app.register_blueprint(api, url_prefix='/api')
